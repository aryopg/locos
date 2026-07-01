#!/usr/bin/env python3
"""Detect retrieval heads for any HuggingFace model.

Architecture-agnostic reimplementation of the retrieval head detection algorithm
from Wu et al. (2024) / github.com/nightdessert/Retrieval_Head. Uses standard
HuggingFace transformers APIs (output_attentions=True with eager attention)
instead of custom model files.

Supports two probing datasets:
  - NIAH (default): Wu et al.'s Needle-in-a-Haystack with high lexical overlap
  - NoLiMa: Adobe Research's No Literal Matching benchmark (ICML 2025)

Produces a JSON file compatible with locos_eval's load_retrieval_heads().

Usage:
    # Download haystack data first (one-time)
    python locos/download_haystack_data.py
    python locos/download_haystack_data.py --dataset nolima

    # Quick test with NIAH (default)
    python -m locos.detectors.behavioral \\
        --model Qwen/Qwen3.5-2B --max-length 2000 --num-lengths 3

    # Detection with NoLiMa (minimal lexical overlap probing)
    python -m locos.detectors.behavioral \\
        --model Qwen/Qwen3.5-2B --dataset nolima --max-length 8000

    # Full detection (matches original defaults)
    python -m locos.detectors.behavioral \\
        --model Qwen/Qwen3.5-2B --min-length 1000 --max-length 50000

    # Resume from checkpoint
    python -m locos.detectors.behavioral \\
        --model Qwen/Qwen3.5-2B --resume

Requires: GPU and the project runtime dependencies.
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `locos.*` imports work
# regardless of the working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import gc
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rouge_score import rouge_scorer

from locos.utils.common import (
    build_context_depth_ranges,
    extract_model_config,
    format_duration,
)
from locos.utils.common import load_checkpoint as _load_checkpoint_generic
from locos.utils.common import save_checkpoint as _save_checkpoint_generic
from locos.utils.datasets import (
    load_niah_haystack_texts,
    load_niah_needles,
)
from locos.utils.model_utils import (
    detect_thinking_tokens,
    format_prompt_with_chat_template,
    get_input_device,
    get_stop_token_ids,
    set_model_attn_impl,
    strip_thinking_content,
    tokenizer_adds_bos,
)
from locos.utils.needle_utils import (
    build_period_token_positions,
    find_needle_idx_from_tokens,
    get_period_tokens,
    insert_needle_tokens,
)

console = Console()
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


# ---------------------------------------------------------------------------
# Checkpoint wrappers (adapt generic save/load to NIAH-specific data structure)
# ---------------------------------------------------------------------------


def save_checkpoint(head_counter, completed_trials, checkpoint_path):
    _save_checkpoint_generic(
        {"head_counter": dict(head_counter), "completed_trials": completed_trials},
        checkpoint_path,
    )


def load_checkpoint(checkpoint_path):
    data = _load_checkpoint_generic(checkpoint_path)
    if data is None:
        return defaultdict(list), []
    head_counter = defaultdict(list)
    for k, v in data["head_counter"].items():
        head_counter[k] = v
    return head_counter, data["completed_trials"]


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------


def detect_single_trial(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    needle_start: int,
    needle_end: int,
    num_layers: int,
    num_heads: int,
    prefill_attn_impl: str,
    max_decode_steps: int = 50,
    newline_token_id: int | None = None,
    stop_token_ids: set[int] | None = None,
    think_start_id: int | None = None,
    think_end_id: int | None = None,
    max_thinking_tokens: int = 0,
) -> tuple[list[list[float]], str]:
    """Run one needle-in-haystack detection trial.

    Faithful to the original's decode + retrieval_calculate methods.
    Supports multi-GPU via device_map="auto" (attention tensors may live
    on different devices per layer).

    For reasoning models that produce ``<think>…</think>`` blocks: when
    *think_start_id* and *think_end_id* are provided, the decode loop
    skips retrieval scoring inside thinking blocks, does not stop on
    newlines within them, and enforces a separate *max_thinking_tokens*
    budget so the answer still has room to be generated.

    Args:
        input_ids: Full prompt token IDs, shape (1, seq_len).
        needle_start, needle_end: Token positions of the needle in input_ids.
        think_start_id, think_end_id: Token IDs for ``<think>``/``</think>``
            markers (None to disable thinking handling).
        max_thinking_tokens: Maximum tokens allowed inside thinking blocks.

    Returns:
        (retrieval_scores, generated_text) where retrieval_scores is
        [num_layers][num_heads] floats.  *generated_text* excludes any
        thinking content.
    """
    prompt_ids = input_ids[0].cpu()  # keep on CPU for token matching
    prompt_ids_np = prompt_ids.numpy()
    device = get_input_device(model)

    # Initialize scores: retrieval_score[layer][head] = 0.0
    retrieval_score = np.zeros((num_layers, num_heads), dtype=np.float32)

    # Move input to model's input device (first param device for device_map="auto")
    input_ids = input_ids.to(device)

    # Prefill: all tokens except last (no attention extraction needed)
    # Use a memory-friendlier attention backend if requested, then switch
    # back to eager before decode (decode needs output_attentions=True).
    switched_for_prefill = False
    if prefill_attn_impl != "eager":
        switched_for_prefill = set_model_attn_impl(model, prefill_attn_impl)

    with torch.inference_mode():
        try:
            prefill_out = model(
                input_ids=input_ids[:, :-1],
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
        except Exception as err:
            # Fallback once if alternate prefill backend is unsupported.
            if switched_for_prefill:
                set_model_attn_impl(model, "eager")
                prefill_out = model(
                    input_ids=input_ids[:, :-1],
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True,
                )
                console.print(
                    f"[yellow]Prefill backend '{prefill_attn_impl}' failed; "
                    f"fell back to eager. ({type(err).__name__})[/yellow]"
                )
            else:
                raise
        past_kv = prefill_out.past_key_values

    # Ensure eager mode for decode attention extraction.
    if prefill_attn_impl != "eager":
        set_model_attn_impl(model, "eager")

    # Autoregressive decode with attention extraction
    current_token = input_ids[:, -1:].clone()  # (1, 1)
    generated_ids = []  # content tokens only (excludes thinking)

    if newline_token_id is None:
        newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

    has_thinking = think_start_id is not None and think_end_id is not None
    in_thinking = False
    thinking_token_count = 0
    content_steps = 0

    with torch.inference_mode():
        for _ in range(max_decode_steps + max_thinking_tokens):
            outputs = model(
                input_ids=current_token,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=not in_thinking,
                return_dict=True,
            )
            past_kv = outputs.past_key_values

            # Greedy next token (logits may be on last GPU, use .item() to avoid device issues)
            next_token_id_val = outputs.logits[0, -1].argmax().item()

            # --- Thinking block handling ---
            if has_thinking:
                if next_token_id_val == think_start_id:
                    in_thinking = True
                    thinking_token_count += 1
                    current_token[0, 0] = next_token_id_val
                    continue
                if next_token_id_val == think_end_id:
                    in_thinking = False
                    thinking_token_count += 1
                    current_token[0, 0] = next_token_id_val
                    continue
                if in_thinking:
                    thinking_token_count += 1
                    if thinking_token_count > max_thinking_tokens:
                        break
                    current_token[0, 0] = next_token_id_val
                    continue

            # --- Stop checks (before appending, so stop tokens stay out of generated_ids) ---
            step_token = tokenizer.convert_ids_to_tokens(next_token_id_val)
            if step_token == "<0x0A>" or next_token_id_val == newline_token_id:
                break
            if stop_token_ids and next_token_id_val in stop_token_ids:
                break

            # --- Content token ---
            generated_ids.append(next_token_id_val)
            content_steps += 1

            # Retrieval scoring (vectorized to minimize GPU→CPU syncs)
            # outputs.attentions: tuple of num_layers tensors
            # Each: (batch=1, num_heads, query_len=1, key_len)
            # Do argmax on GPU per layer (cheap), transfer only the small
            # index tensor to CPU. This replaces num_layers × num_heads
            # individual .item() calls with num_layers small transfers.
            argmax_positions = torch.stack(
                [a[0, :, -1, :].argmax(dim=-1).cpu() for a in outputs.attentions]
            ).numpy()  # (num_layers, num_heads)

            # Vectorized needle-span + token-match check on CPU
            in_span = (argmax_positions >= needle_start) & (argmax_positions < needle_end)
            if in_span.any():
                span_positions = argmax_positions[in_span]
                token_match = prompt_ids_np[span_positions] == next_token_id_val
                score_increment = 1.0 / (needle_end - needle_start)
                layer_idxs, head_idxs = np.where(in_span)
                if token_match.any():
                    retrieval_score[
                        layer_idxs[token_match],
                        head_idxs[token_match],
                    ] += score_increment

            if content_steps >= max_decode_steps:
                break

            current_token[0, 0] = next_token_id_val

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return retrieval_score.tolist(), generated_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Detect retrieval heads for any HuggingFace model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: retrieval_heads/<model_name>[_<dataset>].json)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="niah",
        choices=["niah", "nolima"],
        help="Probing dataset: niah (Wu et al.) or nolima (Adobe Research)",
    )
    parser.add_argument(
        "--haystack-dir", type=Path, default=None, help="Directory with NIAH data (default: data/haystack_for_detect)"
    )
    parser.add_argument(
        "--nolima-dir", type=Path, default=Path("data/nolima"), help="Directory with NoLiMa data (default: data/nolima)"
    )
    parser.add_argument(
        "--question-type",
        type=str,
        default="onehop",
        choices=["onehop", "twohop", "twohop2"],
        help="NoLiMa question type (default: onehop)",
    )
    parser.add_argument(
        "--nolima-variant",
        type=str,
        default="needle_set",
        choices=["needle_set", "needle_set_hard"],
        help="NoLiMa needle set variant (default: needle_set)",
    )
    parser.add_argument(
        "--max-characters-per-entry",
        type=int,
        default=1,
        help=(
            "Max character names per NoLiMa entry (default: 1). Each NoLiMa "
            "entry has 10 character names (e.g., Yuki, Stuart, Katie...) that "
            "get plugged into the needle template. More characters = more "
            "trials with different names but same needle structure. "
            "1 is sufficient for head detection; increase for ranking stability."
        ),
    )
    parser.add_argument("--min-length", type=int, default=1000, help="Minimum context length in tokens")
    parser.add_argument("--max-length", type=int, default=50000, help="Maximum context length in tokens")
    parser.add_argument(
        "--num-lengths", type=int, default=20, help="Number of context length intervals (original default: 20)"
    )
    parser.add_argument(
        "--num-depths", type=int, default=10, help="Number of document depth intervals (original default: 10)"
    )
    parser.add_argument("--max-decode-steps", type=int, default=50)
    parser.add_argument(
        "--debug-trials",
        type=int,
        default=0,
        help="Print raw generated text and ROUGE for the first N trials (for diagnosing model output)",
    )
    parser.add_argument(
        "--rouge-threshold", type=float, default=50.0, help="Minimum ROUGE-1 recall * 100 to accept a trial"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--prefill-attn-impl",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help=("Attention backend for prefill only. Decode still uses eager so " "output_attentions=True works."),
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        help="Transformers device_map strategy for model placement",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--use-hardcoded-periods",
        action="store_true",
        dest="use_hardcoded_periods",
        default=False,
        help="Use hardcoded period tokens for known models instead of computing from tokenizer",
    )
    parser.add_argument(
        "--force-llama2-periods",
        action="store_true",
        help="[DEBUG] Force Llama-2 period tokens regardless of model (for testing)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-thinking-tokens",
        type=int,
        default=0,
        help=(
            "Max tokens to allow inside <think>…</think> blocks during decode. "
            "Set >0 for reasoning models (e.g. 512). 0 disables thinking handling."
        ),
    )
    parser.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Wrap the prompt in the tokenizer's chat template before detection. "
            "Required for models that need structured prompts. "
            "Needle positions are re-located after re-tokenization."
        ),
    )
    parser.add_argument(
        "--prompt-suffix",
        type=str,
        default=None,
        help=(
            "String to append after the prompt (post chat-template if enabled). "
            "Tokenized and appended to the token sequence before detection."
        ),
    )
    args = parser.parse_args()

    # Resolve output path
    model_short_name = args.model.split("/")[-1]
    dataset_suffix = f"_{args.dataset}" if args.dataset != "niah" else ""
    if args.output is None:
        output_path = Path("retrieval_heads") / f"{model_short_name}{dataset_suffix}.json"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    # Resolve haystack dir default based on dataset
    if args.haystack_dir is None:
        args.haystack_dir = Path("data/haystack_for_detect")

    console.rule("[bold]Retrieval Head Detection[/bold]")

    # Build test grid (faithful to original: np.linspace)
    context_lengths, depth_percents = build_context_depth_ranges(
        args.min_length, args.max_length, args.num_lengths, args.num_depths
    )

    # Build dataset trials
    from locos.utils.datasets import build_niah_dataset, build_nolima_dataset

    if args.dataset == "niah":
        needles = load_niah_needles(args.haystack_dir)
        haystack_texts = load_niah_haystack_texts(args.haystack_dir, args.max_length)
        assert len(haystack_texts) == len(needles), f"Expected {len(needles)} haystack parts, got {len(haystack_texts)}"
        dataset_trials = build_niah_dataset(
            args.haystack_dir,
            context_lengths,
            depth_percents,
            max_tokens=args.max_length,
        )
        dataset_info = (
            f"{len(needles)} needles, "
            f"{args.num_lengths} lengths x {args.num_depths} depths x {len(needles)} needles"
        )
    elif args.dataset == "nolima":
        from locos.utils.datasets import load_nolima_needle_set

        nolima_entries = load_nolima_needle_set(args.nolima_dir, args.nolima_variant)
        dataset_trials = build_nolima_dataset(
            args.nolima_dir,
            context_lengths,
            depth_percents,
            question_type=args.question_type,
            variant=args.nolima_variant,
            max_tokens=args.max_length,
            max_characters_per_entry=args.max_characters_per_entry,
            seed=args.seed,
        )
        dataset_info = (
            f"{len(nolima_entries)} entries ({args.nolima_variant}), "
            f"question_type={args.question_type}, "
            f"max_chars/entry={args.max_characters_per_entry}"
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    config_table = Table(title="Configuration", show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="bold")
    config_table.add_column("Value")
    config_table.add_row("Model", args.model)
    config_table.add_row("Dataset", args.dataset)
    config_table.add_row("Output", str(output_path))
    config_table.add_row("Context range", f"{args.min_length} – {args.max_length} tokens")
    config_table.add_row("Dataset info", dataset_info)
    config_table.add_row("Total trials", str(len(dataset_trials)))
    config_table.add_row("Prefill backend", f"{args.prefill_attn_impl} (decode uses eager)")
    config_table.add_row("Chat template", str(args.chat_template))
    if args.prompt_suffix:
        config_table.add_row("Prompt suffix", args.prompt_suffix)
    console.print(config_table)

    # Load model
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    console.print(f"\nLoading model ({args.dtype}) ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
    tokenizer_has_bos = tokenizer_adds_bos(tokenizer)
    think_start_id, think_end_id = detect_thinking_tokens(tokenizer)
    if think_start_id is not None and args.max_thinking_tokens > 0:
        console.print(
            f"[cyan]Thinking tokens detected:[/cyan] "
            f"start={think_start_id}, end={think_end_id}, "
            f"budget={args.max_thinking_tokens}"
        )
    elif think_start_id is not None and args.max_thinking_tokens == 0:
        console.print(
            "[yellow]Thinking tokens detected but --max-thinking-tokens=0. " "Set >0 for reasoning models.[/yellow]"
        )
    if args.chat_template:
        has_template = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template
        if has_template:
            console.print("[cyan]Chat template enabled:[/cyan] prompts will be wrapped before detection")
        else:
            console.print("[yellow]--chat-template requested but tokenizer has no chat template[/yellow]")
    prompt_suffix_ids: list[int] = []
    if args.prompt_suffix:
        prompt_suffix_ids = tokenizer.encode(args.prompt_suffix, add_special_tokens=False)
        decoded_back = tokenizer.convert_ids_to_tokens(prompt_suffix_ids)
        console.print(
            f"[cyan]Prompt suffix:[/cyan] {args.prompt_suffix!r} → " f"{len(prompt_suffix_ids)} tokens {decoded_back}"
        )
    period_tokens = get_period_tokens(
        args.model,
        tokenizer,
        args.use_hardcoded_periods,
        args.force_llama2_periods,
    )
    haystack_token_cache: dict[str, list[int]] = {}
    haystack_period_cache: dict[str, list[int]] = {}
    needle_token_cache: dict[str, list[int]] = {}
    question_token_cache: dict[str, list[int]] = {}
    answer_token_cache: dict[str, list[int]] = {}
    template_split_cache: dict[str, tuple[str, str]] = {}

    visible_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Prefer multi-GPU-friendly placement when multiple devices are visible.
    # `auto` may keep the entire model on GPU0 if weights fit there.
    device_map = args.device_map

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="eager",  # Required for output_attentions=True
        trust_remote_code=True,
    ).eval()

    num_layers, num_heads = extract_model_config(model)
    stop_token_ids = get_stop_token_ids(tokenizer, model)
    console.print(f"[dim]Stop token IDs ({len(stop_token_ids)}): {stop_token_ids}[/dim]")

    # Report model & device info
    model_table = Table(title="Model", show_header=False, box=None, padding=(0, 2))
    model_table.add_column("Key", style="bold")
    model_table.add_column("Value")
    model_table.add_row("Architecture", f"{num_layers} layers, {num_heads} heads")
    model_table.add_row("Dtype", args.dtype)
    model_table.add_row("Device map strategy", device_map)
    model_table.add_row("CUDA_VISIBLE_DEVICES", visible_cuda)
    model_table.add_row("CUDA devices", str(gpu_count))
    if hasattr(model, "hf_device_map"):
        per_device = defaultdict(int)
        for _, dev in model.hf_device_map.items():
            per_device[str(dev)] += 1
        devices_used = sorted(per_device.keys())
        detail = ", ".join(f"{dev}: {count} modules" for dev, count in sorted(per_device.items()))
        model_table.add_row("Placement", f"{len(devices_used)} device(s): {detail}")
    elif torch.cuda.is_available():
        model_table.add_row("Device", str(get_input_device(model)))
    console.print(model_table)
    console.print("[green]Model loaded[/green]\n")

    # Resume or fresh start
    if args.resume:
        head_counter, completed_trials = load_checkpoint(checkpoint_path)
        console.print(f"Resumed: {len(completed_trials)} trials already completed")
    else:
        head_counter = defaultdict(list)
        completed_trials = []

    completed_set = set(t if isinstance(t, str) else tuple(t) for t in completed_trials)

    # Filter to remaining trials
    trials = [t for t in dataset_trials if t.trial_id not in completed_set]
    total_trials = len(dataset_trials)
    console.print(f"Trials: {total_trials} total, {len(trials)} remaining\n")

    if not trials:
        console.print("[yellow]All trials already completed.[/yellow]")
    else:
        num_passed = 0
        loop_start_time = time.time()
        processed_this_run = 0
        oom_count = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[status]}[/dim]"),
            console=console,
        ) as progress:
            ptask = progress.add_task(
                "Detecting retrieval heads",
                total=len(trials),
                status="",
            )

            for trial in trials:
                elapsed_s = time.time() - loop_start_time
                if processed_this_run > 0:
                    avg_s = elapsed_s / processed_this_run
                    eta_s = (len(trials) - processed_this_run) * avg_s
                    timing_status = f"elapsed={format_duration(elapsed_s)} eta={format_duration(eta_s)}"
                else:
                    timing_status = "elapsed=00:00 eta=estimating..."

                progress.update(
                    ptask,
                    status=(
                        f"ctx={trial.context_length} depth={trial.depth_percent}% "
                        f"id={trial.trial_id[:30]} | {timing_status}"
                    ),
                )

                # Build context with needle inserted
                haystack_tokens = haystack_token_cache.get(trial.haystack_text)
                if haystack_tokens is None:
                    haystack_tokens = tokenizer.encode(trial.haystack_text, add_special_tokens=False)
                    haystack_token_cache[trial.haystack_text] = haystack_tokens
                    haystack_period_cache[trial.haystack_text] = build_period_token_positions(
                        haystack_tokens, period_tokens
                    )
                needle_tokens = needle_token_cache.get(trial.needle_text)
                if needle_tokens is None:
                    needle_tokens = tokenizer.encode(trial.needle_text, add_special_tokens=False)
                    needle_token_cache[trial.needle_text] = needle_tokens

                context_tokens, needle_start, needle_end = insert_needle_tokens(
                    haystack_tokens,
                    needle_tokens,
                    trial.context_length,
                    trial.depth_percent,
                    period_positions=haystack_period_cache[trial.haystack_text],
                )

                assert (
                    needle_start >= 0 and needle_end > needle_start
                ), f"Invalid needle position: start={needle_start}, end={needle_end}"

                if trial.prompt_template is not None and "{haystack}" in trial.prompt_template:
                    # NoLiMa: use the full prompt template with {haystack} filled in.
                    # Decode the context tokens back to text, fill into template,
                    # then re-tokenize the full prompt. This preserves NoLiMa's
                    # intended prompt framing (system prompt, instructions, etc.).
                    template_parts = template_split_cache.get(trial.prompt_template)
                    if template_parts is None:
                        template_parts = trial.prompt_template.split("{haystack}", 1)
                        assert (
                            len(template_parts) == 2
                        ), "Prompt template must contain exactly one {haystack} placeholder"
                        template_split_cache[trial.prompt_template] = (template_parts[0], template_parts[1])
                    haystack_with_needle = tokenizer.decode(context_tokens, skip_special_tokens=True)
                    full_prompt = template_parts[0] + haystack_with_needle + template_parts[1]
                    full_tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
                    # Re-locate needle after re-tokenization (positions may shift)
                    needle_start, needle_end = find_needle_idx_from_tokens(torch.tensor(full_tokens), needle_tokens)
                    if needle_start == -1:
                        console.print(
                            f"[yellow]Warning: needle not found after template fill "
                            f"({trial.trial_id}). Skipping.[/yellow]"
                        )
                        completed_trials.append(trial.trial_id)
                        processed_this_run += 1
                        progress.advance(ptask)
                        continue
                else:
                    # NIAH: simple concatenation (context + question)
                    question_tokens = question_token_cache.get(trial.question)
                    if question_tokens is None:
                        question_tokens = tokenizer.encode(trial.question, add_special_tokens=False)
                        question_token_cache[trial.question] = question_tokens
                    full_tokens = context_tokens + question_tokens

                # Optionally wrap in chat template for models that need
                # structured prompts to produce sensible output.
                chat_template_applied = False
                if args.chat_template:
                    prompt_text = tokenizer.decode(full_tokens, skip_special_tokens=True)
                    formatted = format_prompt_with_chat_template(tokenizer, prompt_text)
                    if formatted is not None:
                        full_tokens = tokenizer.encode(formatted, add_special_tokens=False)
                        needle_start, needle_end = find_needle_idx_from_tokens(torch.tensor(full_tokens), needle_tokens)
                        if needle_start == -1:
                            console.print(
                                f"[yellow]Warning: needle not found after chat template "
                                f"({trial.trial_id}). Skipping.[/yellow]"
                            )
                            completed_trials.append(trial.trial_id)
                            processed_this_run += 1
                            progress.advance(ptask)
                            continue
                        chat_template_applied = True

                # Tokenize as model expects (with BOS if needed).
                # Skip when chat template already includes special tokens.
                if not chat_template_applied and tokenizer_has_bos:
                    full_tokens = [tokenizer.bos_token_id, *full_tokens]
                    needle_start += 1
                    needle_end += 1

                # Append prompt suffix when the caller needs model-specific
                # generation-control tokens.
                if prompt_suffix_ids:
                    full_tokens = full_tokens + prompt_suffix_ids

                input_ids = torch.tensor([full_tokens])

                # Determine needle span for retrieval scoring.
                # For the NIAH path (no template re-tokenization), needle_start/
                # needle_end from insert_needle() are already correct — use them
                # directly. This avoids the 90% overlap heuristic which can
                # false-match short answer texts (e.g., 1-2 token character names).
                #
                # For the NoLiMa template path, we already re-located the needle
                # span above via find_needle_idx after re-tokenization.
                #
                # For NIAH, we still refine to the real_needle (answer) span
                # within the broader needle sentence, using the original heuristic.
                if trial.nolima_entry_id is None:
                    # NIAH: find the real_needle (answer tokens) within the
                    # already-known needle span. Use find_needle_idx for this
                    # since needle_start/end point to the full needle sentence,
                    # but scoring should use the answer span specifically.
                    answer_tokens = answer_token_cache.get(trial.answer_text)
                    if answer_tokens is None:
                        answer_tokens = tokenizer.encode(trial.answer_text, add_special_tokens=False)
                        answer_token_cache[trial.answer_text] = answer_tokens
                    found_start, found_end = find_needle_idx_from_tokens(input_ids[0], answer_tokens)
                    if found_start != -1:
                        needle_start, needle_end = found_start, found_end
                    # If not found, keep the broader needle span from insert_needle
                # else: NoLiMa — needle_start/needle_end already set above
                #   (either from insert_needle for simple path, or find_needle_idx
                #   for template path). Use the full needle span for scoring, since
                #   the answer (character name) is embedded within it.

                if needle_start < 0 or needle_end <= needle_start:
                    console.print(f"[yellow]Warning: invalid needle span " f"({trial.trial_id}). Skipping.[/yellow]")
                    completed_trials.append(trial.trial_id)
                    processed_this_run += 1
                    progress.advance(ptask)
                    continue

                # Run detection
                try:
                    retrieval_scores, generated_text = detect_single_trial(
                        model,
                        tokenizer,
                        input_ids,
                        needle_start,
                        needle_end,
                        num_layers,
                        num_heads,
                        args.prefill_attn_impl,
                        args.max_decode_steps,
                        newline_token_id,
                        stop_token_ids=stop_token_ids,
                        think_start_id=think_start_id if args.max_thinking_tokens > 0 else None,
                        think_end_id=think_end_id if args.max_thinking_tokens > 0 else None,
                        max_thinking_tokens=args.max_thinking_tokens,
                    )
                except torch.cuda.OutOfMemoryError:
                    console.print(f"[red]OOM at ctx={trial.context_length}. Skipping.[/red]")
                    oom_count += 1
                    torch.cuda.empty_cache()
                    gc.collect()
                    completed_trials.append(trial.trial_id)
                    processed_this_run += 1
                    progress.advance(ptask)
                    continue

                # ROUGE gate (strip any residual thinking content as safety net)
                rouge_text = strip_thinking_content(generated_text) if args.max_thinking_tokens > 0 else generated_text
                rouge_result = scorer.score(trial.answer_text, rouge_text)["rouge1"].recall * 100

                if args.debug_trials > 0 and processed_this_run < args.debug_trials:
                    progress.console.print(
                        f"\n[bold]--- Debug trial {processed_this_run + 1}/{args.debug_trials} ---[/bold]\n"
                        f"  [dim]trial_id:[/dim] {trial.trial_id}\n"
                        f"  [dim]ctx_len:[/dim]  {trial.context_length}  "
                        f"[dim]depth:[/dim] {trial.depth_percent}%\n"
                        f"  [dim]answer:[/dim]   {trial.answer_text!r}\n"
                        f"  [dim]generated (raw, repr):[/dim]\n    {generated_text!r}\n"
                        f"  [dim]ROUGE-1 recall:[/dim] {rouge_result:.1f}  "
                        f"({'[green]PASS[/green]' if rouge_result > args.rouge_threshold else '[red]FAIL[/red]'})"
                    )

                if rouge_result > args.rouge_threshold:
                    num_passed += 1
                    for layer_idx in range(num_layers):
                        for head_idx in range(num_heads):
                            key = f"{layer_idx}-{head_idx}"
                            head_counter[key].append(retrieval_scores[layer_idx][head_idx])

                completed_trials.append(trial.trial_id)
                processed_this_run += 1
                progress.advance(ptask)

                elapsed_s = time.time() - loop_start_time
                avg_s = elapsed_s / max(1, processed_this_run)
                remaining = len(trials) - processed_this_run
                eta_s = remaining * avg_s
                progress.update(
                    ptask,
                    status=(
                        f"ctx={trial.context_length} depth={trial.depth_percent}% "
                        f"id={trial.trial_id[:30]} | "
                        f"elapsed={format_duration(elapsed_s)} eta={format_duration(eta_s)}"
                    ),
                )

                # Checkpoint every 10 trials
                if len(completed_trials) % 10 == 0:
                    save_checkpoint(head_counter, completed_trials, checkpoint_path)

        total_elapsed_s = time.time() - loop_start_time
        avg_trial_s = total_elapsed_s / max(1, processed_this_run)
        throughput = processed_this_run / max(1e-9, total_elapsed_s)
        console.print(f"\n[green]{num_passed}[/green] / {len(trials)} trials passed ROUGE gate")
        timing_table = Table(title="Timing Summary")
        timing_table.add_column("Metric")
        timing_table.add_column("Value", justify="right")
        timing_table.add_row("Trials processed (this run)", str(processed_this_run))
        timing_table.add_row("OOM skips", str(oom_count))
        timing_table.add_row("Elapsed", format_duration(total_elapsed_s))
        timing_table.add_row("Avg time / trial", f"{avg_trial_s:.2f}s")
        timing_table.add_row("Throughput", f"{throughput:.3f} trials/s")
        console.print(timing_table)

    # Save final results
    # Ensure all layer-head combos are present (even if empty)
    result = {}
    for layer in range(num_layers):
        for head in range(num_heads):
            key = f"{layer}-{head}"
            result[key] = head_counter.get(key, [])

    output_path.write_text(json.dumps(result))
    console.print(f"\n[bold green]Saved retrieval head scores to {output_path}[/bold green]")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Print top retrieval heads
    scored = [(key, float(np.mean(scores)) if scores else 0.0) for key, scores in result.items()]
    scored.sort(key=lambda x: x[1], reverse=True)

    table = Table(title="Top 20 Retrieval Heads")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Layer-Head")
    table.add_column("Mean Score", justify="right")
    table.add_column("Trials", justify="right")

    for i, (key, mean_score) in enumerate(scored[:20]):
        table.add_row(
            str(i + 1),
            key,
            f"{mean_score:.4f}",
            str(len(result[key])),
        )
    console.print(table)

    # Validate output is loadable
    sys.path.insert(0, ".")
    from locos_eval.retrieval_heads import load_retrieval_heads

    heads = load_retrieval_heads(str(output_path), num_heads=10)
    console.print(f"\nValidation: load_retrieval_heads() returned {len(heads)} heads")
    console.print(f"Top 10: {heads}")


if __name__ == "__main__":
    main()
