#!/usr/bin/env python3
"""Attention-only spatial-contrast retrieval head detector.

Same spatial contrast as LOCOS (logit_contrib.py) but drops the OV
projection. For each answer decode step t and head (l, h):

    phi_{t,j}^{(l,h)} = alpha_{t,j}^{(l,h)}
    Phi+    = sum_{j in needle} phi_j
    Phi-    = (needle_len / off_needle_len) * sum_{j not in needle} phi_j
    S^tau   = (1/|T_ans|) sum_t (Phi+_t - Phi-_t)

This is the controlled ablation requested by reviewers: if LOCOS still
dominates this baseline, the OV projection is doing real work; if not,
the claim narrows to attention-pattern detectors.
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gc
import json
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rouge_score import rouge_scorer

from locos.detectors.contrastive import identify_answer_steps
from locos.utils.common import (
    build_context_depth_ranges,
    extract_model_config,
    format_duration,
    load_checkpoint,
    save_checkpoint,
)
from locos.utils.datasets import (
    build_niah_dataset,
    build_nolima_dataset,
    load_nolima_needle_set,
    stratified_sample,
)
from locos.utils.model_utils import (
    detect_thinking_tokens,
    get_input_device,
    get_stop_token_ids,
    set_model_attn_impl,
    strip_thinking_content,
    tokenizer_adds_bos,
)
from locos.utils.needle_utils import (
    build_period_token_positions,
    build_tracked_prompt,
    get_period_tokens,
)
from locos.utils.needle_utils import load_needles as load_niah_needles

console = Console()
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def compute_attention_spatial_per_step(
    attn_weights: torch.Tensor,
    needle_start: int,
    needle_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-head needle vs off-needle attention with span-length rescaling.

    Args:
        attn_weights: (num_heads, key_len) attention weights for one layer
            at one decode step. Rows are expected to sum to ~1 (softmax output)
            but this is not enforced — the kernel sums whatever is passed.
        needle_start, needle_end: Needle span [start, end) in key positions.

    Returns:
        (phi_needle, phi_off_rescaled), each shape (num_heads,) as np.float32.
        phi_needle: sum of attention over needle positions.
        phi_off_rescaled: sum over off-needle positions, multiplied by
            (needle_len / off_needle_len) so both represent the contribution
            of a span of length needle_len.
    """
    assert attn_weights.ndim == 2, f"Expected (num_heads, key_len), got {attn_weights.shape}"
    key_len = attn_weights.shape[1]
    needle_len = needle_end - needle_start
    off_needle_len = key_len - needle_len
    assert needle_len > 0, f"needle_len must be > 0 (got start={needle_start}, end={needle_end})"
    assert off_needle_len > 0, f"No off-needle positions: key_len={key_len}, needle_len={needle_len}"

    phi_needle = attn_weights[:, needle_start:needle_end].sum(dim=-1)
    phi_total = attn_weights.sum(dim=-1)
    phi_off = phi_total - phi_needle

    scale = needle_len / off_needle_len
    phi_off_rescaled = phi_off * scale

    return (
        phi_needle.float().cpu().numpy(),
        phi_off_rescaled.float().cpu().numpy(),
    )


@dataclass
class AttentionSpatialTrialResult:
    """Result of a single attention-only spatial-contrast detection trial."""

    S_tau: np.ndarray  # (num_layers, num_heads) per-trial score S = L+ - L-
    L_plus: np.ndarray  # (num_layers, num_heads) needle attention sum
    L_minus: np.ndarray  # (num_layers, num_heads) off-needle attention sum (rescaled)
    generated_text: str
    num_answer_steps: int
    num_total_steps: int


def detect_single_trial_attention_spatial(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    needle_start: int,
    needle_end: int,
    num_layers: int,
    num_heads: int,
    answer_text: str,
    prefill_attn_impl: str,
    max_decode_steps: int = 50,
    newline_token_id: int | None = None,
    stop_token_ids: set[int] | None = None,
    think_start_id: int | None = None,
    think_end_id: int | None = None,
    max_thinking_tokens: int = 0,
) -> AttentionSpatialTrialResult:
    """Run one attention-only spatial-contrast retrieval head detection trial.

    Mirrors :func:`detect_single_trial_logit_contrib` but drops the OV
    projection: the per-position score is just the attention weight, with
    needle-vs-off-needle contrast and span-length rescaling applied via
    :func:`compute_attention_spatial_per_step`.

    Args:
        model: HuggingFace model (with output_attentions support).
        tokenizer: Tokenizer instance.
        input_ids: Full prompt token IDs, shape (1, seq_len).
        needle_start, needle_end: Token positions of the needle in input_ids.
        num_layers, num_heads: Model dims.
        answer_text: Gold answer string for answer step identification.
        prefill_attn_impl: Attention backend for prefill.
        max_decode_steps: Maximum decode steps before stopping.
        newline_token_id: Token ID for newline (stop condition).
        think_start_id, think_end_id: Token IDs for ``<think>``/``</think>``
            markers (None to disable thinking handling).
        max_thinking_tokens: Maximum tokens allowed inside thinking blocks.

    Returns:
        AttentionSpatialTrialResult with S_tau, L_plus, L_minus, and diagnostics.
        *generated_text* in the result excludes thinking content.
    """
    device = get_input_device(model)
    input_ids = input_ids.to(device)
    # --- Prefill ---
    switched = False
    if prefill_attn_impl != "eager":
        switched = set_model_attn_impl(model, prefill_attn_impl)

    with torch.inference_mode():
        try:
            prefill_out = model(
                input_ids=input_ids[:, :-1],
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
        except Exception as err:
            if switched:
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

    if prefill_attn_impl != "eager":
        set_model_attn_impl(model, "eager")

    # --- Autoregressive decode ---
    current_token = input_ids[:, -1:].clone()
    generated_ids: list[int] = []  # content tokens only (excludes thinking)

    # Store per-step attention weights for all layers (content steps only)
    # Each entry: list of (num_heads, key_len) tensors, one per layer
    step_attentions: list[list[torch.Tensor]] = []

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

            next_token_id = outputs.logits[0, -1].argmax().item()

            # --- Thinking block handling ---
            if has_thinking:
                if next_token_id == think_start_id:
                    in_thinking = True
                    thinking_token_count += 1
                    current_token[0, 0] = next_token_id
                    continue
                if next_token_id == think_end_id:
                    in_thinking = False
                    thinking_token_count += 1
                    current_token[0, 0] = next_token_id
                    continue
                if in_thinking:
                    thinking_token_count += 1
                    if thinking_token_count > max_thinking_tokens:
                        break
                    current_token[0, 0] = next_token_id
                    continue

            # --- Stop checks (before appending, so stop tokens stay out of generated_ids) ---
            step_token = tokenizer.convert_ids_to_tokens(next_token_id)
            if step_token == "<0x0A>" or next_token_id == newline_token_id:
                break
            if stop_token_ids and next_token_id in stop_token_ids:
                break

            # --- Content token ---
            generated_ids.append(next_token_id)
            content_steps += 1

            # Verify attention head count matches what main() reported. logit_contrib
            # derives num_heads from o_proj weight shape (which can disagree with
            # config.num_attention_heads on some VLMs, e.g. Gemma 4); we don't load
            # o_proj here, so we assert against the actual decode-time attention
            # tensor to catch shape mismatches before they corrupt the score matrix.
            actual_num_heads = outputs.attentions[0].shape[1]
            assert actual_num_heads == num_heads, (
                f"num_heads mismatch: caller passed {num_heads}, but model attention "
                f"output reports {actual_num_heads} heads. This will produce a different "
                f"score matrix shape than logit_contrib (which derives heads from o_proj "
                f"weights) and break per-head comparison. Override num_heads in main()."
            )

            # No `effective_len` truncation here (cf. logit_contrib): we don't index
            # into a V-cache, so we use whatever key_len the attention tensor reports.
            # For sliding-window/hybrid caches, HF returns attention with the actual
            # attended length already.
            step_attn = [a[0, :, -1, :].detach().cpu() for a in outputs.attentions]  # (num_heads, key_len)
            step_attentions.append(step_attn)

            if content_steps >= max_decode_steps:
                break

            current_token[0, 0] = next_token_id

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    num_total_steps = len(generated_ids)

    # --- Identify answer steps ---
    t_ans, _ = identify_answer_steps(generated_ids, answer_text, tokenizer)

    if len(t_ans) == 0:
        zeros = np.zeros((num_layers, num_heads), dtype=np.float32)
        return AttentionSpatialTrialResult(
            S_tau=zeros,
            L_plus=zeros,
            L_minus=zeros,
            generated_text=generated_text,
            num_answer_steps=0,
            num_total_steps=num_total_steps,
        )

    # --- Compute attention-spatial contributions for answer steps ---
    # For each answer step, we only need the attention weights at that step
    # (no V cache, no o_proj, no unembedding — that's the whole point of this
    # ablation relative to logit_contrib).

    # Accumulate L+ and L- across answer steps
    L_plus_accum = np.zeros((num_layers, num_heads), dtype=np.float64)
    L_minus_accum = np.zeros((num_layers, num_heads), dtype=np.float64)

    for t in t_ans:
        attn_at_t = step_attentions[t]  # list of (num_heads, key_len) per layer
        for layer_idx in range(num_layers):
            attn_weights = attn_at_t[layer_idx]  # already on CPU
            phi_needle, phi_off = compute_attention_spatial_per_step(
                attn_weights=attn_weights,
                needle_start=needle_start,
                needle_end=needle_end,
            )
            L_plus_accum[layer_idx] += phi_needle
            L_minus_accum[layer_idx] += phi_off

    # Average over answer steps
    n_ans = len(t_ans)
    L_plus = (L_plus_accum / n_ans).astype(np.float32)
    L_minus = (L_minus_accum / n_ans).astype(np.float32)
    S_tau = (L_plus - L_minus).astype(np.float32)

    return AttentionSpatialTrialResult(
        S_tau=S_tau,
        L_plus=L_plus,
        L_minus=L_minus,
        generated_text=generated_text,
        num_answer_steps=n_ans,
        num_total_steps=num_total_steps,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Attention-only spatial-contrast retrieval head detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: retrieval_heads/<model>_attention_spatial[_<dataset>].json)",
    )
    parser.add_argument("--dataset", type=str, default="nolima", choices=["niah", "nolima"], help="Probing dataset")
    parser.add_argument(
        "--haystack-dir", type=Path, default=None, help="Directory with NIAH data (default: data/haystack_for_detect)"
    )
    parser.add_argument("--nolima-dir", type=Path, default=Path("data/nolima"), help="Directory with NoLiMa data")
    parser.add_argument("--question-type", type=str, default="onehop", choices=["onehop", "twohop", "twohop2"])
    parser.add_argument("--nolima-variant", type=str, default="needle_set", choices=["needle_set", "needle_set_hard"])
    parser.add_argument("--max-characters-per-entry", type=int, default=1)
    parser.add_argument("--min-length", type=int, default=1000, help="Minimum context length in tokens")
    parser.add_argument("--max-length", type=int, default=50000, help="Maximum context length in tokens")
    parser.add_argument("--num-lengths", type=int, default=20)
    parser.add_argument("--num-depths", type=int, default=10)
    parser.add_argument(
        "--num-examples", type=int, default=None, help="If set, use stratified sampling to limit trial count"
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
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--use-hardcoded-periods", action="store_true", default=False)
    parser.add_argument("--force-llama2-periods", action="store_true")
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
        output_path = Path("retrieval_heads") / f"{model_short_name}_attention_spatial{dataset_suffix}.json"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    if args.haystack_dir is None:
        args.haystack_dir = Path("data/haystack_for_detect")

    console.rule("[bold]Attention-Only Spatial Retrieval Head Detection[/bold]")

    # Build test grid
    context_lengths, depth_percents = build_context_depth_ranges(
        args.min_length,
        args.max_length,
        args.num_lengths,
        args.num_depths,
    )

    # Build dataset trials
    if args.dataset == "niah":
        needles = load_niah_needles(args.haystack_dir)
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

    if args.num_examples is not None:
        dataset_trials = stratified_sample(
            dataset_trials,
            args.num_examples,
            seed=args.seed,
        )
        dataset_info += f" → sampled {len(dataset_trials)} trials"

    config_table = Table(title="Configuration", show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="bold")
    config_table.add_column("Value")
    config_table.add_row("Model", args.model)
    config_table.add_row("Method", "attention-only spatial contrast (no OV)")
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

    # Token caches
    haystack_token_cache: dict[str, list[int]] = {}
    haystack_period_cache: dict[str, list[int]] = {}
    needle_token_cache: dict[str, list[int]] = {}
    question_token_cache: dict[str, list[int]] = {}

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        attn_implementation="eager",
        trust_remote_code=True,
    ).eval()

    num_layers, num_heads = extract_model_config(model)

    model_table = Table(title="Model", show_header=False, box=None, padding=(0, 2))
    model_table.add_column("Key", style="bold")
    model_table.add_column("Value")
    model_table.add_row("Architecture", f"{num_layers}L × {num_heads}H")
    model_table.add_row("Dtype", args.dtype)
    console.print(model_table)
    stop_token_ids = get_stop_token_ids(tokenizer, model)
    console.print(f"[dim]Stop token IDs ({len(stop_token_ids)}): {stop_token_ids}[/dim]")
    console.print("[green]Model loaded[/green]\n")

    # Resume or fresh start
    if args.resume:
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt is not None:
            head_counter = defaultdict(list)
            for k, v in ckpt.get("head_counter", {}).items():
                head_counter[k] = v
            completed_trials = ckpt.get("completed_trials", [])
            passed_trial_ids = ckpt.get("passed_trial_ids", [])
            console.print(f"Resumed: {len(completed_trials)} trials already completed")
        else:
            head_counter = defaultdict(list)
            completed_trials = []
            passed_trial_ids = []
    else:
        head_counter = defaultdict(list)
        completed_trials = []
        passed_trial_ids = []

    completed_set = set(t if isinstance(t, str) else tuple(t) for t in completed_trials)
    trials = [t for t in dataset_trials if t.trial_id not in completed_set]
    total_trials = len(dataset_trials)
    console.print(f"Trials: {total_trials} total, {len(trials)} remaining\n")

    num_passed = 0
    num_no_answer_steps = 0

    # Pooled accumulators for global aggregation
    pooled_L_plus = np.zeros((num_layers, num_heads), dtype=np.float64)
    pooled_L_minus = np.zeros((num_layers, num_heads), dtype=np.float64)
    pooled_ans_count = 0
    # Per-trial S values for bootstrap CI
    per_trial_S: list[np.ndarray] = []

    if not trials:
        console.print("[yellow]All trials already completed.[/yellow]")
    else:
        loop_start_time = time.time()
        processed = 0
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
                "Detecting retrieval heads (attention-spatial)",
                total=len(trials),
                status="",
            )

            for trial in trials:
                elapsed_s = time.time() - loop_start_time
                if processed > 0:
                    avg_s = elapsed_s / processed
                    eta_s = (len(trials) - processed) * avg_s
                    timing = f"elapsed={format_duration(elapsed_s)} eta={format_duration(eta_s)}"
                else:
                    timing = "elapsed=00:00 eta=estimating..."

                progress.update(
                    ptask,
                    status=(
                        f"ctx={trial.context_length} depth={trial.depth_percent}% "
                        f"id={trial.trial_id[:30]} | {timing}"
                    ),
                )

                # --- Build context with needle ---
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

                # Token-space composition with tracked needle position.
                question_tokens = None
                if not (trial.prompt_template is not None and "{haystack}" in trial.prompt_template):
                    question_tokens = question_token_cache.get(trial.question)
                    if question_tokens is None:
                        question_tokens = tokenizer.encode(trial.question, add_special_tokens=False)
                        question_token_cache[trial.question] = question_tokens

                input_ids, needle_start, needle_end, _, _ = build_tracked_prompt(
                    haystack_tokens=haystack_tokens,
                    needle_tokens=needle_tokens,
                    context_length=trial.context_length,
                    depth_percent=trial.depth_percent,
                    period_positions=haystack_period_cache[trial.haystack_text],
                    tokenizer=tokenizer,
                    prompt_template=trial.prompt_template if "{haystack}" in (trial.prompt_template or "") else None,
                    question_tokens=question_tokens,
                    use_chat_template=bool(args.chat_template),
                    prompt_suffix_ids=prompt_suffix_ids,
                    add_bos=tokenizer_has_bos,
                )

                if needle_start < 0 or needle_end <= needle_start:
                    console.print(f"[yellow]Warning: invalid needle span ({trial.trial_id}). Skipping.[/yellow]")
                    completed_trials.append(trial.trial_id)
                    processed += 1
                    progress.advance(ptask)
                    continue

                # --- Run attention-spatial detection ---
                try:
                    result = detect_single_trial_attention_spatial(
                        model,
                        tokenizer,
                        input_ids,
                        needle_start,
                        needle_end,
                        num_layers,
                        num_heads,
                        trial.answer_text,
                        args.prefill_attn_impl,
                        max_decode_steps=args.max_decode_steps,
                        newline_token_id=newline_token_id,
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
                    processed += 1
                    progress.advance(ptask)
                    continue

                # --- ROUGE gate (strip residual thinking as safety net) ---
                rouge_text = (
                    strip_thinking_content(result.generated_text)
                    if args.max_thinking_tokens > 0
                    else result.generated_text
                )
                rouge_result = scorer.score(trial.answer_text, rouge_text)["rouge1"].recall * 100

                if args.debug_trials > 0 and processed < args.debug_trials:
                    progress.console.print(
                        f"\n[bold]--- Debug trial {processed + 1}/{args.debug_trials} ---[/bold]\n"
                        f"  [dim]trial_id:[/dim] {trial.trial_id}\n"
                        f"  [dim]ctx_len:[/dim]  {trial.context_length}  "
                        f"[dim]depth:[/dim] {trial.depth_percent}%\n"
                        f"  [dim]answer:[/dim]   {trial.answer_text!r}\n"
                        f"  [dim]generated (raw, repr):[/dim]\n    {result.generated_text!r}\n"
                        f"  [dim]answer_steps:[/dim] {result.num_answer_steps}/{result.num_total_steps}\n"
                        f"  [dim]ROUGE-1 recall:[/dim] {rouge_result:.1f}  "
                        f"({'[green]PASS[/green]' if rouge_result > args.rouge_threshold else '[red]FAIL[/red]'})"
                    )

                if rouge_result > args.rouge_threshold and result.num_answer_steps > 0:
                    num_passed += 1
                    passed_trial_ids.append(trial.trial_id)

                    # Accumulate for global pooling
                    pooled_L_plus += result.L_plus * result.num_answer_steps
                    pooled_L_minus += result.L_minus * result.num_answer_steps
                    pooled_ans_count += result.num_answer_steps

                    # Store per-trial S for bootstrap
                    per_trial_S.append(result.S_tau)

                    # Also store per-trial scores in head_counter
                    for layer_idx in range(num_layers):
                        for head_idx in range(num_heads):
                            key = f"{layer_idx}-{head_idx}"
                            head_counter[key].append(float(result.S_tau[layer_idx, head_idx]))

                elif rouge_result > args.rouge_threshold and result.num_answer_steps == 0:
                    num_no_answer_steps += 1

                completed_trials.append(trial.trial_id)
                processed += 1
                progress.advance(ptask)

                # Checkpoint every 10 trials
                if len(completed_trials) % 10 == 0:
                    save_checkpoint(
                        {
                            "head_counter": dict(head_counter),
                            "completed_trials": completed_trials,
                            "passed_trial_ids": passed_trial_ids,
                        },
                        checkpoint_path,
                    )

        total_elapsed = time.time() - loop_start_time
        console.print(f"\n[green]{num_passed}[/green] / {len(trials)} trials passed ROUGE gate")
        if num_no_answer_steps > 0:
            console.print(f"[yellow]{num_no_answer_steps} trials passed ROUGE but had 0 answer steps[/yellow]")
        if oom_count > 0:
            console.print(f"[yellow]{oom_count} OOM skips[/yellow]")
        console.print(f"Total time: {format_duration(total_elapsed)}")

    # --- Compute global pooled scores + bootstrap CI ---
    pooled_ci_lower = None
    pooled_ci_upper = None

    if pooled_ans_count > 0:
        console.print(f"\n[bold]Pooled:[/bold] {pooled_ans_count} answer steps across {num_passed} trials")

        # Bootstrap CI over trials
        if len(per_trial_S) >= 10:
            n_bootstrap = 1000
            rng = np.random.RandomState(args.seed)
            S_arr = np.stack(per_trial_S)  # (num_trials, num_layers, num_heads)
            boot_means = np.empty((n_bootstrap, num_layers, num_heads), dtype=np.float32)
            for b in range(n_bootstrap):
                idx = rng.randint(0, len(S_arr), size=len(S_arr))
                boot_means[b] = S_arr[idx].mean(axis=0)
            pooled_ci_lower = np.percentile(boot_means, 2.5, axis=0).astype(np.float32)
            pooled_ci_upper = np.percentile(boot_means, 97.5, axis=0).astype(np.float32)

            significant = (pooled_ci_lower > 0).sum()
            significant_neg = (pooled_ci_upper < 0).sum()
            console.print(
                f"[bold]Bootstrap 95% CI:[/bold] "
                f"{significant}/{num_layers * num_heads} heads with CI > 0 (needle-dominant), "
                f"{significant_neg}/{num_layers * num_heads} heads with CI < 0 (off-needle-dominant)"
            )

    # --- Save results ---
    scores_dict = {}
    for layer in range(num_layers):
        for head in range(num_heads):
            key = f"{layer}-{head}"
            scores_dict[key] = head_counter.get(key, [])

    result_envelope = {
        "meta": {
            "method": "attention_spatial",
            "dataset": args.dataset,
            "num_trials_passed": num_passed,
            "num_trials_total": total_trials,
            "pooled_ans_steps": pooled_ans_count,
            "rouge_threshold": args.rouge_threshold,
            "model": args.model,
            "question_type": args.question_type if args.dataset == "nolima" else None,
            "context_range": [args.min_length, args.max_length],
            "num_lengths": args.num_lengths,
            "num_depths": args.num_depths,
            "max_decode_steps": args.max_decode_steps,
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "scores": scores_dict,
        "trial_ids": passed_trial_ids,
    }

    if pooled_ci_lower is not None:
        ci_dict = {}
        for layer in range(num_layers):
            for head in range(num_heads):
                key = f"{layer}-{head}"
                ci_dict[key] = [
                    float(pooled_ci_lower[layer, head]),
                    float(pooled_ci_upper[layer, head]),
                ]
        result_envelope["confidence_intervals"] = ci_dict

    output_path.write_text(json.dumps(result_envelope, indent=2))
    console.print(f"\n[bold green]Saved attention-spatial scores to {output_path}[/bold green]")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Print top heads
    scored = [(key, float(np.mean(scores)) if scores else 0.0) for key, scores in scores_dict.items()]
    scored.sort(key=lambda x: x[1], reverse=True)

    table = Table(title="Top 20 Retrieval Heads (attention-spatial)")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Layer-Head")
    table.add_column("Mean S", justify="right")
    table.add_column("Trials", justify="right")

    for i, (key, mean_score) in enumerate(scored[:20]):
        table.add_row(
            str(i + 1),
            key,
            f"{mean_score:.6f}",
            str(len(scores_dict[key])),
        )
    console.print(table)

    # Also print bottom 5 (most negative — heads writing answer info from non-needle)
    table_neg = Table(title="Bottom 5 Retrieval Heads (off-needle dominant)")
    table_neg.add_column("Rank", style="dim", width=4)
    table_neg.add_column("Layer-Head")
    table_neg.add_column("Mean S", justify="right")

    scored_asc = sorted(scored, key=lambda x: x[1])
    for i, (key, mean_score) in enumerate(scored_asc[:5]):
        table_neg.add_row(str(len(scored) - i), key, f"{mean_score:.6f}")
    console.print(table_neg)

    # Validate output
    from locos_eval.retrieval_heads import load_retrieval_heads

    heads = load_retrieval_heads(str(output_path), num_heads=10)
    console.print(f"\nValidation: load_retrieval_heads() returned {len(heads)} heads")
    console.print(f"Top 10: {heads}")


if __name__ == "__main__":
    main()
