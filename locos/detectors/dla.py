#!/usr/bin/env python3
"""Head-level direct logit attribution (DLA) — LOCOS without spatial contrast.

Reviewer-requested controlled ablation for the spatial-contrast component of
LOCOS. Per answer decode step t and head (l, h):

    phi_{t,j}^{(l,h)} = u_{y_t}^T * W_O^{(l,h)} * v_j^{(l,h)} * alpha_{t,j}^{(l,h)}
    S^DLA(l, h)       = (1/|T_ans|) * sum_t sum_j phi_{t,j}^{(l,h)}

Same per-position OV projection as :mod:`logit_contrib`, but aggregates over
*all* key positions instead of decomposing into needle vs off-needle. Isolates
the contribution of LOCOS's spatial contrast: if DLA tracks LOCOS, the spatial
decomposition is decoration; if LOCOS dominates DLA, spatial contrast is
load-bearing.
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
from locos.detectors.logit_contrib import (
    extract_head_config,
    get_o_proj_weights,
    get_unembedding_matrix,
)
from locos.utils.common import (
    build_context_depth_ranges,
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


def compute_dla_per_step(
    attn_weights: torch.Tensor,
    value_cache: torch.Tensor,
    o_proj_weight: torch.Tensor,
    u_y: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
) -> np.ndarray:
    """Compute per-head DLA at one decode step, summed over all key positions.

    Mirrors :func:`compute_logit_contribution_per_step` but skips the
    needle/off-needle decomposition — returns the total contribution from
    every key position, which is the "no spatial contrast" aggregation.

    Args:
        attn_weights: (num_heads, key_len) attention weights for this step.
        value_cache: (num_kv_heads, key_len, head_dim) V cache up to this step.
        o_proj_weight: (num_heads, head_dim, hidden_dim) per-head output projection.
        u_y: (hidden_dim,) unembedding vector for the correct answer token.
        num_heads: Number of Q-heads.
        num_kv_heads: Number of KV-heads (for GQA expansion).

    Returns:
        Per-head total logit contribution, shape ``(num_heads,)`` as np.float32.
    """
    if num_kv_heads != num_heads:
        gqa_ratio = num_heads // num_kv_heads
        V = value_cache.repeat_interleave(gqa_ratio, dim=0)
    else:
        V = value_cache

    assert V.shape[0] == num_heads

    compute_device = V.device
    o_proj_weight = o_proj_weight.to(compute_device)
    u_y = u_y.to(compute_device)
    attn_weights = attn_weights.to(compute_device)

    u_projected = torch.einsum("hde,e->hd", o_proj_weight, u_y)  # (num_heads, head_dim)
    logit_contrib = torch.einsum("hkd,hd->hk", V, u_projected)  # (num_heads, key_len)
    phi = attn_weights * logit_contrib  # (num_heads, key_len)

    return phi.sum(dim=-1).float().cpu().numpy()  # (num_heads,)


@dataclass
class DLATrialResult:
    """Result of a single DLA detection trial."""

    S: np.ndarray  # (num_layers, num_heads) per-trial DLA score (no needle/off split)
    generated_text: str
    num_answer_steps: int
    num_total_steps: int


def detect_single_trial_dla(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    answer_text: str,
    prefill_attn_impl: str,
    o_proj_weights: list[torch.Tensor],
    W_U: torch.Tensor,
    max_decode_steps: int = 50,
    newline_token_id: int | None = None,
    stop_token_ids: set[int] | None = None,
    think_start_id: int | None = None,
    think_end_id: int | None = None,
    max_thinking_tokens: int = 0,
) -> DLATrialResult:
    """Run one DLA retrieval head detection trial.

    Same prefill + decode pipeline as :func:`detect_single_trial_logit_contrib`,
    but the per-step aggregation skips needle/off-needle decomposition.
    """
    device = get_input_device(model)
    input_ids = input_ids.to(device)

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

    current_token = input_ids[:, -1:].clone()
    generated_ids: list[int] = []
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

            step_token = tokenizer.convert_ids_to_tokens(next_token_id)
            if step_token == "<0x0A>" or next_token_id == newline_token_id:
                break
            if stop_token_ids and next_token_id in stop_token_ids:
                break

            generated_ids.append(next_token_id)
            content_steps += 1

            actual_num_heads = outputs.attentions[0].shape[1]
            assert actual_num_heads == num_heads, (
                f"num_heads mismatch: caller passed {num_heads}, but model attention "
                f"output reports {actual_num_heads} heads."
            )

            step_attn = [a[0, :, -1, :].detach().cpu() for a in outputs.attentions]
            step_attentions.append(step_attn)

            if content_steps >= max_decode_steps:
                break

            current_token[0, 0] = next_token_id

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    num_total_steps = len(generated_ids)

    t_ans, _ = identify_answer_steps(generated_ids, answer_text, tokenizer)

    if len(t_ans) == 0:
        zeros = np.zeros((num_layers, num_heads), dtype=np.float32)
        return DLATrialResult(
            S=zeros,
            generated_text=generated_text,
            num_answer_steps=0,
            num_total_steps=num_total_steps,
        )

    answer_token_ids = [generated_ids[t] for t in t_ans]
    u_vectors = W_U[answer_token_ids].to(device)  # (num_answer_steps, hidden_dim)

    S_accum = np.zeros((num_layers, num_heads), dtype=np.float64)

    for ans_idx, t in enumerate(t_ans):
        u_y = u_vectors[ans_idx]
        attn_at_t = step_attentions[t]

        for layer_idx in range(num_layers):
            if hasattr(past_kv, "layers"):
                v_cache = past_kv.layers[layer_idx].values[0]
            elif hasattr(past_kv, "value_cache"):
                v_cache = past_kv.value_cache[layer_idx][0]
            else:
                v_cache = past_kv[layer_idx][1][0]

            key_len_at_t = attn_at_t[layer_idx].shape[1]
            cache_len = v_cache.shape[1]
            effective_len = min(key_len_at_t, cache_len)
            v_cache_at_t = v_cache[:, :effective_len, :]

            attn_weights = attn_at_t[layer_idx][:, :effective_len].to(device)
            o_proj_w = o_proj_weights[layer_idx].to(device)

            phi_total = compute_dla_per_step(
                attn_weights=attn_weights,
                value_cache=v_cache_at_t,
                o_proj_weight=o_proj_w,
                u_y=u_y,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )

            S_accum[layer_idx] += phi_total

    n_ans = len(t_ans)
    S = (S_accum / n_ans).astype(np.float32)

    return DLATrialResult(
        S=S,
        generated_text=generated_text,
        num_answer_steps=n_ans,
        num_total_steps=num_total_steps,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Head-level direct logit attribution (DLA) — LOCOS without spatial contrast.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: retrieval_heads/<model>_dla[_<dataset>].json)",
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
            "Required for models that need structured prompts (e.g. GPT-oss). "
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

    model_short_name = args.model.split("/")[-1]
    dataset_suffix = f"_{args.dataset}" if args.dataset != "niah" else ""
    if args.output is None:
        output_path = Path("retrieval_heads") / f"{model_short_name}_dla{dataset_suffix}.json"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    if args.haystack_dir is None:
        args.haystack_dir = Path("data/haystack_for_detect")

    console.rule("[bold]DLA Retrieval Head Detection (no spatial contrast)[/bold]")

    context_lengths, depth_percents = build_context_depth_ranges(
        args.min_length,
        args.max_length,
        args.num_lengths,
        args.num_depths,
    )

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
    config_table.add_row("Method", "head-level DLA (no spatial contrast)")
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        attn_implementation="eager",
        trust_remote_code=True,
    ).eval()

    num_layers, num_heads, num_kv_heads, head_dim = extract_head_config(model)

    console.print("Extracting o_proj weights and unembedding matrix...")
    o_proj_weights, actual_num_heads = get_o_proj_weights(model, num_layers, num_heads, head_dim)
    if actual_num_heads != num_heads:
        console.print(f"[cyan]Overriding num_heads: {num_heads} → {actual_num_heads} (from o_proj weight shape)[/cyan]")
        num_heads = actual_num_heads
    W_U = get_unembedding_matrix(model)

    model_table = Table(title="Model", show_header=False, box=None, padding=(0, 2))
    model_table.add_column("Key", style="bold")
    model_table.add_column("Value")
    model_table.add_row("Architecture", f"{num_layers}L × {num_heads}H (Q) × {num_kv_heads}H (KV) × {head_dim}d")
    model_table.add_row("GQA ratio", str(num_heads // num_kv_heads))
    model_table.add_row("Vocab size", str(W_U.shape[0]))
    model_table.add_row("Dtype", args.dtype)
    console.print(model_table)
    stop_token_ids = get_stop_token_ids(tokenizer, model)
    console.print(f"[dim]Stop token IDs ({len(stop_token_ids)}): {stop_token_ids}[/dim]")
    console.print("[green]Model loaded[/green]\n")

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

    pooled_S = np.zeros((num_layers, num_heads), dtype=np.float64)
    pooled_ans_count = 0
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
                "Detecting retrieval heads (DLA)",
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

                try:
                    result = detect_single_trial_dla(
                        model,
                        tokenizer,
                        input_ids,
                        num_layers,
                        num_heads,
                        num_kv_heads,
                        trial.answer_text,
                        args.prefill_attn_impl,
                        o_proj_weights=o_proj_weights,
                        W_U=W_U,
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

                    pooled_S += result.S * result.num_answer_steps
                    pooled_ans_count += result.num_answer_steps

                    per_trial_S.append(result.S)

                    for layer_idx in range(num_layers):
                        for head_idx in range(num_heads):
                            key = f"{layer_idx}-{head_idx}"
                            head_counter[key].append(float(result.S[layer_idx, head_idx]))

                elif rouge_result > args.rouge_threshold and result.num_answer_steps == 0:
                    num_no_answer_steps += 1

                completed_trials.append(trial.trial_id)
                processed += 1
                progress.advance(ptask)

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

    pooled_ci_lower = None
    pooled_ci_upper = None

    if pooled_ans_count > 0:
        console.print(f"\n[bold]Pooled:[/bold] {pooled_ans_count} answer steps across {num_passed} trials")

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
                f"{significant}/{num_layers * num_heads} heads with CI > 0, "
                f"{significant_neg}/{num_layers * num_heads} heads with CI < 0"
            )

    scores_dict = {}
    for layer in range(num_layers):
        for head in range(num_heads):
            key = f"{layer}-{head}"
            scores_dict[key] = head_counter.get(key, [])

    result_envelope = {
        "meta": {
            "method": "dla",
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
    console.print(f"\n[bold green]Saved DLA scores to {output_path}[/bold green]")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    scored = [(key, float(np.mean(scores)) if scores else 0.0) for key, scores in scores_dict.items()]
    scored.sort(key=lambda x: x[1], reverse=True)

    table = Table(title="Top 20 Retrieval Heads (DLA)")
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

    from locos_eval.retrieval_heads import load_retrieval_heads

    heads = load_retrieval_heads(str(output_path), num_heads=10)
    console.print(f"\nValidation: load_retrieval_heads() returned {len(heads)} heads")
    console.print(f"Top 10: {heads}")


if __name__ == "__main__":
    main()
