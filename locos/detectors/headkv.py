#!/usr/bin/env python3
"""HeadKV/SnapKV-style retrieval head detector (strict variant).

Per trial τ and head (l, h):

    phi_t^{(l,h)} = sum_{j in needle} alpha_{t,j}^{(l,h)}     [per anchor step]
    S_tau^{(l,h)} = max_{t in anchor_window} phi_t^{(l,h)}

The anchor window is the last K prompt tokens (default K=8, matching
SnapKV's typical observation-window size). No generation, no ROUGE gate,
no spatial contrast — this is the cleanest "different family" baseline:
score heads by how strongly they attend to the needle from the prompt's
final positions, before any answer is generated.

References:
- Fu et al., "Not All Heads Matter: Head-Level KV Cache Compression with
  Importance Estimation" (HeadKV).
- Li et al., SnapKV; Cai et al., PyramidKV — all use a last-K-tokens
  observation window for attention-based importance scoring.

Distinct from `attention_spatial.py` (which uses decode-time answer
steps, mean aggregation, and spatial contrast).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import gc
import json
import time
from collections import defaultdict

import numpy as np
import torch
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

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
    get_input_device,
    set_model_attn_impl,
    tokenizer_adds_bos,
)
from locos.utils.needle_utils import (
    build_period_token_positions,
    build_tracked_prompt,
    get_period_tokens,
)
from locos.utils.needle_utils import load_needles as load_niah_needles

console = Console()


def compute_needle_attention_per_step(
    attn_weights: torch.Tensor,
    needle_start: int,
    needle_end: int,
) -> np.ndarray:
    """Sum attention weights over needle key positions, per head.

    Args:
        attn_weights: (num_heads, key_len) attention weights for one layer
            at one anchor-window step. Rows are expected to sum to ~1
            (softmax output) but this is not enforced.
        needle_start, needle_end: Needle span [start, end) in key positions.

    Returns:
        phi of shape (num_heads,) as np.float32 — per-head needle attention
        mass at this anchor step. The trial-level aggregation
        (max over anchor window) lives in the trial loop, not here.
    """
    assert attn_weights.ndim == 2, f"Expected (num_heads, key_len), got {attn_weights.shape}"
    needle_len = needle_end - needle_start
    assert needle_len > 0, f"needle_len must be > 0 (got start={needle_start}, end={needle_end})"
    assert needle_end <= attn_weights.shape[1], f"needle_end ({needle_end}) exceeds key_len ({attn_weights.shape[1]})"

    phi = attn_weights[:, needle_start:needle_end].sum(dim=-1)
    return phi.float().cpu().numpy()


@dataclass
class HeadKVTrialResult:
    """Result of a single HeadKV-style detection trial."""

    S_tau: np.ndarray  # (num_layers, num_heads) per-trial score: max over anchor window
    per_step_phi: np.ndarray  # (num_layers, num_heads, anchor_window) raw per-step needle attention
    anchor_window: int


def detect_single_trial_headkv(
    model,
    input_ids: torch.Tensor,
    needle_start: int,
    needle_end: int,
    num_layers: int,
    num_heads: int,
    prefill_attn_impl: str,
    anchor_window: int = 8,
) -> HeadKVTrialResult:
    """Run one HeadKV-style detection trial.

    Splits the prompt into a prefill segment input_ids[:, :-anchor_window]
    and an anchor segment input_ids[:, -anchor_window:]. Runs the prefill
    with output_attentions=False (cheap), then runs anchor_window incremental
    forward passes (one prompt token at a time) with output_attentions=True,
    capturing the (num_heads, key_len) attention vector per layer per step.

    For each layer, head, anchor step t:
        phi_t = sum_{j in needle} alpha_{t,j}
    Per trial:
        S_tau = max_t phi_t

    Args:
        model: HuggingFace model (eager attention for output_attentions support).
        input_ids: Full prompt token IDs, shape (1, seq_len). Must have
            seq_len >= anchor_window + 1.
        needle_start, needle_end: Token positions of the needle in input_ids
            (in key-space, which equals token-space here since we're not
            generating). Must satisfy 0 <= needle_start < needle_end <= seq_len.
            The needle MUST lie entirely in the prefill segment (i.e.
            needle_end <= seq_len - anchor_window) — otherwise the anchor
            window overlaps the needle and the score is degenerate.
        num_layers, num_heads: Model architecture dimensions.
        prefill_attn_impl: Attention backend for prefill ("eager" / "sdpa" / "flash_attention_2").
        anchor_window: Number of prompt-suffix tokens to score over (HeadKV's K).

    Returns:
        HeadKVTrialResult with S_tau, per_step_phi, anchor_window.
    """
    device = get_input_device(model)
    input_ids = input_ids.to(device)

    seq_len = input_ids.shape[1]
    assert seq_len > anchor_window, (
        f"seq_len ({seq_len}) must exceed anchor_window ({anchor_window}); " "the prompt is too short."
    )
    assert needle_end <= seq_len - anchor_window, (
        f"needle_end ({needle_end}) overlaps anchor window "
        f"[{seq_len - anchor_window}, {seq_len}). HeadKV requires the needle "
        f"to lie strictly in the prefill segment."
    )

    # --- Prefill all but the last anchor_window tokens ---
    switched = False
    if prefill_attn_impl != "eager":
        switched = set_model_attn_impl(model, prefill_attn_impl)

    with torch.inference_mode():
        try:
            prefill_out = model(
                input_ids=input_ids[:, : seq_len - anchor_window],
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
        except Exception as err:
            if switched:
                set_model_attn_impl(model, "eager")
                prefill_out = model(
                    input_ids=input_ids[:, : seq_len - anchor_window],
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

    # --- Anchor-window forward passes (output_attentions=True) ---
    per_step_phi = np.zeros((num_layers, num_heads, anchor_window), dtype=np.float32)

    with torch.inference_mode():
        for k_idx in range(anchor_window):
            anchor_token_pos = seq_len - anchor_window + k_idx
            current_token = input_ids[:, anchor_token_pos : anchor_token_pos + 1]
            outputs = model(
                input_ids=current_token,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            past_kv = outputs.past_key_values

            # Catch VLM head-count drift (mirrors attention_spatial's runtime guard)
            actual_num_heads = outputs.attentions[0].shape[1]
            assert actual_num_heads == num_heads, (
                f"num_heads mismatch: caller passed {num_heads}, but model "
                f"attention output reports {actual_num_heads} heads. Override "
                f"num_heads in main()."
            )

            for layer_idx in range(num_layers):
                # outputs.attentions[layer_idx] shape: (1, num_heads, 1, key_len)
                attn = outputs.attentions[layer_idx][0, :, -1, :]  # (num_heads, key_len)
                phi = compute_needle_attention_per_step(
                    attn_weights=attn,
                    needle_start=needle_start,
                    needle_end=needle_end,
                )
                per_step_phi[layer_idx, :, k_idx] = phi

    # --- HeadKV aggregation: max over anchor window ---
    S_tau = per_step_phi.max(axis=-1).astype(np.float32)  # (num_layers, num_heads)

    return HeadKVTrialResult(
        S_tau=S_tau,
        per_step_phi=per_step_phi,
        anchor_window=anchor_window,
    )


def main():
    parser = argparse.ArgumentParser(
        description="HeadKV-style retrieval head detection (anchor-window attention).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: retrieval_heads/<model>_headkv[_<dataset>].json)",
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
    parser.add_argument(
        "--anchor-window",
        type=int,
        default=8,
        help="Number of last prompt tokens used as the HeadKV anchor window.",
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
            "Tokenized and appended to the token sequence before detection. "
            "E.g. '<|channel|>final<|message|>' for GPT-oss to skip reasoning."
        ),
    )
    args = parser.parse_args()

    # Resolve output path
    model_short_name = args.model.split("/")[-1]
    dataset_suffix = f"_{args.dataset}" if args.dataset != "niah" else ""
    if args.output is None:
        output_path = Path("retrieval_heads") / f"{model_short_name}_headkv{dataset_suffix}.json"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    if args.haystack_dir is None:
        args.haystack_dir = Path("data/haystack_for_detect")

    console.rule("[bold]HeadKV-Style Retrieval Head Detection[/bold]")

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
    config_table.add_row(
        "Method",
        f"HeadKV (anchor window K={args.anchor_window}, max-aggregation, raw needle mass)",
    )
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
    tokenizer_has_bos = tokenizer_adds_bos(tokenizer)
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

    num_processed = 0
    num_skipped = 0

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
                "Detecting retrieval heads (headkv)",
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

                if needle_end > input_ids.shape[1] - args.anchor_window:
                    console.print(
                        f"[yellow]Skipping {trial.trial_id}: needle overlaps anchor window "
                        f"(needle_end={needle_end}, anchor_start={input_ids.shape[1] - args.anchor_window}).[/yellow]"
                    )
                    num_skipped += 1
                    completed_trials.append(trial.trial_id)
                    processed += 1
                    progress.advance(ptask)
                    continue

                # --- Run HeadKV detection ---
                try:
                    result = detect_single_trial_headkv(
                        model,
                        input_ids,
                        needle_start,
                        needle_end,
                        num_layers,
                        num_heads,
                        args.prefill_attn_impl,
                        anchor_window=args.anchor_window,
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

                num_processed += 1
                passed_trial_ids.append(trial.trial_id)
                per_trial_S.append(result.S_tau)
                for layer_idx in range(num_layers):
                    for head_idx in range(num_heads):
                        key = f"{layer_idx}-{head_idx}"
                        head_counter[key].append(float(result.S_tau[layer_idx, head_idx]))

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
        console.print(f"\n[green]{num_processed}[/green] / {len(trials)} trials processed")
        if num_skipped > 0:
            console.print(f"[yellow]{num_skipped} trials skipped (anchor window overlap)[/yellow]")
        if oom_count > 0:
            console.print(f"[yellow]{oom_count} OOM skips[/yellow]")
        console.print(f"Total time: {format_duration(total_elapsed)}")

    # --- Compute bootstrap CI ---
    pooled_ci_lower = None
    pooled_ci_upper = None

    if num_processed > 0:
        console.print(f"\n[bold]Trials processed:[/bold] {num_processed} (skipped {num_skipped} for anchor overlap)")

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
            "method": "headkv",
            "dataset": args.dataset,
            # In HeadKV, "passed" = "did not skip due to anchor overlap" (no quality gate).
            # Field name kept for downstream consumer compatibility (load_retrieval_heads, compare scripts).
            "num_trials_passed": num_processed,
            "num_trials_total": total_trials,
            "anchor_window": args.anchor_window,
            "model": args.model,
            "question_type": args.question_type if args.dataset == "nolima" else None,
            "context_range": [args.min_length, args.max_length],
            "num_lengths": args.num_lengths,
            "num_depths": args.num_depths,
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
    console.print(f"\n[bold green]Saved HeadKV scores to {output_path}[/bold green]")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Print top heads
    scored = [(key, float(np.mean(scores)) if scores else 0.0) for key, scores in scores_dict.items()]
    scored.sort(key=lambda x: x[1], reverse=True)

    table = Table(title="Top 20 Retrieval Heads (HeadKV)")
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

    # Also print bottom 5 (heads with lowest max-anchor needle attention —
    # never look at the needle from any anchor position)
    table_neg = Table(title="Bottom 5 Retrieval Heads (low max anchor needle attention)")
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
