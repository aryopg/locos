#!/usr/bin/env python3
"""Contrastive attention-based retrieval head detection.

Implements the scoring method from non_literal_locos.tex:
instead of requiring token-identity between attended and generated positions
(Wu et al. copy-head criterion), this method measures whether a head's
attention to the needle span is *answer-contingent* — i.e., higher during
answer-generating decode steps than during other steps.

Supports two scoring variants:
  - Top-k (default): fraction of top-k attended positions falling in the
    needle span, contrasted between answer and non-answer steps.
  - Mass: total attention probability mass on the needle span, contrasted
    between answer and non-answer steps.

Produces a JSON file (envelope format) compatible with
locos_eval's load_retrieval_heads().

Usage:
    # Quick test with NoLiMa
    python locos/detect_contrastive.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --dataset nolima --max-length 4000 --num-lengths 3

    # Full detection with NIAH
    python locos/detect_contrastive.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --min-length 1000 --max-length 50000

    # Use mass variant instead of top-k
    python locos/detect_contrastive.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --dataset nolima --top-k 0

    # Resume from checkpoint
    python locos/detect_contrastive.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct --resume

Requires: GPU, transformers, rouge-score (pip install -e ".[eval]")
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
from dataclasses import dataclass

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
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


# ---------------------------------------------------------------------------
# Answer step identification
# ---------------------------------------------------------------------------


def identify_answer_steps(
    generated_ids: list[int],
    answer_text: str,
    tokenizer,
) -> tuple[list[int], list[int]]:
    """Classify each decode step as answer-generating or not.

    Identifies which positions in ``generated_ids`` correspond to the
    gold ``answer_text``. Uses exact subsequence matching first, then
    falls back to decoded string matching for tokenization mismatches.

    Args:
        generated_ids: Token IDs produced during autoregressive decode.
        answer_text: Gold answer string (e.g., a character name).
        tokenizer: HuggingFace tokenizer instance.

    Returns:
        (t_ans, t_not_ans): Lists of 0-based decode step indices.
        t_ans may be empty if no match is found.
    """
    if not generated_ids:
        return [], []

    n = len(generated_ids)

    # --- Strategy 1: exact contiguous subsequence match ---
    answer_token_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    if answer_token_ids:
        span_len = len(answer_token_ids)
        for i in range(n - span_len + 1):
            if generated_ids[i : i + span_len] == answer_token_ids:
                t_ans = list(range(i, i + span_len))
                t_not_ans = [j for j in range(n) if j not in set(t_ans)]
                return t_ans, t_not_ans

    # --- Strategy 2: decoded string matching ---
    # Decode each token individually and find where answer_text appears
    # in the concatenated decoded output. This handles tokenization
    # differences (e.g., leading space tokens, BPE splits).
    decoded_tokens = [tokenizer.decode([tid]) for tid in generated_ids]
    cumulative = ""
    # Build (start_char, end_char) per token
    token_char_ranges = []
    for dt in decoded_tokens:
        start = len(cumulative)
        cumulative += dt
        end = len(cumulative)
        token_char_ranges.append((start, end))

    # Find answer_text in the cumulative string (case-sensitive first)
    answer_lower = answer_text.lower()
    cumulative_lower = cumulative.lower()
    match_start = cumulative_lower.find(answer_lower)
    if match_start != -1:
        match_end = match_start + len(answer_lower)
        t_ans = []
        for idx, (cs, ce) in enumerate(token_char_ranges):
            # Token overlaps with the answer span
            if ce > match_start and cs < match_end:
                t_ans.append(idx)
        if t_ans:
            t_not_ans = [j for j in range(n) if j not in set(t_ans)]
            return t_ans, t_not_ans

    # --- No match found ---
    return [], list(range(n))


# ---------------------------------------------------------------------------
# Core detection: contrastive attention scoring
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Result of a single contrastive detection trial."""

    R_tau: np.ndarray  # (num_layers, num_heads) contrastive or A+ scores
    A_plus: np.ndarray  # (num_layers, num_heads) needle attention during answer steps
    A_minus: np.ndarray | None  # (num_layers, num_heads) needle attention during non-answer steps, or None
    generated_text: str
    num_answer_steps: int
    num_not_answer_steps: int
    num_total_steps: int
    used_baseline: bool  # True if contrastive (A+ - A-), False if A+ only

    # Per-step data for pooled mode (only populated when return_step_scores=True)
    step_scores: np.ndarray | None = None  # (num_steps, num_layers, num_heads)
    t_ans: list[int] | None = None  # answer step indices
    t_not_ans: list[int] | None = None  # non-answer step indices


def detect_single_trial_contrastive(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    needle_start: int,
    needle_end: int,
    num_layers: int,
    num_heads: int,
    answer_text: str,
    prefill_attn_impl: str,
    top_k: int = 10,
    use_mass: bool = False,
    min_baseline_steps: int = 0,
    num_permutations: int = 0,
    permutation_percentile: float = 95.0,
    return_step_scores: bool = False,
    max_decode_steps: int = 50,
    newline_token_id: int | None = None,
) -> TrialResult:
    """Run one contrastive retrieval head detection trial.

    Implements the scoring from non_literal_locos.tex:
    for each decode step, compute the needle attention score (top-k overlap
    or attention mass), then contrast answer-phase vs non-answer-phase
    scores.

    When ``min_baseline_steps > 0`` and the number of non-answer steps is
    below this threshold, the contrastive baseline is dropped and R_tau
    is set to A_plus directly (short-generation regime).

    Args:
        model: HuggingFace model (with output_attentions support).
        tokenizer: Tokenizer instance.
        input_ids: Full prompt token IDs, shape (1, seq_len).
        needle_start, needle_end: Token positions of the needle in input_ids.
        num_layers, num_heads: Model architecture dimensions.
        answer_text: Gold answer string for answer step identification.
        prefill_attn_impl: Attention backend for prefill ("eager", "sdpa", etc.).
        top_k: Number of top attention positions for the top-k variant.
            Ignored when use_mass=True.
        use_mass: If True, use attention mass variant instead of top-k.
        min_baseline_steps: Minimum non-answer steps required to use the
            contrastive baseline. If T_not_ans < this, use A+ directly.
            Set to 0 to always use the contrastive baseline.
        num_permutations: Number of label permutations for significance
            filtering. Heads whose contrastive score doesn't exceed the
            percentile of the permutation null are zeroed out. 0 = disabled.
        permutation_percentile: Percentile of null distribution to use
            as threshold (e.g., 95.0 for p < 0.05).
        max_decode_steps: Maximum decode steps before stopping.
        newline_token_id: Token ID for newline (stop condition).

    Returns:
        TrialResult with R_tau, A_plus, A_minus, and diagnostics.
    """
    device = get_input_device(model)
    input_ids = input_ids.to(device)

    # --- Prefill: all tokens except last ---
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

    # Ensure eager mode for decode (required for output_attentions=True)
    if prefill_attn_impl != "eager":
        set_model_attn_impl(model, "eager")

    # --- Autoregressive decode with attention extraction ---
    current_token = input_ids[:, -1:].clone()  # (1, 1)
    generated_ids: list[int] = []
    step_scores: list[np.ndarray] = []  # each (num_layers, num_heads)

    if newline_token_id is None:
        newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

    with torch.inference_mode():
        for _ in range(max_decode_steps):
            outputs = model(
                input_ids=current_token,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            past_kv = outputs.past_key_values

            # Greedy next token
            next_token_id_val = outputs.logits[0, -1].argmax().item()
            generated_ids.append(next_token_id_val)

            # --- Compute per-head needle attention score for this step ---
            # outputs.attentions: tuple of num_layers tensors
            # Each: (batch=1, num_heads, query_len=1, key_len)
            if use_mass:
                # Attention mass variant (Eq. 1 in tex):
                # A_t = sum_{j=s}^{e-1} alpha_{t,j} for each head
                scores_this_step = torch.stack(
                    [a[0, :, -1, needle_start:needle_end].sum(dim=-1).cpu() for a in outputs.attentions]
                ).numpy()  # (num_layers, num_heads)
            else:
                # Top-k variant (Eq. 5 in tex):
                # A_t = |TopK_t ∩ [s,e)| / k for each head
                scores_this_step = torch.stack(
                    [_topk_needle_overlap(a[0, :, -1, :], needle_start, needle_end, top_k) for a in outputs.attentions]
                ).numpy()  # (num_layers, num_heads)

            step_scores.append(scores_this_step)

            # Stop conditions (faithful to original)
            step_token = tokenizer.convert_ids_to_tokens(next_token_id_val)
            if step_token == "<0x0A>" or next_token_id_val == newline_token_id:
                break
            if next_token_id_val == tokenizer.eos_token_id:
                break

            current_token[0, 0] = next_token_id_val

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # --- Classify decode steps as answer vs non-answer ---
    t_ans, t_not_ans = identify_answer_steps(generated_ids, answer_text, tokenizer)
    num_total_steps = len(generated_ids)

    if len(t_ans) == 0:
        zeros = np.zeros((num_layers, num_heads), dtype=np.float32)
        return TrialResult(
            R_tau=zeros,
            A_plus=zeros,
            A_minus=None,
            generated_text=generated_text,
            num_answer_steps=0,
            num_not_answer_steps=len(t_not_ans),
            num_total_steps=num_total_steps,
            used_baseline=False,
            step_scores=np.stack(step_scores) if return_step_scores and step_scores else None,
            t_ans=[],
            t_not_ans=t_not_ans if return_step_scores else None,
        )

    # --- Compute A+ and A- (Eq. 1 in tex) ---
    all_scores = np.stack(step_scores)  # (num_steps, num_layers, num_heads)
    assert all_scores.shape == (
        len(step_scores),
        num_layers,
        num_heads,
    ), f"Expected ({len(step_scores)}, {num_layers}, {num_heads}), got {all_scores.shape}"

    # A^{τ,+}_{l,h}: mean needle attention during answer steps
    A_plus = all_scores[t_ans].mean(axis=0)  # (num_layers, num_heads)

    # A^{τ,-}_{l,h}: mean needle attention during non-answer steps
    if t_not_ans:
        A_minus = all_scores[t_not_ans].mean(axis=0)
    else:
        A_minus = None

    # --- Decide scoring mode: contrastive or A+ only ---
    use_baseline = A_minus is not None and len(t_not_ans) >= min_baseline_steps

    if use_baseline:
        # R^τ_{l,h} = max(A^{τ,+} - A^{τ,-}, 0)  [Eq. 2]
        R_tau = np.maximum(A_plus - A_minus, 0.0).astype(np.float32)
    else:
        # Short-generation regime: use A+ directly (no baseline subtraction)
        R_tau = np.maximum(A_plus, 0.0).astype(np.float32)

    # --- Permutation-based significance filter ---
    # Zero out heads whose contrastive score is indistinguishable from
    # chance under random label permutation.
    if num_permutations > 0 and num_total_steps > 1 and len(t_ans) > 0 and len(t_ans) < num_total_steps:
        null_threshold = permutation_null_threshold(
            all_scores,
            num_ans_steps=len(t_ans),
            num_permutations=num_permutations,
            percentile=permutation_percentile,
        )
        R_tau = np.where(R_tau > null_threshold, R_tau, 0.0).astype(np.float32)

    return TrialResult(
        R_tau=R_tau,
        A_plus=A_plus,
        A_minus=A_minus,
        generated_text=generated_text,
        num_answer_steps=len(t_ans),
        num_not_answer_steps=len(t_not_ans),
        step_scores=all_scores if return_step_scores else None,
        t_ans=t_ans if return_step_scores else None,
        t_not_ans=t_not_ans if return_step_scores else None,
        num_total_steps=num_total_steps,
        used_baseline=use_baseline,
    )


def _topk_needle_overlap(
    attn_weights: torch.Tensor,
    needle_start: int,
    needle_end: int,
    k: int,
) -> torch.Tensor:
    """Compute fraction of top-k attended positions in needle span.

    Args:
        attn_weights: Shape (num_heads, key_len) — attention weights for
            one layer at one decode step.
        needle_start, needle_end: Needle span [start, end) in key positions.
        k: Number of top positions to consider.

    Returns:
        Tensor of shape (num_heads,) with values in [0, 1].
    """
    assert attn_weights.ndim == 2, f"Expected (num_heads, key_len), got {attn_weights.shape}"
    # Clamp to key_len to handle short sequences, but divide by the
    # original k per Eq. 5 in the spec: |TopK ∩ [s,e)| / k.
    actual_k = min(k, attn_weights.shape[1])
    topk_indices = torch.topk(attn_weights, actual_k, dim=-1).indices  # (num_heads, actual_k)
    # Count how many top-k indices fall in [needle_start, needle_end)
    in_span = (topk_indices >= needle_start) & (topk_indices < needle_end)
    overlap_count = in_span.sum(dim=-1).float()  # (num_heads,)
    return (overlap_count / k).cpu()


# ---------------------------------------------------------------------------
# Permutation-based significance filter
# ---------------------------------------------------------------------------


def permutation_null_threshold(
    all_scores: np.ndarray,
    num_ans_steps: int,
    num_permutations: int = 100,
    percentile: float = 95.0,
    seed: int | None = None,
) -> np.ndarray:
    """Compute per-head significance thresholds via label permutation.

    For each permutation, randomly assigns ``num_ans_steps`` decode steps
    as "answer" and the rest as "non-answer", computes the contrastive
    difference (A+ - A-), and returns the per-head percentile of this
    null distribution.  Heads whose real contrastive score doesn't exceed
    their threshold are indistinguishable from chance.

    Args:
        all_scores: Per-step needle attention scores, shape
            (num_steps, num_layers, num_heads).
        num_ans_steps: Number of steps to assign as "answer" in each
            permutation (matches the real trial's answer step count).
        num_permutations: Number of random permutations to run.
        percentile: Percentile of the null distribution to use as
            threshold (e.g., 95.0 for p < 0.05).
        seed: Random seed for reproducibility.

    Returns:
        Threshold array of shape (num_layers, num_heads).
    """
    num_steps = all_scores.shape[0]
    assert 0 < num_ans_steps < num_steps, f"Need 0 < num_ans_steps ({num_ans_steps}) < num_steps ({num_steps})"

    rng = np.random.RandomState(seed)
    null_diffs = np.empty(
        (num_permutations, all_scores.shape[1], all_scores.shape[2]),
        dtype=np.float32,
    )

    indices = np.arange(num_steps)
    for i in range(num_permutations):
        rng.shuffle(indices)
        perm_ans = indices[:num_ans_steps]
        perm_not_ans = indices[num_ans_steps:]
        perm_a_plus = all_scores[perm_ans].mean(axis=0)
        perm_a_minus = all_scores[perm_not_ans].mean(axis=0)
        null_diffs[i] = perm_a_plus - perm_a_minus

    # Per-head percentile threshold
    return np.percentile(null_diffs, percentile, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Contrastive attention-based retrieval head detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: retrieval_heads/<model>_contrastive[_<dataset>].json)",
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
        help="NoLiMa question type",
    )
    parser.add_argument(
        "--nolima-variant",
        type=str,
        default="needle_set",
        choices=["needle_set", "needle_set_hard"],
        help="NoLiMa needle set variant",
    )
    parser.add_argument("--max-characters-per-entry", type=int, default=1, help="Max character names per NoLiMa entry")
    parser.add_argument("--min-length", type=int, default=1000, help="Minimum context length in tokens")
    parser.add_argument("--max-length", type=int, default=50000, help="Maximum context length in tokens")
    parser.add_argument("--num-lengths", type=int, default=20, help="Number of context length intervals")
    parser.add_argument("--num-depths", type=int, default=10, help="Number of document depth intervals")
    parser.add_argument(
        "--num-examples", type=int, default=None, help="If set, use stratified sampling to limit trial count"
    )
    parser.add_argument("--max-decode-steps", type=int, default=50)
    parser.add_argument(
        "--rouge-threshold", type=float, default=50.0, help="Minimum ROUGE-1 recall * 100 to accept a trial"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Top-k positions for needle overlap. Set to 0 for mass variant."
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-9, help="Small constant for trial-wise normalization denominator"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help=(
            "Skip trial-wise normalization — accumulate raw R_tau "
            "(or raw A+) values. Useful as a diagnostic to check "
            "whether the ranking is sharp before normalization."
        ),
    )
    parser.add_argument(
        "--pooled",
        action="store_true",
        help=(
            "Pool all answer/non-answer steps across trials before "
            "contrasting. Instead of per-trial A+ - A-, computes a "
            "single global contrast per head with all observations. "
            "Produces one score per head (not per-trial lists). "
            "Ignores --no-normalize, --num-permutations, --delta-multiplier, "
            "--min-baseline-steps (these are per-trial options)."
        ),
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=0,
        help=(
            "Number of label permutations for per-head significance "
            "filtering. For each trial, randomly permutes answer/non-answer "
            "labels to build a null distribution and zeros out heads whose "
            "contrastive score doesn't exceed the percentile threshold. "
            "Set to 0 to disable (default). Recommended: 100."
        ),
    )
    parser.add_argument(
        "--permutation-percentile",
        type=float,
        default=95.0,
        help="Percentile of permutation null to use as threshold (default: 95)",
    )
    parser.add_argument(
        "--min-baseline-steps",
        type=int,
        default=0,
        help=(
            "Minimum non-answer decode steps required to use the "
            "contrastive baseline (A+ - A-). When T_not_ans < this, "
            "R_tau = A+ directly (no baseline subtraction). "
            "Set to 0 to always use the contrastive baseline. "
            "Recommended: 5 for short-answer tasks like NoLiMa."
        ),
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--prefill-attn-impl",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend for prefill only (decode uses eager for output_attentions)",
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
        help="Use hardcoded period tokens for known models",
    )
    parser.add_argument(
        "--force-llama2-periods",
        action="store_true",
        help="[DEBUG] Force Llama-2 period tokens regardless of model",
    )
    parser.add_argument(
        "--delta-multiplier",
        type=float,
        default=2.0,
        help=(
            "Minimum contrastive signal gate, expressed as a multiple "
            "of the expected attention mass under uniform attention: "
            "delta = multiplier * (needle_len / context_len). Trials "
            "where max_{l,h} R^tau < delta are excluded as noise "
            "before trial-wise normalization. Set to 0 to disable."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Determine variant
    use_mass = args.top_k == 0

    # Resolve output path
    model_short_name = args.model.split("/")[-1]
    dataset_suffix = f"_{args.dataset}" if args.dataset != "niah" else ""
    variant_suffix = "_mass" if use_mass else f"_topk{args.top_k}"
    pooled_suffix = "_pooled" if args.pooled else ""
    if args.output is None:
        output_path = (
            Path("retrieval_heads")
            / f"{model_short_name}_contrastive{dataset_suffix}{variant_suffix}{pooled_suffix}.json"
        )
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    # Resolve haystack dir default
    if args.haystack_dir is None:
        args.haystack_dir = Path("data/haystack_for_detect")

    console.rule("[bold]Contrastive Retrieval Head Detection[/bold]")

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

    # Optional stratified sampling
    if args.num_examples is not None:
        dataset_trials = stratified_sample(
            dataset_trials,
            args.num_examples,
            seed=args.seed,
        )
        dataset_info += f" → sampled {len(dataset_trials)} trials"

    variant_label = "mass" if use_mass else f"top-k (k={args.top_k})"
    config_table = Table(title="Configuration", show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="bold")
    config_table.add_column("Value")
    config_table.add_row("Model", args.model)
    config_table.add_row("Method", f"contrastive ({variant_label})")
    config_table.add_row("Dataset", args.dataset)
    config_table.add_row("Output", str(output_path))
    config_table.add_row("Context range", f"{args.min_length} – {args.max_length} tokens")
    config_table.add_row("Dataset info", dataset_info)
    config_table.add_row("Total trials", str(len(dataset_trials)))
    config_table.add_row("Prefill backend", f"{args.prefill_attn_impl} (decode uses eager)")
    config_table.add_row("Epsilon", str(args.epsilon))
    delta_label = f"{args.delta_multiplier}× uniform baseline" if args.delta_multiplier > 0 else "disabled"
    config_table.add_row("Signal gate (δ)", delta_label)
    baseline_label = (
        f"contrastive (A⁺ − A⁻) when T_¬ans ≥ {args.min_baseline_steps}, else A⁺ only"
        if args.min_baseline_steps > 0
        else "always contrastive (A⁺ − A⁻)"
    )
    config_table.add_row("Scoring mode", baseline_label)
    console.print(config_table)

    # Load model
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    console.print(f"\nLoading model ({args.dtype}) ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
    tokenizer_has_bos = tokenizer_adds_bos(tokenizer)
    period_tokens = get_period_tokens(
        args.model,
        tokenizer,
        args.use_hardcoded_periods,
        args.force_llama2_periods,
    )

    # Token caches for efficiency
    haystack_token_cache: dict[str, list[int]] = {}
    haystack_period_cache: dict[str, list[int]] = {}
    needle_token_cache: dict[str, list[int]] = {}
    question_token_cache: dict[str, list[int]] = {}

    visible_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    device_map = args.device_map

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="eager",  # Required for output_attentions=True
        trust_remote_code=True,
    ).eval()

    # Extract num_layers and num_heads
    num_layers, num_heads = extract_model_config(model)

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

    num_passed = 0
    num_no_answer_steps = 0
    num_below_delta = 0
    num_used_baseline = 0  # trials scored with contrastive A+ - A-
    num_used_aplus_only = 0  # trials scored with A+ only (short generation)

    # Track trial IDs of passing trials (aligned with head_counter score lists)
    # Enables post-hoc analysis linking scores to trial metadata.
    passed_trial_ids: list[str] = []

    # A+/A- diagnostics: track per-trial top-head values for analysis
    aplus_aminus_log: list[dict] = []

    # Pooled mode accumulators: running sums + counts for global contrast
    if args.pooled:
        pooled_ans_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
        pooled_ans_count = 0
        pooled_notans_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
        pooled_notans_count = 0
        # Store per-step scores for bootstrap CI (list of (num_layers, num_heads) arrays)
        pooled_ans_steps: list[np.ndarray] = []
        pooled_notans_steps: list[np.ndarray] = []

    if not trials:
        console.print("[yellow]All trials already completed.[/yellow]")
    else:
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
                "Detecting retrieval heads (contrastive)",
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

                # --- Build context with needle inserted ---
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

                # Compose the prompt in token space (tracked needle position).
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
                    add_bos=tokenizer_has_bos,
                )
                full_tokens = input_ids[0].tolist()

                # NOTE: Unlike detect_retrieval_heads.py, we do NOT narrow
                # the needle span to answer tokens for NIAH. The contrastive
                # method measures attention mass over the full needle span
                # and does not require token-identity matching, so the full
                # needle is the correct scoring region for both datasets.

                if needle_start < 0 or needle_end <= needle_start:
                    console.print(f"[yellow]Warning: invalid needle span " f"({trial.trial_id}). Skipping.[/yellow]")
                    completed_trials.append(trial.trial_id)
                    processed_this_run += 1
                    progress.advance(ptask)
                    continue

                # --- Run contrastive detection ---
                try:
                    trial_result = detect_single_trial_contrastive(
                        model,
                        tokenizer,
                        input_ids,
                        needle_start,
                        needle_end,
                        num_layers,
                        num_heads,
                        trial.answer_text,
                        args.prefill_attn_impl,
                        top_k=args.top_k,
                        use_mass=use_mass,
                        min_baseline_steps=args.min_baseline_steps,
                        num_permutations=args.num_permutations if not args.pooled else 0,
                        permutation_percentile=args.permutation_percentile,
                        return_step_scores=args.pooled,
                        max_decode_steps=args.max_decode_steps,
                        newline_token_id=newline_token_id,
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

                R_tau = trial_result.R_tau
                generated_text = trial_result.generated_text

                # --- ROUGE gate ---
                rouge_result = scorer.score(trial.answer_text, generated_text)["rouge1"].recall * 100

                if rouge_result > args.rouge_threshold and trial_result.num_answer_steps > 0:

                    if args.pooled:
                        # --- Pooled mode: accumulate per-step scores globally ---
                        num_passed += 1
                        passed_trial_ids.append(trial.trial_id)
                        ss = trial_result.step_scores  # (num_steps, L, H)
                        for idx in trial_result.t_ans:
                            pooled_ans_sum += ss[idx]
                            pooled_ans_count += 1
                            pooled_ans_steps.append(ss[idx])
                        for idx in trial_result.t_not_ans:
                            pooled_notans_sum += ss[idx]
                            pooled_notans_count += 1
                            pooled_notans_steps.append(ss[idx])

                    else:
                        # --- Per-trial mode: gate, normalize/raw, accumulate ---
                        context_len = len(full_tokens)
                        needle_len = needle_end - needle_start
                        delta = args.delta_multiplier * (needle_len / context_len)

                        if args.delta_multiplier > 0 and R_tau.max() < delta:
                            num_below_delta += 1
                        else:
                            num_passed += 1
                            passed_trial_ids.append(trial.trial_id)
                            if trial_result.used_baseline:
                                num_used_baseline += 1
                            else:
                                num_used_aplus_only += 1

                            # Accumulate scores (normalized or raw)
                            if args.no_normalize:
                                scores_to_store = R_tau
                            else:
                                total_score = R_tau.sum() + args.epsilon
                                scores_to_store = R_tau / total_score

                            for layer_idx in range(num_layers):
                                for head_idx in range(num_heads):
                                    key = f"{layer_idx}-{head_idx}"
                                    head_counter[key].append(float(scores_to_store[layer_idx, head_idx]))

                        if not args.pooled:
                            # Log A+/A- diagnostics for the top head in this trial
                            top_idx = np.unravel_index(R_tau.argmax(), R_tau.shape)
                            log_entry = {
                                "trial_id": trial.trial_id,
                                "num_steps": trial_result.num_total_steps,
                                "num_ans_steps": trial_result.num_answer_steps,
                                "num_not_ans_steps": trial_result.num_not_answer_steps,
                                "used_baseline": trial_result.used_baseline,
                                "top_head": f"{top_idx[0]}-{top_idx[1]}",
                                "top_A_plus": float(trial_result.A_plus[top_idx]),
                                "top_R_tau": float(R_tau[top_idx]),
                            }
                            if trial_result.A_minus is not None:
                                log_entry["top_A_minus"] = float(trial_result.A_minus[top_idx])
                            aplus_aminus_log.append(log_entry)

                elif rouge_result > args.rouge_threshold and trial_result.num_answer_steps == 0:
                    num_no_answer_steps += 1

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
        console.print(f"\n[green]{num_passed}[/green] / {len(trials)} trials passed all gates")
        if num_no_answer_steps > 0:
            console.print(
                f"[yellow]{num_no_answer_steps} trials passed ROUGE but had "
                f"0 answer steps identified (skipped)[/yellow]"
            )
        if num_below_delta > 0:
            console.print(
                f"[yellow]{num_below_delta} trials passed ROUGE + answer steps but "
                f"fell below signal gate δ (skipped)[/yellow]"
            )
        if num_passed > 0:
            console.print(
                f"Scoring mode breakdown: {num_used_baseline} contrastive (A⁺ − A⁻), " f"{num_used_aplus_only} A⁺ only"
            )

        # A+/A- diagnostics summary
        if aplus_aminus_log:
            aplus_vals = [e["top_A_plus"] for e in aplus_aminus_log]
            aminus_vals = [e["top_A_minus"] for e in aplus_aminus_log if "top_A_minus" in e]
            step_counts = [e["num_steps"] for e in aplus_aminus_log]
            ans_counts = [e["num_ans_steps"] for e in aplus_aminus_log]

            diag_table = Table(title="A⁺/A⁻ Diagnostics (top head per trial)")
            diag_table.add_column("Metric")
            diag_table.add_column("Value", justify="right")
            diag_table.add_row("Trials analysed", str(len(aplus_aminus_log)))
            diag_table.add_row("Mean decode steps", f"{sum(step_counts)/len(step_counts):.1f}")
            diag_table.add_row("Mean answer steps", f"{sum(ans_counts)/len(ans_counts):.1f}")
            diag_table.add_row("Mean A⁺ (top head)", f"{sum(aplus_vals)/len(aplus_vals):.6f}")
            diag_table.add_row("Max A⁺ (top head)", f"{max(aplus_vals):.6f}")
            if aminus_vals:
                diag_table.add_row("Mean A⁻ (top head)", f"{sum(aminus_vals)/len(aminus_vals):.6f}")
                diag_table.add_row("Max A⁻ (top head)", f"{max(aminus_vals):.6f}")
                diffs = [a - b for a, b in zip(aplus_vals[: len(aminus_vals)], aminus_vals)]
                diag_table.add_row("Mean A⁺ − A⁻ (top head)", f"{sum(diffs)/len(diffs):.6f}")
            console.print(diag_table)

        timing_table = Table(title="Timing Summary")
        timing_table.add_column("Metric")
        timing_table.add_column("Value", justify="right")
        timing_table.add_row("Trials processed (this run)", str(processed_this_run))
        timing_table.add_row("OOM skips", str(oom_count))
        timing_table.add_row("Below signal gate (δ)", str(num_below_delta))
        timing_table.add_row("Elapsed", format_duration(total_elapsed_s))
        timing_table.add_row("Avg time / trial", f"{avg_trial_s:.2f}s")
        timing_table.add_row("Throughput", f"{throughput:.3f} trials/s")
        console.print(timing_table)

    # --- Pooled mode: compute global contrast + bootstrap CI ---
    pooled_scores = None
    pooled_ci_lower = None
    pooled_ci_upper = None
    if args.pooled and pooled_ans_count > 0 and pooled_notans_count > 0:
        global_a_plus = (pooled_ans_sum / pooled_ans_count).astype(np.float32)
        global_a_minus = (pooled_notans_sum / pooled_notans_count).astype(np.float32)
        pooled_scores = (global_a_plus - global_a_minus).astype(np.float32)

        console.print(
            f"\n[bold]Pooled contrast:[/bold] {pooled_ans_count} answer steps, "
            f"{pooled_notans_count} non-answer steps across {num_passed} trials"
        )

        # Bootstrap CI: resample answer and non-answer step pools
        n_bootstrap = 1000
        rng = np.random.RandomState(args.seed)
        ans_arr = np.stack(pooled_ans_steps)  # (n_ans, L, H)
        notans_arr = np.stack(pooled_notans_steps)  # (n_notans, L, H)
        boot_diffs = np.empty((n_bootstrap, num_layers, num_heads), dtype=np.float32)
        for b in range(n_bootstrap):
            boot_ans = ans_arr[rng.randint(0, len(ans_arr), size=len(ans_arr))].mean(axis=0)
            boot_notans = notans_arr[rng.randint(0, len(notans_arr), size=len(notans_arr))].mean(axis=0)
            boot_diffs[b] = boot_ans - boot_notans
        pooled_ci_lower = np.percentile(boot_diffs, 2.5, axis=0).astype(np.float32)
        pooled_ci_upper = np.percentile(boot_diffs, 97.5, axis=0).astype(np.float32)

        # How many heads have CI entirely above 0?
        significant = (pooled_ci_lower > 0).sum()
        console.print(
            f"[bold]Bootstrap 95% CI:[/bold] {significant}/{num_layers * num_heads} heads " f"with CI entirely above 0"
        )

        # Populate scores_dict and head_counter for the standard output path
        for layer in range(num_layers):
            for head in range(num_heads):
                key = f"{layer}-{head}"
                # Store single score (not a list of per-trial scores)
                head_counter[key] = [float(pooled_scores[layer, head])]

    # --- Save final results (envelope format) ---
    scores_dict = {}
    for layer in range(num_layers):
        for head in range(num_heads):
            key = f"{layer}-{head}"
            scores_dict[key] = head_counter.get(key, [])

    result = {
        "meta": {
            "method": "contrastive_pooled" if args.pooled else "contrastive",
            "variant": "mass" if use_mass else "topk",
            "top_k": None if use_mass else args.top_k,
            "pooled": args.pooled,
            "pooled_ans_steps": pooled_ans_count if args.pooled else None,
            "pooled_notans_steps": pooled_notans_count if args.pooled else None,
            "pooled_significant_heads": int((pooled_ci_lower > 0).sum()) if pooled_ci_lower is not None else None,
            "epsilon": args.epsilon,
            "delta_multiplier": args.delta_multiplier,
            "min_baseline_steps": args.min_baseline_steps,
            "num_permutations": args.num_permutations,
            "permutation_percentile": args.permutation_percentile,
            "num_below_delta": num_below_delta,
            "normalize": not args.no_normalize,
            "num_used_baseline": num_used_baseline,
            "num_used_aplus_only": num_used_aplus_only,
            "dataset": args.dataset,
            "num_trials_passed": num_passed,
            "num_trials_total": total_trials,
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

    # Add bootstrap CI to envelope when in pooled mode
    if pooled_ci_lower is not None:
        ci_dict = {}
        for layer in range(num_layers):
            for head in range(num_heads):
                key = f"{layer}-{head}"
                ci_dict[key] = [
                    float(pooled_ci_lower[layer, head]),
                    float(pooled_ci_upper[layer, head]),
                ]
        result["confidence_intervals"] = ci_dict

    output_path.write_text(json.dumps(result, indent=2))
    console.print(f"\n[bold green]Saved contrastive retrieval head scores to {output_path}[/bold green]")

    # Save A+/A- diagnostics log as JSONL alongside main output
    if aplus_aminus_log:
        diag_path = output_path.with_suffix(".diagnostics.jsonl")
        with open(diag_path, "w") as f:
            for entry in aplus_aminus_log:
                f.write(json.dumps(entry) + "\n")
        console.print(f"Saved A⁺/A⁻ diagnostics to {diag_path} ({len(aplus_aminus_log)} trials)")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Print top retrieval heads
    # Final aggregation (Eq. 4 in tex):
    # R_{l,h} = (1/|T_pass|) Σ_{τ∈T_pass} R̃^τ_{l,h}
    scored = [(key, float(np.mean(scores)) if scores else 0.0) for key, scores in scores_dict.items()]
    scored.sort(key=lambda x: x[1], reverse=True)

    table = Table(title="Top 20 Retrieval Heads (contrastive)")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Layer-Head")
    table.add_column("Mean Score", justify="right")
    table.add_column("Trials", justify="right")

    for i, (key, mean_score) in enumerate(scored[:20]):
        table.add_row(
            str(i + 1),
            key,
            f"{mean_score:.6f}",
            str(len(scores_dict[key])),
        )
    console.print(table)

    # Validate output is loadable
    from locos_eval.retrieval_heads import load_retrieval_heads

    heads = load_retrieval_heads(str(output_path), num_heads=10)
    console.print(f"\nValidation: load_retrieval_heads() returned {len(heads)} heads")
    console.print(f"Top 10: {heads}")


if __name__ == "__main__":
    main()
