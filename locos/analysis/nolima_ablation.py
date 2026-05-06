#!/usr/bin/env python3
"""NoLiMa ablation: measure retrieval performance with varying head selections.

Tests whether the identified retrieval heads are actually important by ablating
them (modifying their query projections via forward hooks) and measuring how the
model's NoLiMa retrieval performance changes. No contrastive decoding — just
standard greedy generation with ablated heads.

Two ablation modes are supported:
  - ``zero``: Set q_proj output to 0, producing uniform attention. Simple but
    out-of-distribution (the model never saw zero queries during training).
  - ``mean``: Replace q_proj output with the mean activation computed from a
    calibration pass over a sample of trials. More in-distribution — the head
    produces its "average" query rather than a pathological zero vector.

The baseline run (``--include-baseline``) generates without any ablation.

Results are cached to a JSON file so runs are never repeated. The companion
script ``plot_nolima_ablation.py`` reads these cached results.

Usage:
    # Ablation by top-k with zero ablation (default)
    python locos/run_nolima_ablation.py \\
        --model Qwen/Qwen3-8B \\
        --heads retrieval_heads/Qwen3-8B_nolima.json \\
        --mode top-k --values 1 5 10 20 50 100

    # Mean ablation (more in-distribution)
    python locos/run_nolima_ablation.py \\
        --model Qwen/Qwen3-8B \\
        --heads retrieval_heads/Qwen3-8B_nolima.json \\
        --mode top-k --values 1 5 10 20 50 100 \\
        --ablation-mode mean --num-calibration 50

    # Include unmasked baseline
    python locos/run_nolima_ablation.py \\
        --model Qwen/Qwen3-8B \\
        --heads retrieval_heads/Qwen3-8B_nolima.json \\
        --mode top-k --values 1 5 10 20 50 100 \\
        --include-baseline

    # Bottom-k control (specificity: ablate least important heads)
    python locos/run_nolima_ablation.py \\
        --model Qwen/Qwen3-8B \\
        --heads retrieval_heads/Qwen3-8B_nolima.json \\
        --mode bottom-k --values 1 5 10 20 50 100 \\
        --include-baseline

    # Random heads control (no --heads needed)
    python locos/run_nolima_ablation.py \\
        --model Qwen/Qwen3-8B \\
        --mode top-k --values 1 5 10 20 50 100 \\
        --random-heads --include-baseline

    # Smaller run for debugging
    python locos/run_nolima_ablation.py \\
        --model Qwen/Qwen3-8B \\
        --heads retrieval_heads/Qwen3-8B_nolima.json \\
        --mode top-k --values 1 5 10 \\
        --limit 50

Requires: GPU, transformers, rouge-score
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rouge_score import rouge_scorer

from locos.utils.common import (
    extract_model_config,
)
from locos.utils.common import load_checkpoint as _load_checkpoint_generic
from locos.utils.common import (
    load_model_and_tokenizer,
)
from locos.utils.common import save_checkpoint as _save_checkpoint_generic
from locos.utils.model_utils import get_decoder_layers, get_stop_token_ids

console = Console()


# ---------------------------------------------------------------------------
# Head selection helpers
# ---------------------------------------------------------------------------


def load_all_head_scores(json_path: str | Path) -> list[tuple[str, float]]:
    """Load retrieval head JSON, return sorted (key, mean_score) pairs."""
    with open(json_path) as f:
        data = json.load(f)

    if "scores" in data and isinstance(data["scores"], dict):
        scores_data = data["scores"]
    else:
        scores_data = data

    scored = []
    for key, values in scores_data.items():
        mean = sum(values) / len(values) if values else 0.0
        scored.append((key, mean))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def select_heads(
    all_scored: list[tuple[str, float]],
    mode: str,
    value: float,
) -> list[tuple[int, int]]:
    """Select heads by top-k, bottom-k, or threshold.

    Args:
        all_scored: Sorted (key, mean_score) pairs from load_all_head_scores
            (highest score first).
        mode: "top-k" (highest scoring), "bottom-k" (lowest scoring,
            specificity control), or "threshold" (all heads >= value).
        value: k (int) for top-k/bottom-k, or threshold (float) for threshold mode.

    Returns:
        List of (layer, head) tuples.
    """
    if mode == "top-k":
        k = int(value)
        selected = all_scored[:k]
    elif mode == "bottom-k":
        k = int(value)
        selected = all_scored[-k:] if k > 0 else []
    else:
        selected = [(key, s) for key, s in all_scored if s >= value]

    result = []
    for key, _ in selected:
        l, h = key.split("-")
        result.append((int(l), int(h)))
    return result


def select_random_heads(
    k: int,
    num_layers: int,
    num_heads: int,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Select k random heads uniformly from all layers and heads.

    Uses a fixed seed for reproducibility so the same random set is used
    across runs (and cached correctly).

    Args:
        k: Number of heads to select.
        num_layers: Total number of layers in the model.
        num_heads: Total number of attention heads per layer.
        seed: Random seed (deterministic for caching).

    Returns:
        List of (layer, head) tuples.
    """
    import random as _random

    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    assert k <= len(all_heads), f"Cannot select {k} random heads from {len(all_heads)} total"
    rng = _random.Random(seed)
    return rng.sample(all_heads, k)


def group_heads_by_layer(heads: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Convert [(layer, head), ...] to {layer: [head_idx, ...]}."""
    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, head in heads:
        by_layer[layer].append(head)
    return dict(by_layer)


# ---------------------------------------------------------------------------
# Ablation hooks: zero and mean ablation modes
# ---------------------------------------------------------------------------


def make_zero_hook(heads_to_zero: list[int], num_heads: int, head_dim: int):
    """Create a forward hook that zeros q_proj output for specified heads.

    Installed on ``layer.self_attn.q_proj``. The hook reshapes the flat
    projection output into [batch, seq, num_heads, head_dim], zeros the
    selected head slices, and reshapes back. Zeroing before rotary embedding
    is equivalent to zeroing after (rotating zero gives zero).
    """

    def hook(module, args, output):
        shape = output.shape
        out = output.view(shape[0], shape[1], num_heads, head_dim)
        out[:, :, heads_to_zero, :] = 0.0
        return out.view(shape)

    return hook


def make_mean_hook(
    heads_to_ablate: list[int],
    num_heads: int,
    head_dim: int,
    mean_activations: torch.Tensor,
):
    """Create a forward hook that replaces q_proj output with mean activations.

    Installed on ``layer.self_attn.q_proj``. Replaces the specified heads'
    query projections with pre-computed mean activations from a calibration
    pass. This is a less out-of-distribution intervention than zeroing —
    the head sees its "average" query rather than a zero vector it never
    encountered during training.

    Args:
        heads_to_ablate: Head indices to replace.
        num_heads: Total number of attention heads.
        head_dim: Dimension per head.
        mean_activations: Pre-computed mean q_proj activation for this layer,
            shape (num_heads, head_dim). Broadcast across batch and seq dims.
    """
    assert mean_activations.shape == (
        num_heads,
        head_dim,
    ), f"Expected ({num_heads}, {head_dim}), got {mean_activations.shape}"

    def hook(module, args, output):
        shape = output.shape
        out = output.view(shape[0], shape[1], num_heads, head_dim)
        for h in heads_to_ablate:
            out[:, :, h, :] = mean_activations[h]
        return out.view(shape)

    return hook


def _get_model_head_config(model) -> tuple[int, int]:
    """Extract (num_heads, head_dim) from model config."""
    config = model.config
    text_config = getattr(config, "text_config", config)
    num_heads = getattr(text_config, "num_attention_heads", None) or config.num_attention_heads
    # Prefer explicit head_dim (e.g. Qwen3 where head_dim != hidden_size // num_heads)
    head_dim = getattr(text_config, "head_dim", None) or getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(text_config, "hidden_size", None) or config.hidden_size
        head_dim = hidden_size // num_heads
    return num_heads, head_dim


def calibrate_mean_activations(
    model,
    tokenizer,
    trials: list,
    num_calibration: int = 50,
    seed: int = 42,
) -> dict[int, torch.Tensor]:
    """Compute per-layer, per-head mean q_proj activations from calibration data.

    Runs a set of trials through the model with capture hooks on every
    q_proj, accumulating running means. Used as the replacement value
    for mean-ablation.

    Args:
        model: HuggingFace model.
        tokenizer: Tokenizer instance.
        trials: NoLiMa trials to use for calibration.
        num_calibration: Number of trials to use (sampled from trials).
        seed: Random seed for sampling calibration trials.

    Returns:
        Dict mapping layer_idx to mean activation tensor (num_heads, head_dim).
    """
    import random as _random

    from locos.utils.model_utils import get_input_device
    from locos.utils.needle_utils import insert_needle

    num_heads, head_dim = _get_model_head_config(model)
    decoder_layers = get_decoder_layers(model)
    num_layers = len(decoder_layers)

    # Running accumulators: sum and count per layer
    running_sum: dict[int, torch.Tensor] = {
        i: torch.zeros(num_heads, head_dim, dtype=torch.float64) for i in range(num_layers)
    }
    running_count: dict[int, int] = {i: 0 for i in range(num_layers)}

    # Capture hooks: accumulate mean across batch and sequence dimensions
    handles = []
    for layer_idx, layer in enumerate(decoder_layers):

        def make_capture_hook(lidx):
            def hook(module, args, output):
                # output shape: [batch, seq_len, num_heads * head_dim]
                out = output.detach().view(output.shape[0], output.shape[1], num_heads, head_dim)
                # Mean over batch and sequence → (num_heads, head_dim)
                running_sum[lidx] += out.float().mean(dim=(0, 1)).cpu()
                running_count[lidx] += 1
                return None  # Don't modify output

            return hook

        handle = layer.self_attn.q_proj.register_forward_hook(make_capture_hook(layer_idx))
        handles.append(handle)

    # Sample calibration trials
    rng = _random.Random(seed)
    cal_trials = list(trials)
    rng.shuffle(cal_trials)
    cal_trials = cal_trials[:num_calibration]

    device = get_input_device(model)

    # Cap calibration sequence length to avoid OOM — we only need representative
    # activations, not full-length contexts. We hook q_proj output (pre-RoPE),
    # which is position-invariant, but the residual stream input varies with
    # context length, so we use 5000 tokens to better match the eval distribution.
    max_cal_tokens = 5000

    console.print(f"Calibrating mean activations from {len(cal_trials)} trials (max {max_cal_tokens} tokens)...")
    with torch.inference_mode():
        for trial in cal_trials:
            context_tokens, _, _ = insert_needle(
                haystack_text=trial.haystack_text,
                needle=trial.needle_text,
                context_length=min(trial.context_length, max_cal_tokens),
                depth_percent=trial.depth_percent,
                tokenizer=tokenizer,
            )
            context_text = tokenizer.decode(context_tokens, skip_special_tokens=True)
            if trial.prompt_template:
                prompt_text = trial.prompt_template.replace("{haystack}", context_text)
            else:
                # NIAH: trial.question already contains full 'Question: ...\nAnswer:' scaffolding
                prompt_text = f"{context_text}\n\n{trial.question}"

            # Tokenize as raw text (no chat template) — matches detection scripts
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
            # Truncate to max_cal_tokens as a safety net (template may add tokens)
            if input_ids.shape[1] > max_cal_tokens:
                input_ids = input_ids[:, :max_cal_tokens]

            # Single forward pass (no generation needed — just collect activations)
            model(input_ids=input_ids, output_attentions=False, return_dict=True)

            # Free memory between calibration trials
            del input_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Remove capture hooks
    for h in handles:
        h.remove()

    # Compute means
    mean_activations: dict[int, torch.Tensor] = {}
    for layer_idx in range(num_layers):
        assert running_count[layer_idx] > 0, f"No calibration data for layer {layer_idx}"
        mean_activations[layer_idx] = (running_sum[layer_idx] / running_count[layer_idx]).float()

    console.print(
        f"[green]Calibrated mean q_proj activations " f"({len(cal_trials)} trials, {num_layers} layers)[/green]"
    )
    return mean_activations


def install_ablation_hooks(
    model,
    heads_by_layer: dict[int, list[int]],
    ablation_mode: str = "zero",
    mean_activations: dict[int, torch.Tensor] | None = None,
) -> list[torch.utils.hooks.RemovableHook]:
    """Install q_proj forward hooks to ablate retrieval heads.

    Args:
        model: HuggingFace model.
        heads_by_layer: {layer_idx: [head_indices]} to ablate.
        ablation_mode: "zero" (set q_proj to 0) or "mean" (replace with
            pre-computed mean activation).
        mean_activations: Required when ablation_mode="mean". Dict mapping
            layer_idx to mean activation tensor (num_heads, head_dim).

    Returns:
        List of hook handles for later removal.
    """
    assert ablation_mode in ("zero", "mean"), f"Unknown ablation mode: {ablation_mode}"
    if ablation_mode == "mean":
        assert mean_activations is not None, "mean_activations required for ablation_mode='mean'"

    num_heads, head_dim = _get_model_head_config(model)
    decoder_layers = get_decoder_layers(model)

    hooks = []
    for layer_idx, layer in enumerate(decoder_layers):
        heads = heads_by_layer.get(layer_idx, [])
        if not heads:
            continue
        attn = layer.self_attn

        if ablation_mode == "zero":
            hook_fn = make_zero_hook(heads, num_heads, head_dim)
        else:
            hook_fn = make_mean_hook(
                heads,
                num_heads,
                head_dim,
                mean_activations[layer_idx].to(next(model.parameters()).device),
            )

        hook = attn.q_proj.register_forward_hook(hook_fn)
        hooks.append(hook)

    return hooks


def remove_hooks(hooks: list[torch.utils.hooks.RemovableHook]) -> None:
    """Remove all installed hooks."""
    for h in hooks:
        h.remove()
    hooks.clear()


# ---------------------------------------------------------------------------
# NoLiMa evaluation
# ---------------------------------------------------------------------------


def build_nolima_samples(
    nolima_dir: Path,
    max_length: int,
    num_lengths: int,
    num_depths: int,
    question_type: str,
    limit: int | None,
    seed: int,
) -> list:
    """Build NoLiMa dataset trials for evaluation."""
    from locos.utils.datasets import build_nolima_dataset, stratified_sample

    context_lengths = np.round(np.linspace(1000, max_length, num=num_lengths, endpoint=True)).astype(int).tolist()
    depth_percents = np.round(np.linspace(0, 100, num=num_depths, endpoint=True)).astype(int).tolist()

    trials = build_nolima_dataset(
        nolima_dir,
        context_lengths,
        depth_percents,
        question_type=question_type,
        max_tokens=max_length,
        max_characters_per_entry=1,
        seed=seed,
    )

    if limit is not None and limit < len(trials):
        trials = stratified_sample(trials, limit, seed=seed)

    return trials


def build_niah_samples(
    haystack_dir: Path,
    max_length: int,
    num_lengths: int,
    num_depths: int,
    limit: int | None,
    seed: int,
) -> list:
    """Build NIAH (Wu et al.) dataset trials for evaluation."""
    from locos.utils.datasets import build_niah_dataset, stratified_sample

    context_lengths = np.round(np.linspace(1000, max_length, num=num_lengths, endpoint=True)).astype(int).tolist()
    depth_percents = np.round(np.linspace(0, 100, num=num_depths, endpoint=True)).astype(int).tolist()

    trials = build_niah_dataset(
        haystack_dir,
        context_lengths,
        depth_percents,
        max_tokens=max_length,
    )

    if limit is not None and limit < len(trials):
        trials = stratified_sample(trials, limit, seed=seed)

    return trials


def generate_for_trial(
    model,
    tokenizer,
    trial,
    max_tokens: int,
    prefill_attn_impl: str = "sdpa",
    stop_token_ids: set[int] | None = None,
) -> str:
    """Generate text for a single retrieval trial using HF transformers.

    Handles both NoLiMa (prompt_template with {haystack}) and NIAH (simple
    context + question concatenation — trial.question already includes the
    'Based on the content ... Question: ...\\nAnswer:' scaffolding).

    Uses prefill with optional efficient backend, then eager decode.
    """
    from locos.utils.model_utils import get_input_device, set_model_attn_impl
    from locos.utils.needle_utils import insert_needle

    # Insert needle into haystack
    context_tokens, _, _ = insert_needle(
        haystack_text=trial.haystack_text,
        needle=trial.needle_text,
        context_length=trial.context_length,
        depth_percent=trial.depth_percent,
        tokenizer=tokenizer,
    )

    # Build the full prompt text
    context_text = tokenizer.decode(context_tokens, skip_special_tokens=True)
    if trial.prompt_template:
        # NoLiMa: expand task template (has full framing, system prompt, etc.)
        prompt_text = trial.prompt_template.replace("{haystack}", context_text)
    else:
        # NIAH: trial.question already contains the 'Question: ...\nAnswer:'
        # scaffolding (matches behavioral.py's full_tokens = context + question)
        prompt_text = f"{context_text}\n\n{trial.question}"

    # Apply chat template for models that need it (e.g. Qwen3 generates <think>
    # even with raw text). Disable thinking mode where supported.
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt_text}]
        chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if "enable_thinking" in tokenizer.chat_template:
            chat_kwargs["enable_thinking"] = False
        prompt_text = tokenizer.apply_chat_template(messages, **chat_kwargs)

    input_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False)
    device = get_input_device(model)
    input_ids = input_ids.to(device)

    # Prefill with efficient backend
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
        except Exception:
            if switched:
                set_model_attn_impl(model, "eager")
                prefill_out = model(
                    input_ids=input_ids[:, :-1],
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True,
                )
            else:
                raise
        past_kv = prefill_out.past_key_values

    # Switch back to eager for decode (hooks work with any backend, but
    # staying consistent with detection script)
    if prefill_attn_impl != "eager":
        set_model_attn_impl(model, "eager")

    # Autoregressive decode
    current_token = input_ids[:, -1:].clone()
    generated_ids = []
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

    with torch.inference_mode():
        for _ in range(max_tokens):
            outputs = model(
                input_ids=current_token,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            past_kv = outputs.past_key_values
            next_id = outputs.logits[0, -1].argmax().item()

            # --- Stop checks (before appending, so stop tokens stay out of generated_ids) ---
            step_token = tokenizer.convert_ids_to_tokens(next_id)
            if step_token == "<0x0A>" or next_id == newline_id:
                break
            if stop_token_ids and next_id in stop_token_ids:
                break

            generated_ids.append(next_id)
            current_token[0, 0] = next_id

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_nolima_eval(
    model,
    tokenizer,
    trials: list,
    max_tokens: int,
    prefill_attn_impl: str,
    label: str = "",
    debug_trials: int = 0,
    stop_token_ids: set[int] | None = None,
) -> list[dict]:
    """Run retrieval ablation evaluation and return per-trial results.

    Returns:
        List of dicts with keys: trial_id, output, target, rouge_l, rouge_1,
        rouge_1_recall (Wu et al.'s NIAH gating metric: rouge-1 recall).
    """
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    results = []

    desc = f"Evaluating {label}..." if label else "Evaluating..."
    for idx, trial in enumerate(track(trials, description=desc, console=console)):
        output = generate_for_trial(model, tokenizer, trial, max_tokens, prefill_attn_impl, stop_token_ids)

        scores = rouge.score(trial.answer_text, output)
        rouge_l = scores["rougeL"].fmeasure
        rouge_1 = scores["rouge1"].fmeasure
        rouge_1_recall = scores["rouge1"].recall
        results.append(
            {
                "trial_id": trial.trial_id,
                "output": output,
                "target": trial.answer_text,
                "rouge_l": rouge_l,
                "rouge_1": rouge_1,
                "rouge_1_recall": rouge_1_recall,
            }
        )

        if debug_trials > 0 and idx < debug_trials:
            rl_color = "green" if rouge_l > 0.5 else "red"
            console.print(
                f"\n[bold]--- Debug trial {idx + 1}/{debug_trials} ---[/bold]\n"
                f"  [dim]trial_id:[/dim]  {trial.trial_id}\n"
                f"  [dim]ctx_len:[/dim]   {trial.context_length}  "
                f"[dim]depth:[/dim] {trial.depth_percent}%\n"
                f"  [dim]target:[/dim]    {trial.answer_text!r}\n"
                f"  [dim]generated:[/dim] {output!r}\n"
                f"  [dim]ROUGE-L:[/dim]   [{rl_color}]{rouge_l:.4f}[/{rl_color}]  "
                f"[dim]ROUGE-1:[/dim] {rouge_1:.4f}"
            )

    return results


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def load_cache(cache_path: Path) -> dict:
    """Load results cache (wraps generic checkpoint). Structure: {run_key: {metrics...}}"""
    data = _load_checkpoint_generic(cache_path)
    return data if data is not None else {}


def save_cache(cache: dict, cache_path: Path) -> None:
    """Save results cache atomically (wraps generic checkpoint)."""
    _save_checkpoint_generic(cache, cache_path)


def run_key(
    mode: str,
    value: float,
    model: str,
    limit: int | None,
    ablation_mode: str = "zero",
    random_heads: bool = False,
) -> str:
    """Generate a unique cache key for a specific run configuration."""
    limit_str = f"_limit{limit}" if limit else ""
    abl_str = f"_{ablation_mode}" if ablation_mode != "zero" else ""
    rand_str = "_random" if random_heads else ""
    if mode == "top-k":
        return f"{model}__topk_{int(value)}{rand_str}{abl_str}{limit_str}"
    if mode == "bottom-k":
        return f"{model}__bottomk_{int(value)}{abl_str}{limit_str}"
    if mode == "baseline":
        return f"{model}__baseline{limit_str}"
    return f"{model}__thresh_{value}{rand_str}{abl_str}{limit_str}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="NoLiMa ablation: mask retrieval heads and measure retrieval performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument(
        "--heads", type=Path, default=None, help="Retrieval heads JSON file (required unless --random-heads)"
    )
    parser.add_argument(
        "--mode",
        choices=["top-k", "bottom-k", "threshold"],
        default="top-k",
        help=(
            "Head selection mode: top-k (highest scoring), "
            "bottom-k (lowest scoring, specificity control), or threshold "
            "(not required with --heads-list)"
        ),
    )
    parser.add_argument(
        "--random-heads",
        action="store_true",
        help=(
            "Control condition: select k random heads (uniform across all layers/heads) "
            "instead of top-k from a heads JSON. Uses --values as k counts. "
            "Deterministic per seed for reproducibility."
        ),
    )
    parser.add_argument(
        "--heads-list",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Explicit list of heads to ablate, as 'layer-head' strings "
            "(e.g., --heads-list 22-15 24-26 22-14 24-24). "
            "Runs a single ablation with exactly these heads. "
            "Ignores --heads, --mode, --values. Requires --label for cache key."
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label for this run (used in cache key and output). Required with --heads-list.",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        type=float,
        default=None,
        help="Values to sweep (k values for top-k, thresholds for threshold mode)",
    )
    parser.add_argument(
        "--include-baseline", action="store_true", help="Also run an unmasked baseline (no heads ablated)"
    )
    parser.add_argument(
        "--ablation-mode",
        type=str,
        default="mean",
        choices=["zero", "mean"],
        help=(
            "How to ablate selected heads. "
            "'zero': set q_proj output to 0 (uniform attention, may be out-of-distribution). "
            "'mean': replace q_proj output with its mean activation from a calibration pass "
            "(more in-distribution intervention)."
        ),
    )
    parser.add_argument(
        "--num-calibration",
        type=int,
        default=50,
        help="Number of trials for calibrating mean activations (only used with --ablation-mode mean)",
    )
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate per sample")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (stratified)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="nolima",
        choices=["niah", "nolima"],
        help="Evaluation dataset: nolima (Adobe Research) or niah (Wu et al. Needle-in-a-Haystack)",
    )
    parser.add_argument("--nolima-dir", type=Path, default=Path("data/nolima"), help="Directory with NoLiMa data")
    parser.add_argument(
        "--haystack-dir",
        type=Path,
        default=Path("data/haystack_for_detect"),
        help="Directory with NIAH data (needles.jsonl + part1/part2/part3)",
    )
    parser.add_argument("--max-length", type=int, default=16000, help="Maximum context length for trials")
    parser.add_argument("--num-lengths", type=int, default=5, help="Number of context length intervals")
    parser.add_argument("--num-depths", type=int, default=5, help="Number of depth intervals")
    parser.add_argument("--question-type", type=str, default="onehop", choices=["onehop", "twohop", "twohop2"])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--prefill-attn-impl",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend for prefill (decode uses eager)",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        help="Transformers device_map strategy",
    )
    parser.add_argument(
        "--debug-trials",
        type=int,
        default=0,
        help="Print raw generated text, target, and ROUGE for the first N trials (for diagnosing model output)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("ablation_results"),
        help="Directory for caching results",
    )
    args = parser.parse_args()

    # --- Validate argument combinations ---
    if args.heads_list:
        assert args.label is not None, "--label is required with --heads-list (used as cache key)"
        # Parse heads list: "22-15" -> (22, 15)
        explicit_heads = []
        for h in args.heads_list:
            parts = h.split("-")
            assert len(parts) == 2, f"Invalid head format '{h}', expected 'layer-head'"
            explicit_heads.append((int(parts[0]), int(parts[1])))
        assert len(explicit_heads) > 0, "Empty heads list"
    elif not args.random_heads:
        assert args.heads is not None and args.heads.exists(), (
            f"Heads file required (and must exist) unless --random-heads or --heads-list is used. " f"Got: {args.heads}"
        )
    if not args.heads_list:
        assert args.values is not None and len(args.values) > 0, "--values is required unless --heads-list is used"

    model_short = args.model.split("/")[-1]

    if args.heads_list:
        heads_label = f"{model_short}_{args.label}"
        all_scored = None
    elif args.random_heads:
        heads_label = f"{model_short}_random_seed{args.seed}"
        all_scored = None
    else:
        heads_label = args.heads.stem  # e.g. "Qwen3-8B_nolima"
        all_scored = load_all_head_scores(args.heads)

    # Separate cache file per selection mode to avoid overwriting on HF upload
    if args.mode == "bottom-k":
        heads_label = f"{heads_label}_bottomk"

    cache_path = args.cache_dir / f"{args.dataset}_ablation_{heads_label}.json"
    cache = load_cache(cache_path)

    if args.heads_list:
        heads_display = f"explicit list ({len(explicit_heads)} heads): {args.heads_list[:5]}{'...' if len(args.heads_list) > 5 else ''}"
    elif args.random_heads:
        heads_display = "random (uniform)"
    else:
        heads_display = str(args.heads)
    console.print(
        Panel(
            f"[bold]Model:[/bold] {args.model}\n"
            f"[bold]Heads:[/bold] {heads_display}\n"
            f"[bold]Mode:[/bold] {'explicit list' if args.heads_list else args.mode}\n"
            f"[bold]Ablation:[/bold] {args.ablation_mode}\n"
            f"[bold]Values:[/bold] {args.values}\n"
            f"[bold]Include baseline:[/bold] {args.include_baseline}\n"
            f"[bold]Dataset:[/bold] {args.dataset}\n"
            f"[bold]Cache:[/bold] {cache_path}",
            title=f"[green]{args.dataset.upper()} Ablation[/green]",
        )
    )

    # Report head selection at each value (skip for random/explicit — needs model dims or N/A)
    if not args.random_heads and not args.heads_list:
        selection_table = Table(title="Head Selection Preview")
        selection_table.add_column("Value", justify="right")
        selection_table.add_column("Heads Selected", justify="right")
        selection_table.add_column("Top Head Score", justify="right")
        selection_table.add_column("Min Head Score", justify="right")

        for v in sorted(args.values):
            heads = select_heads(all_scored, args.mode, v)
            if heads:
                head_scores = [s for k, s in all_scored if k in {f"{l}-{h}" for l, h in heads}]
                selection_table.add_row(
                    str(int(v) if args.mode == "top-k" else v),
                    str(len(heads)),
                    f"{max(head_scores):.4f}" if head_scores else "—",
                    f"{min(head_scores):.4f}" if head_scores else "—",
                )
            else:
                selection_table.add_row(str(v), "0", "—", "—")
        console.print(selection_table)

    # Build trials (shared across all runs)
    console.rule(f"[bold]Building {args.dataset.upper()} Dataset[/bold]")
    if args.dataset == "niah":
        trials = build_niah_samples(
            haystack_dir=args.haystack_dir,
            max_length=args.max_length,
            num_lengths=args.num_lengths,
            num_depths=args.num_depths,
            limit=args.limit,
            seed=args.seed,
        )
    else:
        trials = build_nolima_samples(
            nolima_dir=args.nolima_dir,
            max_length=args.max_length,
            num_lengths=args.num_lengths,
            num_depths=args.num_depths,
            question_type=args.question_type,
            limit=args.limit,
            seed=args.seed,
        )
    console.print(f"Built {len(trials)} {args.dataset.upper()} trials")

    # Determine which runs are needed
    runs_to_do: list[tuple[str, float | None, bool]] = []
    # (label, value, masked)

    if args.heads_list:
        # Explicit heads list: single run with the given heads
        cache_key = f"{model_short}__{args.label}"
        if cache_key in cache:
            console.print(f"[dim]Skipping {args.label} (cached)[/dim]")
        else:
            runs_to_do.append((args.label, None, True))

        if args.include_baseline:
            baseline_key = run_key("baseline", 0, model_short, args.limit, args.ablation_mode, False)
            if baseline_key not in cache:
                runs_to_do.append(("baseline", None, False))
            else:
                console.print("[dim]Skipping baseline (cached)[/dim]")
    else:
        if args.include_baseline:
            key = run_key("baseline", 0, model_short, args.limit, args.ablation_mode, args.random_heads)
            if key not in cache:
                runs_to_do.append(("baseline", None, False))
            else:
                console.print("[dim]Skipping baseline (cached)[/dim]")

        mode_label = "random" if args.random_heads else args.mode
        for v in sorted(args.values):
            key = run_key(args.mode, v, model_short, args.limit, args.ablation_mode, args.random_heads)
            if key in cache:
                v_display = int(v) if args.mode == "top-k" else v
                console.print(f"[dim]Skipping {mode_label}={v_display} (cached)[/dim]")
            else:
                runs_to_do.append((f"{mode_label}={v}", v, True))

    if not runs_to_do:
        console.print("\n[green]All runs already cached![/green]")
        if not args.heads_list:
            _print_results_table(
                cache,
                args.mode,
                args.values,
                model_short,
                args.limit,
                args.include_baseline,
                args.ablation_mode,
                args.random_heads,
            )
        return

    console.print(f"\n{len(runs_to_do)} runs remaining")

    # Load model once, reuse across all runs
    console.rule("[bold]Loading Model[/bold]")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
    )
    stop_token_ids = get_stop_token_ids(tokenizer, model)
    console.print(f"[dim]Stop token IDs ({len(stop_token_ids)}): {stop_token_ids}[/dim]")
    console.print("[green]Model loaded[/green]")

    # Calibrate mean activations if needed (one-time cost before ablation runs)
    mean_activations = None
    if args.ablation_mode == "mean":
        any_masked_runs = any(masked for _, _, masked in runs_to_do)
        if any_masked_runs:
            console.rule("[bold]Calibrating Mean Activations[/bold]")
            mean_activations = calibrate_mean_activations(
                model,
                tokenizer,
                trials,
                num_calibration=args.num_calibration,
                seed=args.seed,
            )

    # For random heads, we need model dimensions (available after model load)
    num_layers_model, num_heads_model = extract_model_config(model)

    # Execute runs — install/remove hooks between runs
    for label, value, masked in runs_to_do:
        console.rule(f"[bold]Run: {label}[/bold]")

        hooks = []
        n_heads = 0
        if masked:
            if args.heads_list:
                heads = explicit_heads
                console.print(f"  Using explicit heads list ({len(heads)} heads)")
            elif args.random_heads:
                heads = select_random_heads(
                    int(value),
                    num_layers_model,
                    num_heads_model,
                    seed=args.seed,
                )
                console.print(f"  Selected {len(heads)} random heads (seed={args.seed})")
            else:
                heads = select_heads(all_scored, args.mode, value)
            if not heads:
                console.print(f"[yellow]No heads selected for {label}, skipping[/yellow]")
                continue
            n_heads = len(heads)
            heads_by_layer = group_heads_by_layer(heads)
            hooks = install_ablation_hooks(
                model,
                heads_by_layer,
                ablation_mode=args.ablation_mode,
                mean_activations=mean_activations,
            )
            source = "explicit" if args.heads_list else ("random" if args.random_heads else args.ablation_mode)
            console.print(f"  Ablating {n_heads} heads ({source}, {len(hooks)} layers with hooks)")
        else:
            console.print("  No ablation (baseline)")

        try:
            results = run_nolima_eval(
                model,
                tokenizer,
                trials,
                max_tokens=args.max_tokens,
                prefill_attn_impl=args.prefill_attn_impl,
                label=label,
                debug_trials=args.debug_trials,
                stop_token_ids=stop_token_ids,
            )
        finally:
            # Always remove hooks after run
            remove_hooks(hooks)

        # Compute aggregate metrics
        rouge_l_scores = [r["rouge_l"] for r in results]
        rouge_1_scores = [r["rouge_1"] for r in results]
        rouge_1_recalls = [r["rouge_1_recall"] for r in results]
        # Wu et al. / behavioral.py gating convention: ROUGE-1 recall > 0.5 = hit
        wu_accuracy = sum(1 for r in rouge_1_recalls if r > 0.5) / len(rouge_1_recalls)

        metrics = {
            "rouge_l_mean": sum(rouge_l_scores) / len(rouge_l_scores),
            "rouge_1_mean": sum(rouge_1_scores) / len(rouge_1_scores),
            "rouge_1_recall_mean": sum(rouge_1_recalls) / len(rouge_1_recalls),
            "wu_accuracy": wu_accuracy,
            "rouge_l_std": float(np.std(rouge_l_scores)),
            "dataset": args.dataset,
            "n_samples": len(results),
            "n_heads": n_heads,
            "mode": "explicit" if args.heads_list else (args.mode if masked else "baseline"),
            "ablation_mode": args.ablation_mode if masked else "none",
            "random_heads": args.random_heads,
            "heads_list": args.heads_list if args.heads_list and masked else None,
            "value": value if value is not None else 0,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        }

        # Cache key
        if not masked:
            key = run_key("baseline", 0, model_short, args.limit, args.ablation_mode, args.random_heads)
        elif args.heads_list:
            key = f"{model_short}__{args.label}"
        else:
            key = run_key(args.mode, value, model_short, args.limit, args.ablation_mode, args.random_heads)

        cache[key] = metrics
        save_cache(cache, cache_path)

        # Also save per-trial results
        safe_label = label.replace("=", "_").replace(".", "p")
        trial_path = args.cache_dir / f"{args.dataset}_ablation_{heads_label}_{safe_label}_trials.jsonl"
        trial_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trial_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        console.print(
            f"  ROUGE-L: {metrics['rouge_l_mean']:.4f} (std: {metrics['rouge_l_std']:.4f})  "
            f"ROUGE-1 recall: {metrics['rouge_1_recall_mean']:.4f}  "
            f"Wu acc (R1-recall>0.5): {metrics['wu_accuracy']:.4f}"
        )

    # Print final results table
    if not args.heads_list:
        _print_results_table(
            cache,
            args.mode,
            args.values,
            model_short,
            args.limit,
            args.include_baseline,
            args.ablation_mode,
            args.random_heads,
        )
    console.print(f"\n[green]Results cached at:[/green] {cache_path}")


def _print_results_table(
    cache: dict,
    mode: str,
    values: list[float],
    model_short: str,
    limit: int | None,
    include_baseline: bool,
    ablation_mode: str = "zero",
    random_heads: bool = False,
) -> None:
    """Print a summary table of all results."""
    abl_label = f" ({ablation_mode} ablation)" if ablation_mode != "zero" else ""
    rand_label = " [random control]" if random_heads else ""
    table = Table(title=f"NoLiMa Ablation Results{abl_label}{rand_label}")
    table.add_column("Config", style="bold")
    table.add_column("Heads Ablated", justify="right")
    table.add_column("ROUGE-L", justify="right")
    table.add_column("ROUGE-1", justify="right")
    table.add_column("Samples", justify="right", style="dim")

    baseline_rl = None
    if include_baseline:
        key = run_key("baseline", 0, model_short, limit, ablation_mode, random_heads)
        if key in cache:
            m = cache[key]
            baseline_rl = m["rouge_l_mean"]
            table.add_row(
                "Baseline (no ablation)",
                "0",
                f"{m['rouge_l_mean']:.4f}",
                f"{m['rouge_1_mean']:.4f}",
                str(m["n_samples"]),
            )

    mode_label = "random" if random_heads else mode
    for v in sorted(values):
        key = run_key(mode, v, model_short, limit, ablation_mode, random_heads)
        if key in cache:
            m = cache[key]
            v_display = str(int(v)) if mode == "top-k" else str(v)
            rl = m["rouge_l_mean"]
            rl_str = f"{rl:.4f}"
            if baseline_rl is not None:
                delta = rl - baseline_rl
                sign = "+" if delta > 0 else ""
                if delta > 0:
                    rl_str += f" [green]({sign}{delta:.4f})[/green]"
                elif delta < 0:
                    rl_str += f" [red]({sign}{delta:.4f})[/red]"
                else:
                    rl_str += f" ({sign}{delta:.4f})"
            table.add_row(
                f"{mode_label}={v_display}",
                str(m["n_heads"]),
                rl_str,
                f"{m['rouge_1_mean']:.4f}",
                str(m["n_samples"]),
            )

    console.print(table)


if __name__ == "__main__":
    main()
