#!/usr/bin/env python3
"""Parametric/arithmetic ablation: control experiment for retrieval head specificity.

Tests whether ablated heads are retrieval-specific or generically output-critical
by evaluating parametric recall (City-Country, PopQA) and arithmetic after head
ablation. If retrieval heads are truly retrieval-specific, these capabilities
should remain largely intact even as NoLiMa performance collapses.

Reuses the ablation infrastructure from nolima_ablation.py (hook installation,
head selection, caching) but evaluates on a different dataset with exact-match
scoring.

Usage:
    # Basic: mean-ablation with top-k sweep
    python locos/analysis/parametric_ablation.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --heads retrieval_heads/Meta-Llama-3-8B-Instruct.json \\
        --mode top-k --values 1 5 10 20 50 100 \\
        --ablation-mode mean --include-baseline

    # Random heads control
    python locos/analysis/parametric_ablation.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --mode top-k --values 1 5 10 20 50 100 \\
        --random-heads --include-baseline

    # Custom dataset
    python locos/analysis/parametric_ablation.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --heads retrieval_heads/Meta-Llama-3-8B-Instruct.json \\
        --mode top-k --values 5 20 50 \\
        --dataset aryopg/parametric-arithmetic-eval

Requires: GPU, transformers, datasets
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

import torch
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

from locos.analysis.nolima_ablation import (
    _get_model_head_config,
    group_heads_by_layer,
    install_ablation_hooks,
    load_all_head_scores,
    load_cache,
    remove_hooks,
    run_key,
    save_cache,
    select_heads,
    select_random_heads,
)
from locos.utils.common import (
    extract_model_config,
    load_model_and_tokenizer,
)
from locos.utils.model_utils import get_decoder_layers, get_stop_token_ids

console = Console()

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_prompt_config(path: Path | None = None) -> dict:
    """Load the YAML prompt config for parametric ablation."""
    if path is None:
        path = _PROMPT_DIR / "parametric_ablation.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def build_chat_messages(question: str, prompt_cfg: dict) -> list[dict]:
    """Build chat messages list from the prompt config and a question.

    Returns a messages list suitable for tokenizer.apply_chat_template():
      [system, example_user, example_assistant, ..., user_question]
    """
    messages: list[dict] = []

    # System prompt
    if "system" in prompt_cfg:
        messages.append({"role": "system", "content": prompt_cfg["system"].strip()})

    # Few-shot examples
    for ex in prompt_cfg.get("examples", []):
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})

    # Actual question
    user_template = prompt_cfg.get("user", "{question}")
    messages.append({"role": "user", "content": user_template.format(question=question)})

    return messages


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_parametric_dataset(hf_repo: str, limit: int | None = None) -> list[dict]:
    """Load parametric/arithmetic evaluation dataset from HuggingFace.

    Returns:
        List of dicts with keys: index, question, answer, source
    """
    from datasets import load_dataset

    ds = load_dataset(hf_repo, split="train")
    records = [dict(r) for r in ds]

    if limit is not None and limit < len(records):
        records = records[:limit]

    console.print(f"Loaded {len(records)} samples from {hf_repo}")
    source_counts = defaultdict(int)
    for r in records:
        source_counts[r["source"]] += 1
    for source, count in sorted(source_counts.items()):
        console.print(f"  {source}: {count}")

    return records


# ---------------------------------------------------------------------------
# Calibration for short prompts
# ---------------------------------------------------------------------------


def calibrate_mean_activations_from_prompts(
    model,
    tokenizer,
    prompts: list[str],
    num_calibration: int = 50,
    seed: int = 42,
) -> dict[int, torch.Tensor]:
    """Compute per-layer, per-head mean q_proj activations from short text prompts.

    Simpler variant of nolima_ablation.calibrate_mean_activations() that works
    with raw text prompts instead of NoLiMa trials (no needle insertion).

    Args:
        model: HuggingFace model.
        tokenizer: Tokenizer instance.
        prompts: List of text prompts to calibrate from.
        num_calibration: Number of prompts to use.
        seed: Random seed for sampling.

    Returns:
        Dict mapping layer_idx to mean activation tensor (num_heads, head_dim).
    """
    import random as _random

    from locos.utils.model_utils import get_input_device

    num_heads, head_dim = _get_model_head_config(model)
    decoder_layers = get_decoder_layers(model)
    num_layers = len(decoder_layers)

    # Running accumulators
    running_sum: dict[int, torch.Tensor] = {
        i: torch.zeros(num_heads, head_dim, dtype=torch.float64) for i in range(num_layers)
    }
    running_count: dict[int, int] = {i: 0 for i in range(num_layers)}

    # Capture hooks
    handles = []
    for layer_idx, layer in enumerate(decoder_layers):

        def make_capture_hook(lidx):
            def hook(module, args, output):
                out = output.detach().view(output.shape[0], output.shape[1], num_heads, head_dim)
                running_sum[lidx] += out.float().mean(dim=(0, 1)).cpu()
                running_count[lidx] += 1
                return None

            return hook

        handle = layer.self_attn.q_proj.register_forward_hook(make_capture_hook(layer_idx))
        handles.append(handle)

    # Sample calibration prompts
    rng = _random.Random(seed)
    cal_prompts = list(prompts)
    rng.shuffle(cal_prompts)
    cal_prompts = cal_prompts[:num_calibration]

    device = get_input_device(model)

    console.print(f"Calibrating mean activations from {len(cal_prompts)} prompts...")
    with torch.inference_mode():
        for prompt_text in cal_prompts:
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
            model(input_ids=input_ids, output_attentions=False, return_dict=True)
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
        f"[green]Calibrated mean q_proj activations ({len(cal_prompts)} prompts, {num_layers} layers)[/green]"
    )
    return mean_activations


# ---------------------------------------------------------------------------
# Generation and evaluation
# ---------------------------------------------------------------------------


def generate_for_sample(
    model,
    tokenizer,
    sample: dict,
    max_tokens: int,
    prompt_cfg: dict,
    prefill_attn_impl: str = "sdpa",
    stop_token_ids: set[int] | None = None,
) -> str:
    """Generate text for a single parametric/arithmetic question.

    Same generation pattern as nolima_ablation.generate_for_trial() but
    simpler: no needle insertion, just a few-shot prompted question.
    """
    from locos.utils.model_utils import get_input_device, set_model_attn_impl

    messages = build_chat_messages(sample["question"], prompt_cfg)

    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if "enable_thinking" in tokenizer.chat_template:
            chat_kwargs["enable_thinking"] = False
        prompt_text = tokenizer.apply_chat_template(messages, **chat_kwargs)
    else:
        # Fallback for models without chat template: concatenate messages
        parts = []
        for msg in messages:
            if msg["role"] == "system":
                parts.append(msg["content"])
            elif msg["role"] == "user":
                parts.append(f"Q: {msg['content']}")
            elif msg["role"] == "assistant":
                parts.append(f"A: {msg['content']}")
        parts.append("A:")
        prompt_text = "\n".join(parts)

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

    # Switch back to eager for decode
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


def score_sample(output: str, target: str) -> bool:
    """Score a single sample with exact match (case-insensitive, stripped)."""
    return output.strip().lower() == target.strip().lower()


def run_parametric_eval(
    model,
    tokenizer,
    samples: list[dict],
    max_tokens: int,
    prompt_cfg: dict,
    prefill_attn_impl: str,
    label: str = "",
    stop_token_ids: set[int] | None = None,
    debug_trials: int = 0,
) -> list[dict]:
    """Run parametric/arithmetic evaluation and return per-sample results.

    Returns:
        List of dicts with keys: index, source, output, target, correct
    """
    results = []

    desc = f"Evaluating {label}..." if label else "Evaluating..."
    for idx, sample in enumerate(track(samples, description=desc, console=console)):
        output = generate_for_sample(
            model, tokenizer, sample, max_tokens, prompt_cfg, prefill_attn_impl, stop_token_ids
        )
        correct = score_sample(output, sample["answer"])

        results.append(
            {
                "index": sample["index"],
                "source": sample["source"],
                "output": output,
                "target": sample["answer"],
                "correct": correct,
            }
        )

        if debug_trials > 0 and idx < debug_trials:
            correct_color = "green" if correct else "red"
            console.print(
                f"\n[bold]--- Debug sample {idx + 1}/{debug_trials} ---[/bold]\n"
                f"  [dim]source:[/dim]    {sample['source']}\n"
                f"  [dim]question:[/dim]  {sample['question']!r}\n"
                f"  [dim]target:[/dim]    {sample['answer']!r}\n"
                f"  [dim]generated:[/dim] {output!r}\n"
                f"  [dim]correct:[/dim]   [{correct_color}]{correct}[/{correct_color}]"
            )

    return results


def compute_parametric_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from parametric eval results.

    Returns dict with overall and per-source accuracy.
    """
    by_source: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_source[r["source"]].append(r["correct"])

    overall = [r["correct"] for r in results]

    metrics = {
        "accuracy": sum(overall) / len(overall) if overall else 0.0,
        "n_samples": len(results),
    }
    for source, correct_list in sorted(by_source.items()):
        metrics[f"{source}_accuracy"] = sum(correct_list) / len(correct_list) if correct_list else 0.0
        metrics[f"{source}_n"] = len(correct_list)

    return metrics


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------


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
    table = Table(title=f"Parametric/Arithmetic Ablation Results{abl_label}{rand_label}")
    table.add_column("Config", style="bold")
    table.add_column("Heads", justify="right")
    table.add_column("Overall Acc", justify="right")
    table.add_column("City/Country", justify="right")
    table.add_column("PopQA", justify="right")
    table.add_column("Arithmetic", justify="right")
    table.add_column("Samples", justify="right", style="dim")

    baseline_acc = None
    if include_baseline:
        key = run_key("baseline", 0, model_short, limit, ablation_mode, random_heads)
        if key in cache:
            m = cache[key]
            baseline_acc = m["accuracy"]
            table.add_row(
                "Baseline (no ablation)",
                "0",
                f"{m['accuracy']:.4f}",
                f"{m.get('city_country_accuracy', 0):.4f}",
                f"{m.get('popqa_accuracy', 0):.4f}",
                f"{m.get('arithmetic_accuracy', 0):.4f}",
                str(m["n_samples"]),
            )

    mode_label = "random" if random_heads else mode
    for v in sorted(values):
        key = run_key(mode, v, model_short, limit, ablation_mode, random_heads)
        if key in cache:
            m = cache[key]
            v_display = str(int(v)) if mode == "top-k" else str(v)
            acc = m["accuracy"]
            acc_str = f"{acc:.4f}"
            if baseline_acc is not None:
                delta = acc - baseline_acc
                sign = "+" if delta > 0 else ""
                if delta > 0:
                    acc_str += f" [green]({sign}{delta:.4f})[/green]"
                elif delta < 0:
                    acc_str += f" [red]({sign}{delta:.4f})[/red]"
                else:
                    acc_str += f" ({sign}{delta:.4f})"
            table.add_row(
                f"{mode_label}={v_display}",
                str(m.get("n_heads", "?")),
                acc_str,
                f"{m.get('city_country_accuracy', 0):.4f}",
                f"{m.get('popqa_accuracy', 0):.4f}",
                f"{m.get('arithmetic_accuracy', 0):.4f}",
                str(m["n_samples"]),
            )

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Parametric/arithmetic ablation: control experiment for retrieval head specificity",
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
        help="Control condition: select k random heads instead of top-k from a heads JSON",
    )
    parser.add_argument(
        "--heads-list",
        nargs="+",
        type=str,
        default=None,
        help="Explicit list of heads as 'layer-head' strings (e.g., --heads-list 22-15 24-26)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label for this run (used in cache key). Required with --heads-list.",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        type=float,
        default=None,
        help="Values to sweep (k for top-k, thresholds for threshold mode)",
    )
    parser.add_argument("--include-baseline", action="store_true", help="Also run an unmasked baseline")
    parser.add_argument(
        "--ablation-mode",
        type=str,
        default="zero",
        choices=["zero", "mean"],
        help="How to ablate selected heads: 'zero' or 'mean'",
    )
    parser.add_argument(
        "--num-calibration",
        type=int,
        default=50,
        help="Number of samples for calibrating mean activations (only with --ablation-mode mean)",
    )
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate per sample")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dataset samples")
    parser.add_argument(
        "--dataset",
        type=str,
        default="aryopg/parametric-arithmetic-eval",
        help="HuggingFace dataset repo for parametric/arithmetic eval",
    )
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
        help="Print raw generated text, target, and correctness for the first N samples (for diagnosing model output)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("ablation_parametric_results"),
        help="Directory for caching results",
    )
    args = parser.parse_args()

    # --- Validate argument combinations ---
    if args.heads_list:
        assert args.label is not None, "--label is required with --heads-list"
        explicit_heads = []
        for h in args.heads_list:
            parts = h.split("-")
            assert len(parts) == 2, f"Invalid head format '{h}', expected 'layer-head'"
            explicit_heads.append((int(parts[0]), int(parts[1])))
        assert len(explicit_heads) > 0, "Empty heads list"
    elif not args.random_heads:
        assert (
            args.heads is not None and args.heads.exists()
        ), f"Heads file required (and must exist) unless --random-heads or --heads-list. Got: {args.heads}"
    if not args.heads_list:
        assert args.values is not None and len(args.values) > 0, "--values is required unless --heads-list"

    model_short = args.model.split("/")[-1]

    # Load prompt config
    prompt_cfg = load_prompt_config()

    if args.heads_list:
        heads_label = f"{model_short}_{args.label}"
        all_scored = None
    elif args.random_heads:
        heads_label = f"{model_short}_random_seed{args.seed}"
        all_scored = None
    else:
        heads_label = args.heads.stem
        all_scored = load_all_head_scores(args.heads)

    # Separate cache file per selection mode to avoid overwriting on HF upload
    if args.mode == "bottom-k":
        heads_label = f"{heads_label}_bottomk"

    cache_path = args.cache_dir / f"parametric_ablation_{heads_label}.json"
    cache = load_cache(cache_path)

    if args.heads_list:
        heads_display = (
            f"explicit list ({len(explicit_heads)} heads): "
            f"{args.heads_list[:5]}{'...' if len(args.heads_list) > 5 else ''}"
        )
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
            f"[bold]Dataset:[/bold] {args.dataset}\n"
            f"[bold]Include baseline:[/bold] {args.include_baseline}\n"
            f"[bold]Cache:[/bold] {cache_path}",
            title="[green]Parametric/Arithmetic Ablation[/green]",
        )
    )

    # Load dataset
    console.rule("[bold]Loading Dataset[/bold]")
    samples = load_parametric_dataset(args.dataset, limit=args.limit)

    # Determine which runs are needed
    runs_to_do: list[tuple[str, float | None, bool]] = []

    if args.heads_list:
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

    # Load model
    console.rule("[bold]Loading Model[/bold]")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
    )
    stop_token_ids = get_stop_token_ids(tokenizer, model)
    console.print(f"[dim]Stop token IDs ({len(stop_token_ids)}): {stop_token_ids}[/dim]")
    console.print("[green]Model loaded[/green]")

    # Calibrate mean activations if needed
    mean_activations = None
    if args.ablation_mode == "mean":
        any_masked_runs = any(masked for _, _, masked in runs_to_do)
        if any_masked_runs:
            console.rule("[bold]Calibrating Mean Activations[/bold]")
            # Build prompts for calibration (same format as generation)
            cal_prompts = []
            for s in samples:
                messages = build_chat_messages(s["question"], prompt_cfg)
                if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                    chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
                    if "enable_thinking" in tokenizer.chat_template:
                        chat_kwargs["enable_thinking"] = False
                    prompt_text = tokenizer.apply_chat_template(messages, **chat_kwargs)
                else:
                    parts = []
                    for msg in messages:
                        if msg["role"] == "system":
                            parts.append(msg["content"])
                        elif msg["role"] == "user":
                            parts.append(f"Q: {msg['content']}")
                        elif msg["role"] == "assistant":
                            parts.append(f"A: {msg['content']}")
                    parts.append("A:")
                    prompt_text = "\n".join(parts)
                cal_prompts.append(prompt_text)

            mean_activations = calibrate_mean_activations_from_prompts(
                model,
                tokenizer,
                cal_prompts,
                num_calibration=args.num_calibration,
                seed=args.seed,
            )

    # For random heads, we need model dimensions
    num_layers_model, num_heads_model = extract_model_config(model)

    # Execute runs
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
            results = run_parametric_eval(
                model,
                tokenizer,
                samples,
                max_tokens=args.max_tokens,
                prompt_cfg=prompt_cfg,
                prefill_attn_impl=args.prefill_attn_impl,
                label=label,
                stop_token_ids=stop_token_ids,
                debug_trials=args.debug_trials,
            )
        finally:
            remove_hooks(hooks)

        # Compute metrics
        metrics = compute_parametric_metrics(results)
        metrics.update(
            {
                "n_heads": n_heads,
                "mode": "explicit" if args.heads_list else (args.mode if masked else "baseline"),
                "ablation_mode": args.ablation_mode if masked else "none",
                "random_heads": args.random_heads,
                "heads_list": args.heads_list if args.heads_list and masked else None,
                "value": value if value is not None else 0,
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            }
        )

        # Cache key
        if not masked:
            key = run_key("baseline", 0, model_short, args.limit, args.ablation_mode, args.random_heads)
        elif args.heads_list:
            key = f"{model_short}__{args.label}"
        else:
            key = run_key(args.mode, value, model_short, args.limit, args.ablation_mode, args.random_heads)

        cache[key] = metrics
        save_cache(cache, cache_path)

        # Save per-trial results
        safe_label = label.replace("=", "_").replace(".", "p")
        trial_path = args.cache_dir / f"parametric_ablation_{heads_label}_{safe_label}_trials.jsonl"
        trial_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trial_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        console.print(f"  Overall accuracy: {metrics['accuracy']:.4f}")
        for source_key in sorted(k for k in metrics if k.endswith("_accuracy") and k != "accuracy"):
            source_name = source_key.replace("_accuracy", "")
            console.print(f"    {source_name}: {metrics[source_key]:.4f}")

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


if __name__ == "__main__":
    main()
