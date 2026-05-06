#!/usr/bin/env python3
"""Plot NoLiMa ablation comparison: multiple head sources on one figure.

Reads cached results from ``run_nolima_ablation.py`` across multiple head
sources (contrastive, Wu NoLiMa, Wu NIAH, random, cluster-based) and produces
a single line plot comparing their ablation curves.

Usage:
    # Compare all methods (mean ablation)
    python locos/plot_ablation_comparison.py \
        --caches \
            ablation_results/nolima_ablation_Meta-Llama-3-8B-Instruct_contrastive_nolima_topk10_pooled.json \
            ablation_results/nolima_ablation_Meta-Llama-3-8B-Instruct_nolima.json \
            ablation_results/nolima_ablation_Meta-Llama-3-8B-Instruct.json \
            ablation_results/nolima_ablation_Meta-Llama-3-8B-Instruct_random_seed42.json \
        --labels "Contrastive pooled" "Wu NoLiMa" "Wu NIAH" "Random" \
        --ablation-mode mean \
        --out figures/nolima_ablation_mean.svg

    # Include cluster-based point
    python locos/plot_ablation_comparison.py \
        --caches ... \
        --labels ... \
        --cluster-caches \
            ablation_results/nolima_ablation_Meta-Llama-3-8B-Instruct_core_cluster_14h.json \
        --cluster-labels "Core 14h" \
        --ablation-mode mean

    # Average across random/control or calibration seeds (comma-separated paths).
    # Curves shade bootstrap CIs from *_trials.jsonl and show seed std error bars.
    python locos/plotting/ablation_comparison.py \
        --caches \
            ablation_results/nolima_ablation_Meta-Llama-3-8B-Instruct_nolima.json \
            ablation_results/nolima_ablation_Meta-Llama-3-8B-Instruct_random_seed42.json,ablation_results/nolima_ablation_Meta-Llama-3-8B-Instruct_random_seed43.json,ablation_results/nolima_ablation_Meta-Llama-3-8B-Instruct_random_seed44.json \
        --labels "Wu NoLiMa" "Random (3 seeds)" \
        --ablation-mode mean

Requires: matplotlib, numpy, rich
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

from locos_eval.utils.plotting import MODEL_PRETTY_NAMES, save_figure, setup_plot_style

console = Console()

COLOR_CODES = ["#0563BA", "#BE1E2D", "#DB808D", "#E8972E", "#41A9AC"]
COLORS = COLOR_CODES
MARKERS = ["o", "s", "D", "^", "v", "P"]

METRIC_LABELS = {
    "rouge_l_mean": "ROUGE-L",
    "rouge_1_mean": "ROUGE-1",
    "rouge_1_recall_mean": "ROUGE-1 recall",
    "wu_accuracy": "Accuracy (R-1 recall > 0.5)",
}
# Metrics that are bounded to [0, 1] and should plot the full range.
ACCURACY_LIKE_METRICS = {"wu_accuracy", "rouge_1_recall_mean"}

TRIAL_METRIC_KEYS = {
    "rouge_l_mean": "rouge_l",
    "rouge_1_mean": "rouge_1",
    "rouge_1_recall_mean": "rouge_1_recall",
    "wu_accuracy": "rouge_1_recall",
}


@dataclass(frozen=True)
class SweepStats:
    """Aggregated sweep values plus uncertainty for one plotted method."""

    ks: list[int]
    means: list[float]
    ci_lows: list[float]
    ci_highs: list[float]
    seed_stds: list[float]
    baseline_mean: float | None
    baseline_ci_low: float | None
    baseline_ci_high: float | None
    baseline_seed_std: float
    n_seeds: int
    uncertainty_source: str


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def load_cache(path: Path) -> dict:
    """Load ablation results cache."""
    assert path.exists(), f"Cache file not found: {path}"
    with open(path) as f:
        return json.load(f)


def _safe_label(label: str) -> str:
    return label.replace("=", "_").replace(".", "p")


def _infer_dataset_and_heads_label(cache_path: Path, metrics: dict) -> tuple[str, str]:
    dataset = metrics.get("dataset")
    stem = cache_path.stem
    if dataset is None:
        dataset = stem.split("_ablation_", maxsplit=1)[0] if "_ablation_" in stem else "nolima"

    prefix = f"{dataset}_ablation_"
    heads_label = stem.removeprefix(prefix)
    return dataset, heads_label


def _trial_label(metrics: dict) -> str:
    mode = metrics.get("mode", "")
    if mode in ("baseline", "greedy"):
        return "baseline"

    value = metrics.get("value", 0)
    if metrics.get("random_heads", False):
        mode_label = "random"
    else:
        mode_label = mode
    return f"{mode_label}={float(value)}"


def infer_trial_path(cache_path: Path, metrics: dict, trial_dir: Path | None = None) -> Path:
    """Infer the per-trial JSONL sidecar emitted by nolima_ablation.py."""
    dataset, heads_label = _infer_dataset_and_heads_label(cache_path, metrics)
    base_dir = trial_dir if trial_dir is not None else cache_path.parent
    return base_dir / f"{dataset}_ablation_{heads_label}_{_safe_label(_trial_label(metrics))}_trials.jsonl"


def _trial_metric_values(trial_path: Path, metric: str) -> list[float] | None:
    if not trial_path.exists():
        return None

    trial_key = TRIAL_METRIC_KEYS[metric]
    values: list[float] = []
    with open(trial_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            value = row[trial_key]
            if metric == "wu_accuracy":
                value = 1.0 if value > 0.5 else 0.0
            values.append(float(value))
    return values


def _trial_values_for_metrics(
    cache_path: Path,
    metrics: dict,
    metric: str,
    trial_dir: Path | None = None,
) -> list[float] | None:
    inferred = infer_trial_path(cache_path, metrics, trial_dir)
    values = _trial_metric_values(inferred, metric)
    if values is not None:
        return values

    dataset, _ = _infer_dataset_and_heads_label(cache_path, metrics)
    base_dir = trial_dir if trial_dir is not None else cache_path.parent
    pattern = f"{dataset}_ablation_*_{_safe_label(_trial_label(metrics))}_trials.jsonl"
    target_mean = metrics.get(metric)
    target_n = metrics.get("n_samples")

    for candidate in sorted(base_dir.glob(pattern)):
        candidate_values = _trial_metric_values(candidate, metric)
        if not candidate_values:
            continue
        if target_n is not None and len(candidate_values) != int(target_n):
            continue
        if target_mean is not None and not np.isclose(float(np.mean(candidate_values)), float(target_mean), atol=1e-12):
            continue
        return candidate_values

    return None


def _bootstrap_ci(values: list[float], n_bootstrap: int, seed: int) -> tuple[float, float] | None:
    if n_bootstrap <= 0 or not values:
        return None

    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))
    boot_means = arr[indices].mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


def extract_sweep_data(
    cache: dict,
    ablation_mode: str = "mean",
    k_filter: set[int] | None = None,
) -> tuple[list[int], list[float], float | None]:
    """Extract (k_values, rouge_l_scores, baseline) from a single sweep cache.

    Filters to entries matching the given ablation_mode for masked runs.
    Baseline is ablation-mode agnostic (no hooks installed).

    Args:
        cache: Loaded cache dict {run_key: metrics_dict}.
        ablation_mode: "zero" or "mean" — filter masked runs by this mode.
        k_filter: If provided, only include these k values.

    Returns:
        (k_values, rouge_l_values, baseline_rouge_l_or_None)
    """
    baseline = None
    points: dict[int, float] = {}

    for _, metrics in cache.items():
        mode = metrics.get("mode", "")

        if mode in ("baseline", "greedy"):
            baseline = metrics["rouge_l_mean"]
            continue

        # Filter by ablation mode
        entry_abl = metrics.get("ablation_mode", "zero")
        if entry_abl != ablation_mode:
            continue

        k = int(metrics.get("value", 0))
        if k > 0:
            if k_filter is not None and k not in k_filter:
                continue
            points[k] = metrics["rouge_l_mean"]

    ks = sorted(points.keys())
    scores = [points[k] for k in ks]
    return ks, scores, baseline


def extract_sweep_data_aggregated(
    caches: list[dict],
    ablation_mode: str = "mean",
    k_filter: set[int] | None = None,
    metric: str = "rouge_l_mean",
) -> tuple[list[int], list[float], list[float], float | None, float | None, int]:
    """Aggregate ``metric`` across multiple cache files (e.g. random seeds).

    Returns:
        (ks, means, sems, baseline_mean, baseline_sem, n_seeds)
        SEM uses ddof=1. Single-cache groups get ``sems=[0.0,...]`` and
        ``baseline_sem=0.0``. Only k values present in at least one cache are
        returned; per-k aggregation skips caches missing that k.
    """
    per_k: dict[int, list[float]] = defaultdict(list)
    baselines: list[float] = []

    for cache in caches:
        for _, metrics in cache.items():
            mode = metrics.get("mode", "")
            if mode in ("baseline", "greedy"):
                if metric in metrics:
                    baselines.append(metrics[metric])
                continue
            if metrics.get("ablation_mode", "zero") != ablation_mode:
                continue
            if metric not in metrics:
                continue
            k = int(metrics.get("value", 0))
            if k <= 0:
                continue
            if k_filter is not None and k not in k_filter:
                continue
            per_k[k].append(metrics[metric])

    def _sem(xs: list[float]) -> float:
        if len(xs) <= 1:
            return 0.0
        return float(np.std(xs, ddof=1) / np.sqrt(len(xs)))

    ks = sorted(per_k.keys())
    means = [float(np.mean(per_k[k])) for k in ks]
    sems = [_sem(per_k[k]) for k in ks]

    baseline_mean = float(np.mean(baselines)) if baselines else None
    baseline_sem = _sem(baselines) if baselines else 0.0
    return ks, means, sems, baseline_mean, baseline_sem, len(caches)


def extract_sweep_stats(
    caches: list[dict],
    cache_paths: list[Path],
    ablation_mode: str = "mean",
    k_filter: set[int] | None = None,
    metric: str = "rouge_l_mean",
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
    trial_dir: Path | None = None,
) -> SweepStats:
    """Aggregate one plotted cache group with bootstrap CIs and seed std.

    Bootstrap CIs are computed from the ``*_trials.jsonl`` sidecars emitted by
    ``nolima_ablation.py``. When multiple cache files are supplied for a method
    (calibration or random-control seeds), the plotted mean is the mean across
    cache-level point estimates and ``seed_stds`` records the sample standard
    deviation across those estimates.
    """
    assert len(caches) == len(cache_paths), "caches and cache_paths must align"
    assert metric in TRIAL_METRIC_KEYS, f"Cannot bootstrap unknown metric {metric!r}"

    per_k_means: dict[int, list[float]] = defaultdict(list)
    per_k_ci_offsets: dict[int, list[tuple[float, float]]] = defaultdict(list)
    baselines: list[float] = []
    baseline_ci_offsets: list[tuple[float, float]] = []

    for cache_idx, (cache, cache_path) in enumerate(zip(caches, cache_paths)):
        for _, metrics in cache.items():
            mode = metrics.get("mode", "")
            is_baseline = mode in ("baseline", "greedy")

            if is_baseline:
                if metric not in metrics:
                    continue
                mean = float(metrics[metric])
                baselines.append(mean)
                values = _trial_values_for_metrics(cache_path, metrics, metric, trial_dir)
                ci = _bootstrap_ci(values or [], bootstrap_samples, bootstrap_seed + cache_idx)
                if ci is not None:
                    baseline_ci_offsets.append((ci[0] - mean, ci[1] - mean))
                continue

            if metrics.get("ablation_mode", "zero") != ablation_mode:
                continue
            if metric not in metrics:
                continue
            k = int(metrics.get("value", 0))
            if k <= 0:
                continue
            if k_filter is not None and k not in k_filter:
                continue

            mean = float(metrics[metric])
            per_k_means[k].append(mean)
            values = _trial_values_for_metrics(cache_path, metrics, metric, trial_dir)
            ci = _bootstrap_ci(values or [], bootstrap_samples, bootstrap_seed + cache_idx + k * 1009)
            if ci is not None:
                per_k_ci_offsets[k].append((ci[0] - mean, ci[1] - mean))

    def _std(xs: list[float]) -> float:
        if len(xs) <= 1:
            return 0.0
        return float(np.std(xs, ddof=1))

    def _bounds(mean: float, means: list[float], offsets: list[tuple[float, float]]) -> tuple[float, float]:
        if offsets:
            low_offset = float(np.mean([lo for lo, _ in offsets]))
            high_offset = float(np.mean([hi for _, hi in offsets]))
            return mean + low_offset, mean + high_offset
        seed_std = _std(means)
        return mean - seed_std, mean + seed_std

    ks = sorted(per_k_means)
    means = [float(np.mean(per_k_means[k])) for k in ks]
    seed_stds = [_std(per_k_means[k]) for k in ks]
    bounds = [_bounds(mean, per_k_means[k], per_k_ci_offsets[k]) for k, mean in zip(ks, means)]
    ci_lows = [low for low, _ in bounds]
    ci_highs = [high for _, high in bounds]

    baseline_mean = float(np.mean(baselines)) if baselines else None
    baseline_seed_std = _std(baselines)
    if baseline_mean is not None:
        baseline_ci_low, baseline_ci_high = _bounds(baseline_mean, baselines, baseline_ci_offsets)
    else:
        baseline_ci_low = baseline_ci_high = None

    has_bootstrap = bool(baseline_ci_offsets) or any(per_k_ci_offsets.values())
    has_seed_std = len(caches) > 1
    if has_bootstrap and has_seed_std:
        uncertainty_source = "bootstrap+seed_std"
    elif has_bootstrap:
        uncertainty_source = "bootstrap"
    elif has_seed_std:
        uncertainty_source = "seed_std"
    else:
        uncertainty_source = "none"

    return SweepStats(
        ks=ks,
        means=means,
        ci_lows=ci_lows,
        ci_highs=ci_highs,
        seed_stds=seed_stds,
        baseline_mean=baseline_mean,
        baseline_ci_low=baseline_ci_low,
        baseline_ci_high=baseline_ci_high,
        baseline_seed_std=baseline_seed_std,
        n_seeds=len(caches),
        uncertainty_source=uncertainty_source,
    )


def extract_cluster_data(
    cache: dict,
    ablation_mode: str = "mean",
    metric: str = "rouge_l_mean",
) -> tuple[int, float] | None:
    """Extract (n_heads, score) from a cluster (--heads-list) cache.

    Returns None if no matching entry is found.
    """
    for _, metrics in cache.items():
        mode = metrics.get("mode", "")
        if mode in ("baseline", "greedy"):
            continue

        entry_abl = metrics.get("ablation_mode", "zero")
        if entry_abl != ablation_mode:
            continue

        n_heads = metrics.get("n_heads", 0)
        if n_heads > 0:
            assert metric in metrics, f"Cluster cache missing metric {metric!r}; available: {sorted(metrics)}"
            return n_heads, metrics[metric]

    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_comparison_plot(
    sweep_data: dict[str, SweepStats],
    cluster_data: dict[str, tuple[int, float]] | None,
    ablation_mode: str,
    out_path: Path,
    metric: str = "rouge_l_mean",
    model_name: str | None = None,
    color_indices: list[int] | None = None,
    marker_indices: list[int] | None = None,
) -> None:
    """Create ablation comparison plot.

    Args:
        sweep_data: Per-method sweep statistics with bootstrap CIs and seed std.
        cluster_data: {label: (n_heads, score)} for cluster ablations.
        ablation_mode: "zero" or "mean" (for axis annotation).
        out_path: Output SVG path.
        metric: Cache field used for the y-axis (e.g. ``rouge_l_mean``,
            ``wu_accuracy`` for Wu et al.'s NIAH gating: R-1 recall > 0.5).
        model_name: Raw model name (looked up in MODEL_PRETTY_NAMES for title).
        color_indices: Per-method indices into COLORS palette.
        marker_indices: Per-method indices into MARKERS list.
    """
    setup_plot_style()
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"

    fig, ax = plt.subplots(figsize=(3.5, 3))

    # Plot baseline as horizontal dashed line (use first available)
    baseline_val = None
    baseline_ci_low = None
    baseline_ci_high = None
    for stats in sweep_data.values():
        if stats.baseline_mean is not None:
            baseline_val = stats.baseline_mean
            baseline_ci_low = stats.baseline_ci_low
            baseline_ci_high = stats.baseline_ci_high
            break
    if baseline_val is not None:
        ax.axhline(
            baseline_val,
            color="grey",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            zorder=1,
            label="Baseline (no ablation)",
        )
        if baseline_ci_low is not None and baseline_ci_high is not None and baseline_ci_high > baseline_ci_low:
            ax.axhspan(
                baseline_ci_low,
                baseline_ci_high,
                color="grey",
                alpha=0.12,
                zorder=0,
            )

    # Plot sweep curves
    for i, (label, stats) in enumerate(sweep_data.items()):
        if not stats.ks:
            continue
        ci = color_indices[i] if color_indices else i
        mi = marker_indices[i] if marker_indices else i
        color = COLORS[ci % len(COLORS)]
        marker = MARKERS[mi % len(MARKERS)]

        display_label = label if stats.n_seeds <= 1 else f"{label} (n={stats.n_seeds})"
        ax.plot(
            stats.ks,
            stats.means,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2,
            zorder=3,
            label=display_label,
        )
        if any(high > low for low, high in zip(stats.ci_lows, stats.ci_highs)):
            ax.fill_between(
                stats.ks,
                stats.ci_lows,
                stats.ci_highs,
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
        if any(std > 0 for std in stats.seed_stds):
            ax.errorbar(
                stats.ks,
                stats.means,
                yerr=stats.seed_stds,
                color=color,
                fmt="none",
                capsize=2.5,
                linewidth=1.0,
                alpha=0.65,
                zorder=4,
            )

    # Plot cluster points as standalone markers
    n_sweep = len(sweep_data)
    if cluster_data:
        for j, (label, (n_heads, score)) in enumerate(cluster_data.items()):
            ci = (n_sweep + j) % len(COLORS)
            mi = (n_sweep + j) % len(MARKERS)
            color = COLORS[ci]
            marker = MARKERS[mi]
            ax.plot(
                n_heads,
                score,
                color=color,
                marker=marker,
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1.5,
                linestyle="none",
                zorder=4,
                label=label,
            )

    ax.set_xlabel(r"Heads ablated ($k$)")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))

    # Integer x-ticks from the union of all k values
    all_ks = sorted({k for stats in sweep_data.values() for k in stats.ks})
    if cluster_data:
        all_ks = sorted(set(all_ks) | {n for n, _ in cluster_data.values()})
    ax.set_xticks(all_ks)
    ax.set_xticklabels([str(k) for k in all_ks])

    # Accuracy-like metrics span [0, 1]; ROUGE-L on NoLiMa baselines around 0.35.
    default_top = 1.02 if metric in ACCURACY_LIKE_METRICS else 0.40
    default_baseline_fallback = 0.95 if metric in ACCURACY_LIKE_METRICS else 0.35
    ax.set_ylim(bottom=-0.02, top=max(default_top, (baseline_val or default_baseline_fallback) + 0.03))

    # Horizontal legend — ncol = number of entries so it's one row.
    # save_figure will extract this and save it as a separate file.
    n_entries = len(sweep_data) + (len(cluster_data) if cluster_data else 0) + (1 if baseline_val is not None else 0)
    ax.legend(loc="lower left", fontsize=8, ncol=n_entries)
    # ax.legend(loc="lower left", fontsize=8, ncol=n_entries, frameon=False)

    # Model name as title (if provided)
    keep_title = False
    if model_name:
        pretty = MODEL_PRETTY_NAMES.get(model_name, model_name)
        ax.set_title(pretty)
        keep_title = True

    fig.tight_layout()
    save_figure(fig, out_path, keep_title=keep_title)
    console.print(f"[green]Saved:[/green] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot NoLiMa ablation comparison across head sources",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--caches",
        nargs="+",
        type=str,
        required=True,
        help=(
            "Cache JSON files from nolima_ablation.py (one group per method). "
            "A group is either a single path or a comma-separated list of paths "
            "(e.g. 'a.json,b.json,c.json') — comma-separated paths are averaged "
            "with seed-standard-deviation error bars. Useful for calibration or "
            "random-control seed sweeps."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        required=True,
        help="Display labels for each cache file (same order as --caches)",
    )
    parser.add_argument(
        "--cluster-caches",
        nargs="*",
        type=Path,
        default=None,
        help="Cache files for cluster-based ablations (plotted as single points)",
    )
    parser.add_argument(
        "--cluster-labels",
        nargs="*",
        type=str,
        default=None,
        help="Display labels for cluster caches",
    )
    parser.add_argument(
        "--ablation-mode",
        type=str,
        default="mean",
        choices=["zero", "mean"],
        help="Filter to this ablation mode",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="rouge_l_mean",
        choices=sorted(METRIC_LABELS),
        help=(
            "Y-axis metric. 'rouge_l_mean' is NoLiMa's default; 'wu_accuracy' "
            "is Wu et al.'s NIAH gating (per-example ROUGE-1 recall > 0.5)."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples over per-trial sidecar scores. Set 0 to disable.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--trial-dir",
        type=Path,
        default=None,
        help="Directory containing *_trials.jsonl sidecars. Defaults to each cache file's directory.",
    )
    parser.add_argument(
        "--k-values",
        nargs="*",
        type=int,
        default=None,
        help="Only include these k values (e.g. --k-values 1 5 10 20 50 100)",
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        type=int,
        default=None,
        help="Per-method color indices into the palette (e.g. --colors 0 1 0 1)",
    )
    parser.add_argument(
        "--markers",
        nargs="+",
        type=int,
        default=None,
        help="Per-method marker indices into MARKERS list (e.g. --markers 0 0 1 1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for figure title (looked up in MODEL_PRETTY_NAMES)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output SVG path (default: figures/nolima_ablation_{ablation_mode}_{metric}.svg)",
    )
    args = parser.parse_args()

    assert len(args.caches) == len(
        args.labels
    ), f"Number of cache groups ({len(args.caches)}) must match labels ({len(args.labels)})"

    # Parse cache groups: each --caches entry may be comma-separated for seed averaging
    cache_groups: list[list[Path]] = []
    for entry in args.caches:
        paths = [Path(p) for p in entry.split(",") if p]
        assert paths, f"Empty cache group: {entry!r}"
        for p in paths:
            assert p.exists(), f"Cache file not found: {p}"
        cache_groups.append(paths)

    # Load + aggregate sweep data
    k_filter = set(args.k_values) if args.k_values else None
    sweep_data: dict[str, SweepStats] = {}
    for paths, label in zip(cache_groups, args.labels):
        caches = [load_cache(p) for p in paths]
        stats = extract_sweep_stats(
            caches=caches,
            cache_paths=paths,
            ablation_mode=args.ablation_mode,
            k_filter=k_filter,
            metric=args.metric,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            trial_dir=args.trial_dir,
        )
        if not stats.ks:
            console.print(f"[yellow]Warning: no {args.ablation_mode} runs in group {label!r}[/yellow]")
        if stats.n_seeds > 1:
            console.print(
                f"[dim]Aggregated {label!r} across {stats.n_seeds} caches " f"(seed std shown as error bars)[/dim]"
            )
        if stats.uncertainty_source == "none" and args.bootstrap_samples > 0:
            console.print(
                f"[yellow]Warning: no per-trial sidecars found for {label!r}; "
                "uncertainty band is unavailable[/yellow]"
            )
        elif "bootstrap" in stats.uncertainty_source:
            console.print(
                f"[dim]Computed {args.bootstrap_samples}x bootstrap CIs for {label!r} " "from per-trial sidecars[/dim]"
            )
        sweep_data[label] = stats

    # Load cluster data
    cluster_data: dict[str, tuple[int, float]] | None = None
    if args.cluster_caches:
        assert args.cluster_labels and len(args.cluster_caches) == len(
            args.cluster_labels
        ), "Must provide --cluster-labels matching --cluster-caches"
        cluster_data = {}
        for path, label in zip(args.cluster_caches, args.cluster_labels):
            cache = load_cache(path)
            result = extract_cluster_data(cache, args.ablation_mode, metric=args.metric)
            if result is None:
                console.print(f"[yellow]Warning: no {args.ablation_mode} cluster run in {path}[/yellow]")
            else:
                cluster_data[label] = result

    # Output path
    if args.out:
        out_path = args.out
    else:
        out_path = Path(f"figures/nolima_ablation_{args.ablation_mode}_{args.metric}.svg")

    make_comparison_plot(
        sweep_data,
        cluster_data,
        args.ablation_mode,
        out_path,
        metric=args.metric,
        model_name=args.model,
        color_indices=args.colors,
        marker_indices=args.markers,
    )

    # Print summary table
    metric_label = METRIC_LABELS.get(args.metric, args.metric)
    console.rule(f"[bold]Ablation Comparison ({args.ablation_mode}, {metric_label})[/bold]")
    from rich.table import Table

    table = Table()
    table.add_column("$k$", justify="right", style="bold")
    for label in sweep_data:
        table.add_column(label, justify="right")
    if cluster_data:
        for label in cluster_data:
            table.add_column(label, justify="right")

    all_ks = sorted({k for stats in sweep_data.values() for k in stats.ks})
    baseline_val = next((stats.baseline_mean for stats in sweep_data.values() if stats.baseline_mean is not None), None)

    if baseline_val is not None:
        row = ["0"] + [f"{baseline_val:.3f}"] * len(sweep_data)
        if cluster_data:
            row += ["—"] * len(cluster_data)
        table.add_row(*row)

    for k in all_ks:
        row = [str(k)]
        for _, stats in sweep_data.items():
            if k in stats.ks:
                idx = stats.ks.index(k)
                if stats.seed_stds[idx] > 0:
                    row.append(f"{stats.means[idx]:.3f}±{stats.seed_stds[idx]:.3f}")
                elif stats.ci_highs[idx] > stats.ci_lows[idx]:
                    row.append(f"{stats.means[idx]:.3f} [{stats.ci_lows[idx]:.3f}, {stats.ci_highs[idx]:.3f}]")
                else:
                    row.append(f"{stats.means[idx]:.3f}")
            else:
                row.append("—")
        if cluster_data:
            for _, (n, s) in cluster_data.items():
                row.append(f"{s:.3f}" if k == n else "—")
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    main()
