#!/usr/bin/env python3
"""Compare NIAH vs NoLiMa ablation curves for LOCOS-detected heads.

One figure per model, saved to --out-dir. Each figure shows two curves for
the same LOCOS (logit-contribution) head set, plus styled gray baselines:
  - Solid line / baseline:     ablation evaluated on NoLiMa (in-distribution)
  - Dash-dot line / baseline:  ablation evaluated on NIAH (transfer)

NoLiMa baseline is read from the LOCOS cache itself; NIAH baseline from
niah_ablation_{model}_random_seed42.json.

Usage:
    python locos/plotting/locos_niah_vs_nolima.py \
        --models Qwen3-8B Qwen3-14B Qwen3-32B \
                 gemma-3-12b-it gemma-3-27b-it Olmo-3.1-32B-Instruct \
        --results-dir /path/to/locos-results/ablation_results \
        --ablation-mode mean \
        --out-dir figures/locos_niah_vs_nolima

Requires: matplotlib, numpy, seaborn, rich
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

from locos_eval.utils.plotting import MODEL_PRETTY_NAMES, save_figure, setup_plot_style

console = Console()

COLOR_CODES = ["#0563BA", "#BE1E2D", "#DB808D", "#E8972E", "#41A9AC"]
COLORS = COLOR_CODES
LINE_COLOR_IDX = 0  # same color for both datasets

NOLIMA_LINESTYLE = "solid"
NIAH_LINESTYLE = (0, (3, 1, 1, 1))  # dash-dot

K_MAX = 50

METRIC_LABELS = {
    "rouge_l_mean": "ROUGE-L",
    "rouge_1_mean": "ROUGE-1",
}

TRIAL_METRIC_KEYS = {
    "rouge_l_mean": "rouge_l",
    "rouge_1_mean": "rouge_1",
}


@dataclass(frozen=True)
class SweepStats:
    """Ablation curve point estimates plus optional bootstrap intervals."""

    ks: list[int]
    scores: list[float]
    ci_lows: list[float]
    ci_highs: list[float]
    baseline: float | None
    baseline_ci_low: float | None
    baseline_ci_high: float | None


def load_cache(path: Path) -> dict:
    assert path.exists(), f"Cache not found: {path}"
    with open(path) as f:
        return json.load(f)


def _safe_label(label: str) -> str:
    return label.replace("=", "_").replace(".", "p")


def _trial_label(entry: dict) -> str:
    mode = entry.get("mode", "")
    if mode in ("baseline", "greedy"):
        return "baseline"
    value = float(entry.get("value", 0))
    mode_label = "random" if entry.get("random_heads", False) else mode
    return f"{mode_label}={value}"


def _infer_dataset_and_heads_label(cache_path: Path, entry: dict) -> tuple[str, str]:
    dataset = entry.get("dataset")
    stem = cache_path.stem
    if dataset is None:
        dataset = stem.split("_ablation_", maxsplit=1)[0] if "_ablation_" in stem else "nolima"
    heads_label = stem.removeprefix(f"{dataset}_ablation_")
    return dataset, heads_label


def infer_trial_path(cache_path: Path, entry: dict, trial_dir: Path | None = None) -> Path:
    """Infer the per-trial JSONL sidecar emitted by nolima_ablation.py."""
    dataset, heads_label = _infer_dataset_and_heads_label(cache_path, entry)
    base_dir = trial_dir if trial_dir is not None else cache_path.parent
    return base_dir / f"{dataset}_ablation_{heads_label}_{_safe_label(_trial_label(entry))}_trials.jsonl"


def _read_trial_values(path: Path, metric: str) -> list[float] | None:
    if not path.exists():
        return None
    values = []
    trial_key = TRIAL_METRIC_KEYS[metric]
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                values.append(float(json.loads(line)[trial_key]))
    return values


def _matches_entry(values: list[float], entry: dict, metric: str) -> bool:
    if "n_samples" in entry and len(values) != int(entry["n_samples"]):
        return False
    return metric not in entry or abs(float(sum(values) / len(values)) - float(entry[metric])) <= 1e-12


def trial_values_for_entry(
    cache_path: Path,
    entry: dict,
    metric: str,
    trial_dir: Path | None = None,
) -> list[float] | None:
    direct_path = infer_trial_path(cache_path, entry, trial_dir)
    values = _read_trial_values(direct_path, metric)
    if values is not None:
        return values

    dataset, _ = _infer_dataset_and_heads_label(cache_path, entry)
    base_dir = trial_dir if trial_dir is not None else cache_path.parent
    pattern = f"{dataset}_ablation_*_{_safe_label(_trial_label(entry))}_trials.jsonl"
    for candidate in sorted(base_dir.glob(pattern)):
        values = _read_trial_values(candidate, metric)
        if values and _matches_entry(values, entry, metric):
            return values
    return None


def _bootstrap_ci(values: list[float], bootstrap_samples: int, seed: int) -> tuple[float, float] | None:
    if bootstrap_samples <= 0 or not values:
        return None
    import numpy as np

    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(bootstrap_samples, len(arr)))
    boot_means = arr[indices].mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


def extract_sweep(
    cache: dict,
    ablation_mode: str,
    metric: str,
    k_max: int = K_MAX,
) -> tuple[list[int], list[float], float | None]:
    """Return (sorted k_values, scores, baseline) from a cache dict."""
    baseline: float | None = None
    points: dict[int, float] = {}

    for _, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        mode = entry.get("mode", "")
        if mode in ("baseline", "greedy"):
            if metric in entry:
                baseline = entry[metric]
            continue
        if entry.get("ablation_mode", "zero") != ablation_mode:
            continue
        if metric not in entry:
            continue
        k = int(entry.get("value", 0))
        if 0 < k <= k_max:
            points[k] = entry[metric]

    ks = sorted(points)
    return ks, [points[k] for k in ks], baseline


def extract_sweep_stats(
    cache: dict,
    cache_path: Path,
    ablation_mode: str,
    metric: str,
    k_max: int = K_MAX,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
    trial_dir: Path | None = None,
) -> SweepStats:
    """Return sweep values with bootstrap CIs from per-trial sidecars."""
    baseline: float | None = None
    baseline_ci_low: float | None = None
    baseline_ci_high: float | None = None
    points: dict[int, float] = {}
    ci_bounds: dict[int, tuple[float, float]] = {}

    for _, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        mode = entry.get("mode", "")
        if mode in ("baseline", "greedy"):
            if metric in entry:
                baseline = float(entry[metric])
                values = trial_values_for_entry(cache_path, entry, metric, trial_dir)
                ci = _bootstrap_ci(values or [], bootstrap_samples, bootstrap_seed)
                if ci is not None:
                    baseline_ci_low, baseline_ci_high = ci
            continue
        if entry.get("ablation_mode", "zero") != ablation_mode:
            continue
        if metric not in entry:
            continue
        k = int(entry.get("value", 0))
        if 0 < k <= k_max:
            points[k] = float(entry[metric])
            values = trial_values_for_entry(cache_path, entry, metric, trial_dir)
            ci = _bootstrap_ci(values or [], bootstrap_samples, bootstrap_seed + k * 1009)
            if ci is not None:
                ci_bounds[k] = ci

    ks = sorted(points)
    scores = [points[k] for k in ks]
    ci_lows = [ci_bounds.get(k, (points[k], points[k]))[0] for k in ks]
    ci_highs = [ci_bounds.get(k, (points[k], points[k]))[1] for k in ks]
    return SweepStats(
        ks=ks,
        scores=scores,
        ci_lows=ci_lows,
        ci_highs=ci_highs,
        baseline=baseline,
        baseline_ci_low=baseline_ci_low,
        baseline_ci_high=baseline_ci_high,
    )


def make_model_figure(
    model: str,
    nolima_stats: SweepStats,
    niah_stats: SweepStats,
    metric: str,
    out_path: Path,
) -> None:
    setup_plot_style()
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"

    fig, ax = plt.subplots(figsize=(3.75, 3.5))
    color = COLORS[LINE_COLOR_IDX]

    # Baselines — gray, styled by dataset (excluded from auto-legend)
    if nolima_stats.baseline is not None:
        ax.axhline(nolima_stats.baseline, color="grey", linestyle=NOLIMA_LINESTYLE, linewidth=1.2, alpha=0.6, zorder=1)
        if (
            nolima_stats.baseline_ci_low is not None
            and nolima_stats.baseline_ci_high is not None
            and nolima_stats.baseline_ci_high > nolima_stats.baseline_ci_low
        ):
            ax.axhspan(
                nolima_stats.baseline_ci_low,
                nolima_stats.baseline_ci_high,
                color="grey",
                alpha=0.10,
                zorder=0,
            )
    if niah_stats.baseline is not None:
        ax.axhline(niah_stats.baseline, color="grey", linestyle=NIAH_LINESTYLE, linewidth=1.2, alpha=0.6, zorder=1)
        if (
            niah_stats.baseline_ci_low is not None
            and niah_stats.baseline_ci_high is not None
            and niah_stats.baseline_ci_high > niah_stats.baseline_ci_low
        ):
            ax.axhspan(
                niah_stats.baseline_ci_low,
                niah_stats.baseline_ci_high,
                color="grey",
                alpha=0.08,
                zorder=0,
            )

    # Ablation curves — same color, style encodes dataset (excluded from auto-legend)
    ax.plot(
        nolima_stats.ks,
        nolima_stats.scores,
        color=color,
        linestyle=NOLIMA_LINESTYLE,
        marker="o",
        markersize=4,
        linewidth=2,
        zorder=3,
    )
    if any(high > low for low, high in zip(nolima_stats.ci_lows, nolima_stats.ci_highs)):
        ax.fill_between(
            nolima_stats.ks,
            nolima_stats.ci_lows,
            nolima_stats.ci_highs,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
    ax.plot(
        niah_stats.ks,
        niah_stats.scores,
        color=color,
        linestyle=NIAH_LINESTYLE,
        marker="o",
        markersize=4,
        linewidth=2,
        zorder=3,
    )
    if any(high > low for low, high in zip(niah_stats.ci_lows, niah_stats.ci_highs)):
        ax.fill_between(
            niah_stats.ks,
            niah_stats.ci_lows,
            niah_stats.ci_highs,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )

    # Proxy artists encoding the two legend dimensions:
    #   color: gray = baseline, blue = LOCOS
    #   style: solid = NoLiMa, dash-dot = NIAH
    ax.plot([], [], color="grey", linewidth=2, label="Baseline")
    ax.plot([], [], color=color, linewidth=2, label="LOCOS")
    ax.plot([], [], color="black", linestyle=NOLIMA_LINESTYLE, linewidth=2, marker="o", markersize=4, label="NoLiMa")
    ax.plot([], [], color="black", linestyle=NIAH_LINESTYLE, linewidth=2, marker="o", markersize=4, label="NIAH")

    all_ks = sorted(set(nolima_stats.ks) | set(niah_stats.ks))
    ax.set_xticks(all_ks)
    ax.set_xticklabels([str(k) for k in all_ks])
    pretty = MODEL_PRETTY_NAMES.get(model, model)
    ax.set_title(pretty)
    ax.set_xlabel(r"Heads ablated ($k$)")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))

    fig.tight_layout()
    ax.legend(loc="lower left", fontsize=8, ncol=4)
    save_figure(fig, out_path, keep_title=True)
    console.print(f"[green]Saved:[/green] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-model NIAH vs NoLiMa ablation figures for LOCOS heads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--ablation-mode", type=str, default="mean", choices=["zero", "mean"])
    parser.add_argument("--metric", type=str, default="rouge_l_mean", choices=sorted(METRIC_LABELS))
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
        help="Directory containing *_trials.jsonl sidecars. Defaults to --results-dir.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/locos_niah_vs_nolima"),
        help="Directory where per-model SVGs are written",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        nolima_path = args.results_dir / f"nolima_ablation_{model}_logit_contrib_nolima.json"
        niah_path = args.results_dir / f"niah_ablation_{model}_logit_contrib_nolima.json"
        niah_baseline_path = args.results_dir / f"niah_ablation_{model}_random_seed42.json"

        assert nolima_path.exists(), f"NoLiMa cache missing: {nolima_path}"
        assert niah_path.exists(), f"NIAH cache missing: {niah_path}"

        nolima_cache = load_cache(nolima_path)
        niah_cache = load_cache(niah_path)

        trial_dir = args.trial_dir or args.results_dir
        nolima_stats = extract_sweep_stats(
            nolima_cache,
            nolima_path,
            args.ablation_mode,
            args.metric,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            trial_dir=trial_dir,
        )
        niah_stats = extract_sweep_stats(
            niah_cache,
            niah_path,
            args.ablation_mode,
            args.metric,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            trial_dir=trial_dir,
        )

        # NoLiMa baseline: fall back to random-seed cache when absent from the LOCOS cache (e.g. Gemma)
        nolima_baseline_path = args.results_dir / f"nolima_ablation_{model}_random_seed42.json"
        if nolima_stats.baseline is None and nolima_baseline_path.exists():
            fallback_stats = extract_sweep_stats(
                load_cache(nolima_baseline_path),
                nolima_baseline_path,
                args.ablation_mode,
                args.metric,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                trial_dir=trial_dir,
            )
            nolima_stats = SweepStats(
                ks=nolima_stats.ks,
                scores=nolima_stats.scores,
                ci_lows=nolima_stats.ci_lows,
                ci_highs=nolima_stats.ci_highs,
                baseline=fallback_stats.baseline,
                baseline_ci_low=fallback_stats.baseline_ci_low,
                baseline_ci_high=fallback_stats.baseline_ci_high,
            )

        # NIAH baseline from random-seed cache (has a baseline entry)
        if niah_baseline_path.exists():
            niah_bl_cache = load_cache(niah_baseline_path)
            fallback_stats = extract_sweep_stats(
                niah_bl_cache,
                niah_baseline_path,
                args.ablation_mode,
                args.metric,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                trial_dir=trial_dir,
            )
            niah_stats = SweepStats(
                ks=niah_stats.ks,
                scores=niah_stats.scores,
                ci_lows=niah_stats.ci_lows,
                ci_highs=niah_stats.ci_highs,
                baseline=fallback_stats.baseline,
                baseline_ci_low=fallback_stats.baseline_ci_low,
                baseline_ci_high=fallback_stats.baseline_ci_high,
            )

        out_path = args.out_dir / f"{model}_{args.metric}.svg"
        make_model_figure(
            model,
            nolima_stats,
            niah_stats,
            args.metric,
            out_path,
        )

        # Summary table per model
        pretty = MODEL_PRETTY_NAMES.get(model, model)
        table = Table(title=pretty)
        table.add_column("k", justify="right", style="bold")
        table.add_column("NoLiMa", justify="right")
        table.add_column("NIAH", justify="right")
        nolima_map = dict(zip(nolima_stats.ks, nolima_stats.scores))
        niah_map = dict(zip(niah_stats.ks, niah_stats.scores))
        for k in sorted(set(nolima_stats.ks) | set(niah_stats.ks)):
            table.add_row(
                str(k),
                f"{nolima_map[k]:.3f}" if k in nolima_map else "—",
                f"{niah_map[k]:.3f}" if k in niah_map else "—",
            )
        console.print(table)


if __name__ == "__main__":
    main()
