#!/usr/bin/env python3
"""Plot Functional Dissociation Score (DS) across head ablation methods.

Computes DS(k) = Delta_R(k) - Delta_P(k) where:
  Delta_R(k) = (R_0 - R(k)) / R_0  (normalized retrieval drop)
  Delta_P(k) = (P_0 - P(k)) / P_0  (normalized parametric drop)

k* = argmax_k DS(k) identifies the point of maximum retrieval specificity.

Requires paired NoLiMa (retrieval) and parametric ablation caches for each
method being compared.

Usage:
    # Compare methods
    python locos/plotting/functional_dissociation.py \
        --nolima \
            ablation_results/nolima_ablation_Model_logit_contrib.json \
            ablation_results/nolima_ablation_Model_random_seed42.json \
        --parametric \
            ablation_results/parametric_ablation_Model_logit_contrib.json \
            ablation_results/parametric_ablation_Model_random_seed42.json \
        --labels "Logit Contribution" "Random" \
        --out figures/functional_dissociation.svg

    # Per-source breakdown (separate DS for city_country, popqa, arithmetic)
    python locos/plotting/functional_dissociation.py \
        --nolima ... --parametric ... --labels ... \
        --per-source \
        --out figures/ds_per_source.svg

Requires: matplotlib, numpy, rich
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from locos_eval.utils.plotting import (  # FIGURE_SIZE,
    LINE_WIDTH,
    MODEL_PRETTY_NAMES,
    facecolor_alpha,
    save_figure,
    save_legend,
    setup_plot_style,
)

console = Console()

COLOR_CODES = ["#0563BA", "#BE1E2D", "#DB808D", "#E8972E", "#41A9AC"]
COLORS = COLOR_CODES
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]

# Per-source styling for breakdown plots
SOURCE_STYLES = {
    "accuracy": {"label": "Overall Parametric", "linestyle": "-"},
    "city_country_accuracy": {"label": "City-Country", "linestyle": "--"},
    "popqa_accuracy": {"label": "PopQA", "linestyle": "-."},
    "arithmetic_accuracy": {"label": "Arithmetic", "linestyle": ":"},
}

FIGURE_SIZE = (3.5, 3)


@dataclass(frozen=True)
class DSStats:
    """DS point estimates plus optional bootstrap intervals."""

    ks: list[int]
    ds: list[float]
    ci_lows: list[float]
    ci_highs: list[float]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_cache(path: Path) -> dict:
    """Load ablation results cache."""
    assert path.exists(), f"Cache file not found: {path}"
    with open(path) as f:
        return json.load(f)


def extract_nolima(cache: dict) -> tuple[dict[int, float], float | None]:
    """Extract {k: rouge_l_mean} and baseline from NoLiMa cache."""
    baseline = None
    runs: dict[int, float] = {}

    for _, metrics in cache.items():
        mode = metrics.get("mode", "")
        if mode in ("greedy", "baseline"):
            baseline = metrics.get("rouge_l_mean")
            continue
        k = int(metrics.get("value", 0))
        if k > 0:
            runs[k] = metrics.get("rouge_l_mean", 0)

    return runs, baseline


def extract_parametric(
    cache: dict,
) -> tuple[dict[int, dict[str, float]], dict[str, float] | None]:
    """Extract {k: {metric: val}} and baseline from parametric cache."""
    baseline = None
    runs: dict[int, dict[str, float]] = {}

    for _, metrics in cache.items():
        mode = metrics.get("mode", "")
        if mode in ("greedy", "baseline"):
            baseline = {k: v for k, v in metrics.items() if k.endswith("_accuracy") or k == "accuracy"}
            continue
        k = int(metrics.get("value", 0))
        if k > 0:
            runs[k] = {k_: v for k_, v in metrics.items() if k_.endswith("_accuracy") or k_ == "accuracy"}

    return runs, baseline


def _find_baseline_metrics(cache: dict) -> dict | None:
    for _, metrics in cache.items():
        if metrics.get("mode", "") in ("greedy", "baseline"):
            return metrics
    return None


def _find_run_metrics_by_k(cache: dict) -> dict[int, dict]:
    runs = {}
    for _, metrics in cache.items():
        if metrics.get("mode", "") in ("greedy", "baseline"):
            continue
        k = int(metrics.get("value", 0))
        if k > 0:
            runs[k] = metrics
    return runs


def _safe_label(label: str) -> str:
    return label.replace("=", "_").replace(".", "p")


def _trial_label(metrics: dict) -> str:
    mode = metrics.get("mode", "")
    if mode in ("baseline", "greedy"):
        return "baseline"

    value = float(metrics.get("value", 0))
    mode_label = "random" if metrics.get("random_heads", False) else mode
    return f"{mode_label}={value}"


def _nolima_trial_path(cache_path: Path, metrics: dict, trial_dir: Path | None = None) -> Path:
    dataset = metrics.get("dataset")
    stem = cache_path.stem
    if dataset is None:
        dataset = stem.split("_ablation_", maxsplit=1)[0] if "_ablation_" in stem else "nolima"
    heads_label = stem.removeprefix(f"{dataset}_ablation_")
    base_dir = trial_dir if trial_dir is not None else cache_path.parent
    return base_dir / f"{dataset}_ablation_{heads_label}_{_safe_label(_trial_label(metrics))}_trials.jsonl"


def _parametric_trial_path(cache_path: Path, metrics: dict, trial_dir: Path | None = None) -> Path:
    heads_label = cache_path.stem.removeprefix("parametric_ablation_")
    base_dir = trial_dir if trial_dir is not None else cache_path.parent
    return base_dir / f"parametric_ablation_{heads_label}_{_safe_label(_trial_label(metrics))}_trials.jsonl"


def _read_nolima_trials(path: Path) -> list[float] | None:
    if not path.exists():
        return None
    values = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                values.append(float(json.loads(line)["rouge_l"]))
    return values


def _parametric_source_from_key(parametric_key: str) -> str | None:
    if parametric_key == "accuracy":
        return None
    assert parametric_key.endswith("_accuracy"), f"Unsupported parametric key: {parametric_key}"
    return parametric_key.removesuffix("_accuracy")


def _read_parametric_trials(path: Path, parametric_key: str) -> list[float] | None:
    if not path.exists():
        return None
    source = _parametric_source_from_key(parametric_key)
    values = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if source is not None and row["source"] != source:
                continue
            values.append(float(bool(row["correct"])))
    return values


def _matches_mean_and_count(values: list[float], target_mean: float | None, target_n: int | None) -> bool:
    if target_n is not None and len(values) != target_n:
        return False
    return target_mean is None or np.isclose(float(np.mean(values)), target_mean, atol=1e-12)


def nolima_trial_values_for_metrics(
    cache_path: Path, metrics: dict, trial_dir: Path | None = None
) -> list[float] | None:
    """Load NoLiMa per-trial ROUGE-L values for a cache entry."""
    direct_path = _nolima_trial_path(cache_path, metrics, trial_dir)
    direct_values = _read_nolima_trials(direct_path)
    if direct_values is not None:
        return direct_values

    dataset = metrics.get("dataset") or cache_path.stem.split("_ablation_", maxsplit=1)[0]
    base_dir = trial_dir if trial_dir is not None else cache_path.parent
    pattern = f"{dataset}_ablation_*_{_safe_label(_trial_label(metrics))}_trials.jsonl"
    target_mean = metrics.get("rouge_l_mean")
    target_n = metrics.get("n_samples")

    for candidate in sorted(base_dir.glob(pattern)):
        values = _read_nolima_trials(candidate)
        if values and _matches_mean_and_count(values, target_mean, target_n):
            return values
    return None


def parametric_trial_values_for_metrics(
    cache_path: Path,
    metrics: dict,
    parametric_key: str,
    trial_dir: Path | None = None,
) -> list[float] | None:
    """Load parametric per-sample correctness values for a cache entry."""
    direct_path = _parametric_trial_path(cache_path, metrics, trial_dir)
    direct_values = _read_parametric_trials(direct_path, parametric_key)
    if direct_values is not None:
        return direct_values

    base_dir = trial_dir if trial_dir is not None else cache_path.parent
    pattern = f"parametric_ablation_*_{_safe_label(_trial_label(metrics))}_trials.jsonl"
    source = _parametric_source_from_key(parametric_key)
    target_mean = metrics.get(parametric_key)
    target_n = metrics.get("n_samples" if source is None else f"{source}_n")

    for candidate in sorted(base_dir.glob(pattern)):
        values = _read_parametric_trials(candidate, parametric_key)
        if values and _matches_mean_and_count(values, target_mean, target_n):
            return values
    return None


# ---------------------------------------------------------------------------
# DS computation
# ---------------------------------------------------------------------------


def compute_ds(
    nolima_runs: dict[int, float],
    nolima_baseline: float,
    parametric_runs: dict[int, dict[str, float]],
    parametric_baseline: dict[str, float],
    parametric_key: str = "accuracy",
    k_filter: set[int] | None = None,
) -> tuple[list[int], list[float]]:
    """Compute DS(k) = Delta_R(k) - Delta_P(k) for shared k values.

    Args:
        nolima_runs: {k: rouge_l_mean} from ablation.
        nolima_baseline: R_0 (no ablation).
        parametric_runs: {k: {metric: val}} from ablation.
        parametric_baseline: {metric: val} at baseline.
        parametric_key: Which parametric metric to use as P(k).
        k_filter: If provided, only include these k values.

    Returns:
        (sorted_k_values, ds_values)
    """
    assert nolima_baseline > 0, f"NoLiMa baseline must be positive, got {nolima_baseline}"
    p0 = parametric_baseline.get(parametric_key, 0)
    assert p0 > 0, f"Parametric baseline ({parametric_key}) must be positive, got {p0}"

    shared_k = sorted(set(nolima_runs.keys()) & set(parametric_runs.keys()))
    if k_filter is not None:
        shared_k = [k for k in shared_k if k in k_filter]
    assert len(shared_k) > 0, "No shared k values between NoLiMa and parametric caches"

    ds_values = []
    for k in shared_k:
        delta_r = (nolima_baseline - nolima_runs[k]) / nolima_baseline
        p_k = parametric_runs[k].get(parametric_key, 0)
        delta_p = (p0 - p_k) / p0
        ds_values.append(delta_r - delta_p)

    return shared_k, ds_values


def _bootstrap_ds_ci(
    nolima_baseline_trials: list[float],
    nolima_k_trials: list[float],
    parametric_baseline_trials: list[float],
    parametric_k_trials: list[float],
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, float] | None:
    if bootstrap_samples <= 0:
        return None
    if not nolima_baseline_trials or not nolima_k_trials or not parametric_baseline_trials or not parametric_k_trials:
        return None

    rng = np.random.default_rng(seed)
    r0 = np.asarray(nolima_baseline_trials, dtype=float)
    rk = np.asarray(nolima_k_trials, dtype=float)
    p0 = np.asarray(parametric_baseline_trials, dtype=float)
    pk = np.asarray(parametric_k_trials, dtype=float)

    r0_means = r0[rng.integers(0, len(r0), size=(bootstrap_samples, len(r0)))].mean(axis=1)
    rk_means = rk[rng.integers(0, len(rk), size=(bootstrap_samples, len(rk)))].mean(axis=1)
    p0_means = p0[rng.integers(0, len(p0), size=(bootstrap_samples, len(p0)))].mean(axis=1)
    pk_means = pk[rng.integers(0, len(pk), size=(bootstrap_samples, len(pk)))].mean(axis=1)

    valid = (r0_means > 0) & (p0_means > 0)
    if not np.any(valid):
        return None
    boot_ds = ((r0_means[valid] - rk_means[valid]) / r0_means[valid]) - (
        (p0_means[valid] - pk_means[valid]) / p0_means[valid]
    )
    low, high = np.percentile(boot_ds, [2.5, 97.5])
    return float(low), float(high)


def compute_ds_stats(
    nolima_runs: dict[int, float],
    nolima_baseline: float,
    parametric_runs: dict[int, dict[str, float]],
    parametric_baseline: dict[str, float],
    parametric_key: str = "accuracy",
    k_filter: set[int] | None = None,
    nolima_trial_values: dict[int, list[float]] | None = None,
    nolima_baseline_trials: list[float] | None = None,
    parametric_trial_values: dict[int, list[float]] | None = None,
    parametric_baseline_trials: list[float] | None = None,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> DSStats:
    """Compute DS(k), optionally with bootstrap CIs over trial-level scores."""
    ks, ds = compute_ds(nolima_runs, nolima_baseline, parametric_runs, parametric_baseline, parametric_key, k_filter)
    ci_lows = list(ds)
    ci_highs = list(ds)

    if (
        nolima_trial_values is None
        or nolima_baseline_trials is None
        or parametric_trial_values is None
        or parametric_baseline_trials is None
    ):
        return DSStats(ks=ks, ds=ds, ci_lows=ci_lows, ci_highs=ci_highs)

    for idx, k in enumerate(ks):
        ci = _bootstrap_ds_ci(
            nolima_baseline_trials=nolima_baseline_trials,
            nolima_k_trials=nolima_trial_values.get(k, []),
            parametric_baseline_trials=parametric_baseline_trials,
            parametric_k_trials=parametric_trial_values.get(k, []),
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed + k * 1009,
        )
        if ci is not None:
            ci_lows[idx], ci_highs[idx] = ci

    return DSStats(ks=ks, ds=ds, ci_lows=ci_lows, ci_highs=ci_highs)


def find_k_star(ks: list[int], ds: list[float]) -> tuple[int, float]:
    """Find k* = argmax_k DS(k)."""
    idx = int(np.argmax(ds))
    return ks[idx], ds[idx]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_ds_plot(
    method_data: dict[str, DSStats],
    out_path: Path,
    model_name: str | None = None,
    color_indices: list[int] | None = None,
    marker_indices: list[int] | None = None,
) -> None:
    """Create DS comparison plot: one line per method, k* annotated.

    Args:
        method_data: {label: DSStats}.
        out_path: Output SVG path.
        model_name: Raw model name (looked up in MODEL_PRETTY_NAMES for title).
        color_indices: Per-method indices into COLORS palette.
        marker_indices: Per-method indices into MARKERS list.
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # DS = 0 reference line
    ax.axhline(0, color="grey", linestyle="--", linewidth=1.2, alpha=0.5, zorder=1)

    for i, (label, stats) in enumerate(method_data.items()):
        if not stats.ks:
            continue
        ci = color_indices[i] if color_indices else i
        mi = marker_indices[i] if marker_indices else i
        color = COLORS[ci % len(COLORS)]
        marker = MARKERS[mi % len(MARKERS)]

        ax.plot(
            stats.ks,
            stats.ds,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=LINE_WIDTH,
            zorder=3,
            label=label,
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

        # Annotate k*
        k_star, ds_star = find_k_star(stats.ks, stats.ds)
        ax.plot(
            k_star,
            ds_star,
            marker=marker,
            markersize=10,
            color=color,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=5,
        )

    ax.set_xlabel(r"Heads ablated ($k$)")
    ax.set_ylabel(r"DS($k$)")

    # Integer x-ticks
    all_ks = sorted({k for stats in method_data.values() for k in stats.ks})
    ax.set_xticks(all_ks)
    ax.set_xticklabels([str(k) for k in all_ks])

    # Legend (extracted and saved separately by save_figure)
    ax.legend(loc="best", ncol=len(method_data))

    keep_title = False
    if model_name:
        ax.set_title(MODEL_PRETTY_NAMES.get(model_name, model_name))
        keep_title = True

    fig.tight_layout()
    save_figure(fig, out_path, keep_title=keep_title)
    console.print(f"[green]Saved:[/green] {out_path}")


def make_per_source_ds_plot(
    nolima_caches: list[tuple[str, dict[int, float], float]],
    parametric_caches: list[tuple[str, dict[int, dict[str, float]], dict[str, float]]],
    out_path: Path,
    k_filter: set[int] | None = None,
    model_name: str | None = None,
    per_source_stats: list[dict[str, DSStats]] | None = None,
) -> None:
    """Create per-source DS breakdown: one subplot per method, lines per parametric source.

    Shows whether retrieval specificity holds uniformly across parametric
    sources or is driven by a single one.
    """
    setup_plot_style()

    n_methods = len(nolima_caches)
    fig, axes = plt.subplots(
        1,
        n_methods,
        figsize=(FIGURE_SIZE[0] * n_methods, FIGURE_SIZE[1]),
        squeeze=False,
        sharey=True,
    )

    source_colors = {
        "accuracy": COLORS[0],
        "city_country_accuracy": COLORS[1],
        "popqa_accuracy": COLORS[2],
        "arithmetic_accuracy": COLORS[3],
    }

    all_handles = []
    all_labels = []

    for idx, ((label, nolima_runs, nolima_bl), (_, param_runs, param_bl)) in enumerate(
        zip(nolima_caches, parametric_caches)
    ):
        ax = axes[0, idx]
        ax.axhline(0, color="grey", linestyle="--", linewidth=1.2, alpha=0.5, zorder=1)

        for source_key, style in SOURCE_STYLES.items():
            if param_bl.get(source_key) is None or param_bl[source_key] == 0:
                continue

            if per_source_stats is not None:
                stats = per_source_stats[idx][source_key]
            else:
                ks, ds = compute_ds(
                    nolima_runs, nolima_bl, param_runs, param_bl, parametric_key=source_key, k_filter=k_filter
                )
                stats = DSStats(ks=ks, ds=ds, ci_lows=list(ds), ci_highs=list(ds))
            color = source_colors[source_key]

            (line,) = ax.plot(
                stats.ks,
                stats.ds,
                color=color,
                linestyle=style["linestyle"],
                marker="o",
                markersize=4,
                linewidth=LINE_WIDTH,
                zorder=3,
            )
            if any(high > low for low, high in zip(stats.ci_lows, stats.ci_highs)):
                ax.fill_between(
                    stats.ks,
                    stats.ci_lows,
                    stats.ci_highs,
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                    zorder=2,
                )

            # Annotate k*
            k_star, ds_star = find_k_star(stats.ks, stats.ds)
            ax.plot(
                k_star,
                ds_star,
                marker="o",
                markersize=8,
                color=color,
                markeredgecolor="black",
                markeredgewidth=1.2,
                zorder=5,
            )

            if idx == 0:
                all_handles.append(line)
                all_labels.append(style["label"])

        ax.set_xlabel(r"Heads ablated ($k$)")
        if idx == 0:
            ax.set_ylabel(r"DS($k$)")

        # Method label as annotation (not title)
        ax.annotate(
            label,
            xy=(0.5, 0.97),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=10,
            fontstyle="italic",
        )

        all_ks = sorted(nolima_runs.keys())
        ax.set_xticks(all_ks)
        ax.set_xticklabels([str(k) for k in all_ks])

    keep_title = False
    if model_name:
        fig.suptitle(MODEL_PRETTY_NAMES.get(model_name, model_name))
        keep_title = True

    fig.tight_layout()
    save_figure(fig, out_path, keep_title=keep_title)

    # Save per-source legend
    legend_path = out_path.with_name(out_path.stem + "_legend" + out_path.suffix)
    save_legend(all_handles, all_labels, legend_path, ncol=len(all_labels))

    console.print(f"[green]Saved:[/green] {out_path}")
    console.print(f"[green]Saved:[/green] {legend_path}")


def make_max_ds_bar_plot(
    method_data: dict[str, tuple[list[int], list[float]]],
    out_path: Path,
    model_name: str | None = None,
    color_indices: list[int] | None = None,
) -> None:
    """Bar chart of max DS per method — compact summary of retrieval specificity.

    Each bar shows max_k DS(k) for one method. The k* value is annotated
    above each bar.
    """
    setup_plot_style()

    labels = list(method_data.keys())
    max_ds = []
    k_stars = []
    for label in labels:
        ks, ds = method_data[label]
        k_star, ds_star = find_k_star(ks, ds)
        max_ds.append(ds_star)
        k_stars.append(k_star)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    x = np.arange(len(labels))
    colors = [COLORS[(color_indices[i] if color_indices else i) % len(COLORS)] for i in range(len(labels))]

    bars = ax.bar(
        x,
        max_ds,
        color=colors,
        edgecolor="black",
        linewidth=LINE_WIDTH,
        zorder=3,
    )

    # DS = 0 reference line
    ax.axhline(0, color="grey", linestyle="--", linewidth=1.2, alpha=0.5, zorder=1)

    # Annotate k* above each bar
    for bar, k_star, ds_val in zip(bars, k_stars, max_ds):
        y_offset = 0.01 if ds_val >= 0 else -0.03
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ds_val + y_offset,
            r"$k^*$" + f"$= {k_star}$",
            ha="center",
            va="bottom" if ds_val >= 0 else "top",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(r"max$_k$ DS($k$)")

    keep_title = False
    if model_name:
        ax.set_title(MODEL_PRETTY_NAMES.get(model_name, model_name))
        keep_title = True

    fig.tight_layout()
    save_figure(fig, out_path, keep_title=keep_title)
    console.print(f"[green]Saved:[/green] {out_path}")


def make_dual_axis_plot(
    method_data: dict[str, DSStats],
    parametric_caches: list[tuple[str, dict[int, dict[str, float]], dict[str, float]]],
    out_path: Path,
    model_name: str | None = None,
    color_indices: list[int] | None = None,
    marker_indices: list[int] | None = None,
) -> None:
    """Dual-axis plot: grouped bars (parametric accuracy) + lines (DS).

    Left y-axis: aggregate parametric accuracy (bars, semi-transparent fill).
    Right y-axis: DS (lines, solid).
    Colors are consistent per method across bars and lines.
    """
    setup_plot_style()

    labels = list(method_data.keys())
    n_methods = len(labels)
    all_ks = sorted({k for stats in method_data.values() for k in stats.ks})
    n_k = len(all_ks)

    fig, ax_left = plt.subplots(figsize=FIGURE_SIZE)
    ax_right = ax_left.twinx()

    # Bar geometry
    bar_width = 0.8 / n_methods
    offsets = np.arange(n_methods) - (n_methods - 1) / 2
    x = np.arange(n_k)

    handles = []
    bar_labels = []
    line_labels = []

    for i, label in enumerate(labels):
        ci = color_indices[i] if color_indices else i
        mi = marker_indices[i] if marker_indices else i
        color = COLORS[ci % len(COLORS)]
        marker = MARKERS[mi % len(MARKERS)]
        _, param_runs, param_bl = parametric_caches[i]
        stats = method_data[label]

        # Bars: parametric accuracy (left y-axis)
        acc_vals = [param_runs.get(k, {}).get("accuracy", float("nan")) for k in all_ks]
        positions = x + offsets[i] * bar_width

        bar = ax_left.bar(
            positions,
            acc_vals,
            width=bar_width,
            color=facecolor_alpha(color, 0.35),
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )
        handles.append(bar[0])
        bar_labels.append(f"{label} (Acc.)")

        # Lines: DS (right y-axis)
        # Map DS k values to x positions
        ds_x = [all_ks.index(k) for k in stats.ks]
        (line,) = ax_right.plot(
            ds_x,
            stats.ds,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=LINE_WIDTH,
            zorder=4,
        )
        handles.append(line)
        line_labels.append(f"{label} (DS)")
        if any(high > low for low, high in zip(stats.ci_lows, stats.ci_highs)):
            ax_right.fill_between(
                ds_x,
                stats.ci_lows,
                stats.ci_highs,
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=3,
            )

        # Annotate k*
        k_star, ds_star = find_k_star(stats.ks, stats.ds)
        star_x = all_ks.index(k_star)
        ax_right.plot(
            star_x,
            ds_star,
            marker=marker,
            markersize=10,
            color=color,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=6,
        )

    # Parametric baseline as horizontal line (left axis)
    # Use first available baseline
    baseline_acc = None
    for _, _, param_bl in parametric_caches:
        if param_bl is not None and "accuracy" in param_bl:
            baseline_acc = param_bl["accuracy"]
            break
    if baseline_acc is not None:
        bl_line = ax_left.axhline(
            baseline_acc,
            color="grey",
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
            zorder=1,
        )
        handles.append(bl_line)

    # DS = 0 reference (right axis)
    # ax_right.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.4, zorder=1)

    # Axes
    ax_left.set_xticks(x)
    ax_left.set_xticklabels([str(k) for k in all_ks])
    ax_left.set_xlabel(r"Heads ablated ($k$)")
    ax_left.set_ylabel("Parametric Accuracy")
    ax_right.set_ylabel(r"DS($k$)")

    ax_left.set_ylim(bottom=0, top=1.05)

    # Remove y-axis ticks and grid (plain box)
    ax_left.tick_params(axis="y", length=0)
    ax_right.tick_params(axis="y", length=0)
    ax_left.grid(False)
    ax_right.grid(False)

    # Combined legend: interleave bar/line per method, then baseline
    combined_handles = []
    combined_labels = []
    for i in range(n_methods):
        combined_handles.append(handles[i * 2])
        combined_labels.append(bar_labels[i])
        combined_handles.append(handles[i * 2 + 1])
        combined_labels.append(line_labels[i])
    if baseline_acc is not None:
        combined_handles.append(handles[-1])
        combined_labels.append("Baseline (no ablation)")
    ax_left.legend(combined_handles, combined_labels, loc="best", ncol=len(combined_labels))

    keep_title = False
    if model_name:
        ax_left.set_title(MODEL_PRETTY_NAMES.get(model_name, model_name))
        keep_title = True

    fig.tight_layout()
    save_figure(fig, out_path, keep_title=keep_title)
    console.print(f"[green]Saved:[/green] {out_path}")


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_ds_table(
    method_data: dict[str, tuple[list[int], list[float]]],
    nolima_caches: list[tuple[str, dict[int, float], float]],
    parametric_caches: list[tuple[str, dict[int, dict[str, float]], dict[str, float]]],
) -> None:
    """Print DS summary table with raw metrics and k* highlighted."""
    labels = list(method_data.keys())

    table = Table()
    table.add_column(r"k", justify="right", style="bold")
    for label in labels:
        table.add_column(f"{label} DS", justify="right")
        table.add_column(r"Delta_R", justify="right")
        table.add_column(r"Delta_P", justify="right")

    all_ks = sorted({k for ks, _ in method_data.values() for k in ks})

    # Pre-compute k* for each method to highlight
    k_stars = {}
    for label, (ks, ds) in method_data.items():
        k_star, _ = find_k_star(ks, ds)
        k_stars[label] = k_star

    for k in all_ks:
        row = [str(k)]
        for i, label in enumerate(labels):
            ks, ds = method_data[label]
            _, nolima_runs, nolima_bl = nolima_caches[i]
            _, param_runs, param_bl = parametric_caches[i]

            if k in dict(zip(ks, ds)):
                idx = ks.index(k)
                ds_val = ds[idx]

                delta_r = (nolima_bl - nolima_runs[k]) / nolima_bl
                p0 = param_bl.get("accuracy", 0)
                p_k = param_runs[k].get("accuracy", 0)
                delta_p = (p0 - p_k) / p0 if p0 > 0 else 0

                if k == k_stars[label]:
                    row.append(f"[bold green]{ds_val:.3f} *[/bold green]")
                else:
                    row.append(f"{ds_val:.3f}")
                row.append(f"{delta_r:.3f}")
                row.append(f"{delta_p:.3f}")
            else:
                row.extend(["---", "---", "---"])
        table.add_row(*row)

    console.print()
    console.rule("[bold]Functional Dissociation Score[/bold]")
    console.print(table)

    # k* summary
    console.print()
    for label in labels:
        k_star = k_stars[label]
        ks, ds = method_data[label]
        idx = ks.index(k_star)
        console.print(f"  [bold]{label}[/bold]: k* = {k_star}  (DS = {ds[idx]:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot Functional Dissociation Score across head ablation methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nolima",
        nargs="+",
        type=Path,
        required=True,
        help="NoLiMa ablation cache JSON file(s), one per method (same order as --parametric)",
    )
    parser.add_argument(
        "--parametric",
        nargs="+",
        type=Path,
        required=True,
        help="Parametric ablation cache JSON file(s), one per method (same order as --nolima)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        required=True,
        help="Display labels for each method (same order as --nolima/--parametric)",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=None,
        help="Only include these k values (e.g. --k-values 1 5 10 20 50 100)",
    )
    parser.add_argument(
        "--per-source",
        action="store_true",
        help="Also generate per-source DS breakdown (city_country, popqa, arithmetic)",
    )
    parser.add_argument(
        "--bar",
        action="store_true",
        help="Also generate bar chart of max DS per method",
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Also generate dual-axis plot (bars: parametric accuracy, lines: DS)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples over NoLiMa and parametric sidecar trials. Set 0 to disable.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Random seed for DS bootstrap resampling.",
    )
    parser.add_argument(
        "--nolima-trial-dir",
        type=Path,
        default=None,
        help="Directory containing NoLiMa *_trials.jsonl sidecars. Defaults to each NoLiMa cache directory.",
    )
    parser.add_argument(
        "--parametric-trial-dir",
        type=Path,
        default=None,
        help="Directory containing parametric *_trials.jsonl sidecars. Defaults to each parametric cache directory.",
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
        default=Path("figures/functional_dissociation.svg"),
        help="Output SVG path",
    )
    args = parser.parse_args()

    assert len(args.nolima) == len(args.parametric) == len(args.labels), (
        f"Must provide same number of --nolima ({len(args.nolima)}), "
        f"--parametric ({len(args.parametric)}), and --labels ({len(args.labels)})"
    )

    # Pass 1: load all caches and find shared baselines.
    # The baseline (no ablation) may only appear in one cache file (e.g. the
    # random-heads run), so we scan all files first and use the first baseline
    # found for each modality.
    raw_nolima: list[tuple[str, Path, dict[int, float], float | None, dict[int, dict], dict | None]] = []
    raw_parametric: list[
        tuple[str, Path, dict[int, dict[str, float]], dict[str, float] | None, dict[int, dict], dict | None]
    ] = []

    shared_nolima_bl: float | None = None
    shared_param_bl: dict[str, float] | None = None
    shared_nolima_bl_ref: tuple[Path, dict] | None = None
    shared_param_bl_ref: tuple[Path, dict] | None = None

    for nolima_path, param_path, label in zip(args.nolima, args.parametric, args.labels):
        nolima_cache = load_cache(nolima_path)
        param_cache = load_cache(param_path)
        nolima_runs, nolima_bl = extract_nolima(nolima_cache)
        param_runs, param_bl = extract_parametric(param_cache)
        nolima_metrics_by_k = _find_run_metrics_by_k(nolima_cache)
        param_metrics_by_k = _find_run_metrics_by_k(param_cache)
        nolima_bl_metrics = _find_baseline_metrics(nolima_cache)
        param_bl_metrics = _find_baseline_metrics(param_cache)

        assert len(nolima_runs) > 0, f"No ablation runs in {nolima_path}"
        assert len(param_runs) > 0, f"No ablation runs in {param_path}"

        if nolima_bl is not None and nolima_bl_metrics is not None and shared_nolima_bl is None:
            shared_nolima_bl = nolima_bl
            shared_nolima_bl_ref = (nolima_path, nolima_bl_metrics)
        if param_bl is not None and param_bl_metrics is not None and shared_param_bl is None:
            shared_param_bl = param_bl
            shared_param_bl_ref = (param_path, param_bl_metrics)

        raw_nolima.append((label, nolima_path, nolima_runs, nolima_bl, nolima_metrics_by_k, nolima_bl_metrics))
        raw_parametric.append((label, param_path, param_runs, param_bl, param_metrics_by_k, param_bl_metrics))

    assert shared_nolima_bl is not None, "No NoLiMa baseline found in any cache file"
    assert shared_param_bl is not None, "No parametric baseline found in any cache file"

    # Pass 2: fill in missing baselines and compute DS
    k_filter = set(args.k_values) if args.k_values else None

    nolima_caches: list[tuple[str, dict[int, float], float]] = []
    parametric_caches: list[tuple[str, dict[int, dict[str, float]], dict[str, float]]] = []
    method_data: dict[str, tuple[list[int], list[float]]] = {}
    method_stats: dict[str, DSStats] = {}
    per_source_stats: list[dict[str, DSStats]] = []

    shared_nolima_bl_trials = (
        nolima_trial_values_for_metrics(shared_nolima_bl_ref[0], shared_nolima_bl_ref[1], args.nolima_trial_dir)
        if shared_nolima_bl_ref is not None
        else None
    )

    for (label, nolima_path, nolima_runs, nolima_bl, nolima_metrics_by_k, nolima_bl_metrics), (
        _,
        param_path,
        param_runs,
        param_bl,
        param_metrics_by_k,
        param_bl_metrics,
    ) in zip(raw_nolima, raw_parametric):
        nolima_bl = nolima_bl if nolima_bl is not None else shared_nolima_bl
        param_bl = param_bl if param_bl is not None else shared_param_bl
        nolima_bl_ref = (nolima_path, nolima_bl_metrics) if nolima_bl_metrics is not None else shared_nolima_bl_ref
        param_bl_ref = (param_path, param_bl_metrics) if param_bl_metrics is not None else shared_param_bl_ref

        nolima_caches.append((label, nolima_runs, nolima_bl))
        parametric_caches.append((label, param_runs, param_bl))

        nolima_trials = {
            k: values
            for k, values in (
                (
                    k,
                    nolima_trial_values_for_metrics(nolima_path, metrics, args.nolima_trial_dir),
                )
                for k, metrics in nolima_metrics_by_k.items()
            )
            if values is not None
        }
        nolima_baseline_trials = (
            nolima_trial_values_for_metrics(nolima_bl_ref[0], nolima_bl_ref[1], args.nolima_trial_dir)
            if nolima_bl_ref is not None
            else shared_nolima_bl_trials
        )
        param_trials = {
            k: values
            for k, values in (
                (
                    k,
                    parametric_trial_values_for_metrics(param_path, metrics, "accuracy", args.parametric_trial_dir),
                )
                for k, metrics in param_metrics_by_k.items()
            )
            if values is not None
        }
        param_baseline_trials = (
            parametric_trial_values_for_metrics(param_bl_ref[0], param_bl_ref[1], "accuracy", args.parametric_trial_dir)
            if param_bl_ref is not None
            else None
        )

        stats = compute_ds_stats(
            nolima_runs,
            nolima_bl,
            param_runs,
            param_bl,
            k_filter=k_filter,
            nolima_trial_values=nolima_trials,
            nolima_baseline_trials=nolima_baseline_trials,
            parametric_trial_values=param_trials,
            parametric_baseline_trials=param_baseline_trials,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        method_stats[label] = stats
        method_data[label] = (stats.ks, stats.ds)

        source_stats = {}
        for source_key in SOURCE_STYLES:
            if param_bl.get(source_key) is None or param_bl[source_key] == 0:
                continue
            source_param_trials = {
                k: values
                for k, values in (
                    (
                        k,
                        parametric_trial_values_for_metrics(param_path, metrics, source_key, args.parametric_trial_dir),
                    )
                    for k, metrics in param_metrics_by_k.items()
                )
                if values is not None
            }
            source_param_baseline_trials = (
                parametric_trial_values_for_metrics(
                    param_bl_ref[0], param_bl_ref[1], source_key, args.parametric_trial_dir
                )
                if param_bl_ref is not None
                else None
            )
            source_stats[source_key] = compute_ds_stats(
                nolima_runs,
                nolima_bl,
                param_runs,
                param_bl,
                parametric_key=source_key,
                k_filter=k_filter,
                nolima_trial_values=nolima_trials,
                nolima_baseline_trials=nolima_baseline_trials,
                parametric_trial_values=source_param_trials,
                parametric_baseline_trials=source_param_baseline_trials,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            )
        per_source_stats.append(source_stats)

    # Main DS line plot
    make_ds_plot(method_stats, args.out, model_name=args.model, color_indices=args.colors, marker_indices=args.markers)

    # Bar chart of max DS
    if args.bar:
        bar_path = args.out.with_name(args.out.stem + "_bar" + args.out.suffix)
        make_max_ds_bar_plot(method_data, bar_path, model_name=args.model, color_indices=args.colors)

    # Dual-axis plot
    if args.dual:
        dual_path = args.out.with_name(args.out.stem + "_dual" + args.out.suffix)
        make_dual_axis_plot(
            method_stats,
            parametric_caches,
            dual_path,
            model_name=args.model,
            color_indices=args.colors,
            marker_indices=args.markers,
        )

    # Per-source breakdown
    if args.per_source:
        per_source_path = args.out.with_name(args.out.stem + "_per_source" + args.out.suffix)
        make_per_source_ds_plot(
            nolima_caches,
            parametric_caches,
            per_source_path,
            k_filter=k_filter,
            model_name=args.model,
            per_source_stats=per_source_stats,
        )

    # Console summary
    print_ds_table(method_data, nolima_caches, parametric_caches)


if __name__ == "__main__":
    main()
