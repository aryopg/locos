#!/usr/bin/env python3
"""Plot parametric/arithmetic ablation vs NoLiMa retrieval ablation.

Shows the dissociation: NoLiMa ROUGE-L collapses as retrieval heads are
ablated, while parametric recall (City-Country, PopQA) and arithmetic
accuracy remain stable.  This demonstrates that the ablated heads are
retrieval-specific, not generically output-critical.

Reads cached results from both ``nolima_ablation.py`` and
``parametric_ablation.py`` and produces a single figure with overlaid
curves.

Usage:
    # Single model — provide both cache files
    python locos/plotting/parametric_ablation.py \\
        --nolima ablation_results/nolima_ablation_Meta-Llama-3-8B-Instruct_logit_contrib.json \\
        --parametric ablation_results/parametric_ablation_Meta-Llama-3-8B-Instruct_logit_contrib.json

    # Multiple models — provide pairs
    python locos/plotting/parametric_ablation.py \\
        --nolima ablation_results/nolima_ablation_Model1.json \\
                 ablation_results/nolima_ablation_Model2.json \\
        --parametric ablation_results/parametric_ablation_Model1.json \\
                     ablation_results/parametric_ablation_Model2.json

    # Custom output path
    python locos/plotting/parametric_ablation.py \\
        --nolima ... --parametric ... \\
        --out figures/parametric_vs_retrieval.svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from locos_eval.utils.plotting import (
    FIGURE_SIZE,
    LINE_WIDTH,
    save_figure,
    save_legend,
    setup_plot_style,
)

# Solid colors for NoLiMa (retrieval), lighter tones for parametric sources
MODEL_COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD"]
MARKERS = ["o", "s", "D", "^", "v", "P"]

# Per-source colors and labels for grouped bar plots
SOURCE_COLORS = {
    "nolima": "#C44E52",
    "city_country": "#4C72B0",
    "popqa": "#55A868",
    "arithmetic": "#8172B2",
}
SOURCE_LABELS = {
    "nolima": "NoLiMa ROUGE-L",
    "city_country": "City-Country",
    "popqa": "PopQA",
    "arithmetic": "Arithmetic",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_cache(path: Path) -> dict:
    """Load ablation results cache."""
    assert path.exists(), f"Cache file not found: {path}"
    with open(path) as f:
        return json.load(f)


def extract_nolima_runs(cache: dict) -> tuple[dict[float, float], float | None]:
    """Extract (k -> ROUGE-L) mapping and optional baseline from nolima cache.

    Returns:
        ({k: rouge_l_mean}, baseline_rouge_l_or_None)
    """
    baseline = None
    runs: dict[float, float] = {}

    for _, metrics in cache.items():
        mode = metrics.get("mode", "")
        if mode in ("greedy", "baseline"):
            baseline = metrics.get("rouge_l_mean")
            continue
        value = metrics.get("value", 0)
        runs[value] = metrics.get("rouge_l_mean", 0)

    return runs, baseline


def extract_parametric_runs(
    cache: dict,
) -> tuple[dict[float, dict[str, float]], float | None]:
    """Extract (k -> {source: accuracy}) mapping and optional baseline.

    Returns:
        ({k: {accuracy, city_country_accuracy, popqa_accuracy, arithmetic_accuracy}},
         baseline_metrics_or_None)
    """
    baseline = None
    runs: dict[float, dict[str, float]] = {}

    for _, metrics in cache.items():
        mode = metrics.get("mode", "")
        if mode in ("greedy", "baseline"):
            baseline = {k: v for k, v in metrics.items() if k.endswith("_accuracy") or k == "accuracy"}
            continue
        value = metrics.get("value", 0)
        runs[value] = {k: v for k, v in metrics.items() if k.endswith("_accuracy") or k == "accuracy"}

    return runs, baseline


def model_label_from_path(path: Path, prefix: str) -> str:
    """Derive display label from cache filename."""
    return path.stem.removeprefix(f"{prefix}_")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_dissociation_plot(
    nolima_data: list[tuple[str, dict[float, float], float | None]],
    parametric_data: list[tuple[str, dict[float, dict[str, float]], float | None]],
    out_path: Path,
) -> None:
    """Create overlaid plot showing retrieval collapse vs parametric stability.

    For each model, plots:
    - NoLiMa ROUGE-L (solid, bold) — expected to collapse
    - Overall parametric accuracy (dashed) — expected to stay stable
    - Per-source accuracy lines (dotted, thin) — optional detail

    Uses dual y-axes: left for ROUGE-L, right for accuracy.
    """
    setup_plot_style()

    fig, ax_left = plt.subplots(figsize=FIGURE_SIZE)
    ax_right = ax_left.twinx()

    handles = []
    labels = []

    for i, ((nolima_label, nolima_runs, nolima_baseline), (_, param_runs, param_baseline)) in enumerate(
        zip(nolima_data, parametric_data)
    ):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        # NoLiMa ROUGE-L (left y-axis, solid)
        sorted_k = sorted(nolima_runs.keys())
        nolima_y = [nolima_runs[k] for k in sorted_k]

        (nolima_line,) = ax_left.plot(
            sorted_k,
            nolima_y,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2.5,
            zorder=3,
            label=f"{nolima_label} ROUGE-L",
        )
        handles.append(nolima_line)
        labels.append(f"{nolima_label} NoLiMa ROUGE-L")

        # NoLiMa baseline
        if nolima_baseline is not None:
            ax_left.axhline(
                nolima_baseline,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.4,
                zorder=1,
            )

        # Parametric overall accuracy (right y-axis, dashed)
        param_sorted_k = sorted(param_runs.keys())
        param_y = [param_runs[k].get("accuracy", 0) for k in param_sorted_k]

        (param_line,) = ax_right.plot(
            param_sorted_k,
            param_y,
            color=color,
            marker=marker,
            markersize=5,
            linewidth=2,
            linestyle="--",
            zorder=3,
            alpha=0.85,
        )
        handles.append(param_line)
        labels.append(f"{nolima_label} Parametric Accuracy")

        # Parametric baseline
        if param_baseline is not None and "accuracy" in param_baseline:
            ax_right.axhline(
                param_baseline["accuracy"],
                color=color,
                linestyle=":",
                linewidth=1.0,
                alpha=0.3,
                zorder=1,
            )

    # X-axis
    all_k = sorted({k for _, runs, _ in nolima_data for k in runs})
    ax_left.set_xticks(all_k)
    ax_left.set_xticklabels([str(int(k)) for k in all_k])
    ax_left.set_xlabel("Number of Ablated Heads ($k$)")

    # Y-axes
    ax_left.set_ylabel("NoLiMa ROUGE-L")
    ax_right.set_ylabel("Parametric/Arithmetic Accuracy")
    ax_left.set_ylim(bottom=-0.02)
    ax_right.set_ylim(bottom=0, top=1.05)

    fig.tight_layout()
    save_figure(fig, out_path)

    # Legend
    legend_path = out_path.with_name(out_path.stem + "_legend" + out_path.suffix)
    save_legend(handles, labels, legend_path, ncol=min(len(labels), 2))

    print(f"Saved figure: {out_path}")
    print(f"Saved legend: {legend_path}")

    _print_comparison_table(nolima_data, parametric_data)


def make_per_source_plot(
    nolima_data: list[tuple[str, dict[float, float], float | None]],
    parametric_data: list[tuple[str, dict[float, dict[str, float]], float | None]],
    out_path: Path,
) -> None:
    """Create a per-source breakdown plot with subplots.

    One subplot per model showing NoLiMa + each parametric source.
    """
    setup_plot_style()

    n_models = len(nolima_data)
    fig, axes = plt.subplots(1, n_models, figsize=(FIGURE_SIZE[0] * n_models, FIGURE_SIZE[1]), squeeze=False)

    source_colors = {
        "nolima": "#C44E52",
        "city_country": "#4C72B0",
        "popqa": "#55A868",
        "arithmetic": "#8172B2",
    }
    source_markers = {
        "nolima": "o",
        "city_country": "s",
        "popqa": "D",
        "arithmetic": "^",
    }
    source_labels = {
        "nolima": "NoLiMa ROUGE-L",
        "city_country": "City-Country",
        "popqa": "PopQA",
        "arithmetic": "Arithmetic",
    }

    all_handles = []
    all_labels = []

    for idx, ((_model_label, nolima_runs, nolima_baseline), (_, param_runs, param_baseline)) in enumerate(
        zip(nolima_data, parametric_data)
    ):
        ax = axes[0, idx]

        # NoLiMa
        sorted_k = sorted(nolima_runs.keys())
        nolima_y = [nolima_runs[k] for k in sorted_k]
        (line,) = ax.plot(
            sorted_k,
            nolima_y,
            color=source_colors["nolima"],
            marker=source_markers["nolima"],
            markersize=6,
            linewidth=2.5,
            zorder=4,
        )
        if idx == 0:
            all_handles.append(line)
            all_labels.append(source_labels["nolima"])

        if nolima_baseline is not None:
            ax.axhline(nolima_baseline, color=source_colors["nolima"], linestyle="--", linewidth=1, alpha=0.4)

        # Per-source parametric accuracy
        param_sorted_k = sorted(param_runs.keys())
        for source in ["city_country", "popqa", "arithmetic"]:
            acc_key = f"{source}_accuracy"
            y_vals = [param_runs[k].get(acc_key, 0) for k in param_sorted_k]
            (line,) = ax.plot(
                param_sorted_k,
                y_vals,
                color=source_colors[source],
                marker=source_markers[source],
                markersize=5,
                linewidth=2,
                linestyle="--",
                zorder=3,
            )
            if idx == 0:
                all_handles.append(line)
                all_labels.append(source_labels[source])

            # Baseline
            if param_baseline and acc_key in param_baseline:
                ax.axhline(
                    param_baseline[acc_key], color=source_colors[source], linestyle=":", linewidth=0.8, alpha=0.3
                )

        ax.set_xlabel("Number of Ablated Heads ($k$)")
        if idx == 0:
            ax.set_ylabel("Score")
        ax.set_ylim(bottom=-0.02, top=1.05)

        all_k = sorted(nolima_runs.keys())
        ax.set_xticks(all_k)
        ax.set_xticklabels([str(int(k)) for k in all_k])

    fig.tight_layout()
    save_figure(fig, out_path)

    legend_path = out_path.with_name(out_path.stem + "_legend" + out_path.suffix)
    save_legend(all_handles, all_labels, legend_path, ncol=min(len(all_labels), 4))

    print(f"Saved figure: {out_path}")
    print(f"Saved legend: {legend_path}")


def make_grouped_bar_plot(
    parametric_data: list[tuple[str, dict[float, dict[str, float]], float | None]],
    out_path: Path,
    nolima_data: list[tuple[str, dict[float, float], float | None]] | None = None,
    k_values: list[int] | None = None,
) -> None:
    """Create grouped bar chart: one subplot per model, bars grouped by source at each k.

    Baseline performance shown as dotted horizontal lines in the background.

    Args:
        parametric_data: [(label, {k: {metric: val}}, baseline_metrics), ...]
        out_path: Output SVG path.
        nolima_data: Optional [(label, {k: rouge_l}, baseline_rouge_l), ...].
            When None, only parametric sources are plotted.
        k_values: Specific k values to plot. If None, uses all available k values.
    """
    setup_plot_style()

    include_nolima = nolima_data is not None
    n_models = len(parametric_data)
    sources = (["nolima"] if include_nolima else []) + ["city_country", "popqa", "arithmetic"]
    n_sources = len(sources)

    fig, axes = plt.subplots(1, n_models, figsize=(FIGURE_SIZE[0] * n_models, FIGURE_SIZE[1]), squeeze=False)

    all_handles = []
    all_labels = []

    for idx, (param_label, param_runs, param_baseline) in enumerate(parametric_data):
        ax = axes[0, idx]

        nolima_runs: dict[float, float] = {}
        nolima_baseline: float | None = None
        if include_nolima:
            _, nolima_runs, nolima_baseline = nolima_data[idx]

        # Determine k values to plot
        available_k = sorted(param_runs.keys())
        if include_nolima:
            available_k = sorted(set(available_k) & set(nolima_runs.keys()))
        if k_values is not None:
            ks = [k for k in k_values if k in available_k]
        else:
            ks = available_k
        assert len(ks) > 0, f"No matching k values for {param_label}"

        n_k = len(ks)
        x = np.arange(n_k)
        bar_width = 0.8 / n_sources
        offsets = np.arange(n_sources) - (n_sources - 1) / 2

        for j, source in enumerate(sources):
            color = SOURCE_COLORS[source]
            positions = x + offsets[j] * bar_width

            # Get values for each k
            if source == "nolima":
                values = [nolima_runs.get(k, 0) for k in ks]
                baseline_val = nolima_baseline
            else:
                acc_key = f"{source}_accuracy"
                values = [param_runs.get(k, {}).get(acc_key, 0) for k in ks]
                baseline_val = param_baseline.get(acc_key) if param_baseline else None

            bars = ax.bar(
                positions,
                values,
                width=bar_width,
                color=color,
                edgecolor="black",
                linewidth=LINE_WIDTH * 0.5,
                zorder=3,
            )

            # Baseline as dotted horizontal line
            if baseline_val is not None:
                ax.axhline(
                    baseline_val,
                    color=color,
                    linestyle=":",
                    linewidth=LINE_WIDTH * 0.75,
                    alpha=0.6,
                    zorder=2,
                )

            if idx == 0:
                all_handles.append(bars[0])
                all_labels.append(SOURCE_LABELS[source])

        ax.set_xticks(x)
        ax.set_xticklabels([str(int(k)) for k in ks])
        ax.set_xlabel("Number of Ablated Heads ($k$)")
        if idx == 0:
            ax.set_ylabel("Accuracy")
        ax.set_ylim(bottom=0, top=1.05)

        if n_models > 1:
            ax.annotate(
                param_label,
                xy=(0.5, 0.97),
                xycoords="axes fraction",
                ha="center",
                va="top",
                fontsize=10,
                fontstyle="italic",
            )

    fig.tight_layout()
    save_figure(fig, out_path)

    legend_path = out_path.with_name(out_path.stem + "_legend" + out_path.suffix)
    save_legend(all_handles, all_labels, legend_path, ncol=n_sources)

    print(f"Saved figure: {out_path}")
    print(f"Saved legend: {legend_path}")


def _print_comparison_table(
    nolima_data: list[tuple[str, dict[float, float], float | None]],
    parametric_data: list[tuple[str, dict[float, dict[str, float]], float | None]],
) -> None:
    """Print side-by-side results table to console."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    for (model_label, nolima_runs, nolima_baseline), (_, param_runs, param_baseline) in zip(
        nolima_data, parametric_data
    ):
        table = Table(title=f"{model_label} — Retrieval vs Parametric Ablation")

        table.add_column("k", justify="right", style="bold")
        table.add_column("Heads", justify="right")
        table.add_column("NoLiMa ROUGE-L", justify="right")
        table.add_column("Param. Acc", justify="right")
        table.add_column("City/Country", justify="right")
        table.add_column("PopQA", justify="right")
        table.add_column("Arithmetic", justify="right")

        # Baseline row
        if nolima_baseline is not None or param_baseline is not None:
            nolima_bl = f"{nolima_baseline:.4f}" if nolima_baseline is not None else "—"
            param_bl = f"{param_baseline.get('accuracy', 0):.4f}" if param_baseline else "—"
            cc_bl = f"{param_baseline.get('city_country_accuracy', 0):.4f}" if param_baseline else "—"
            pq_bl = f"{param_baseline.get('popqa_accuracy', 0):.4f}" if param_baseline else "—"
            ar_bl = f"{param_baseline.get('arithmetic_accuracy', 0):.4f}" if param_baseline else "—"
            table.add_row("Baseline", "0", nolima_bl, param_bl, cc_bl, pq_bl, ar_bl)

        # Ablation rows
        all_k = sorted(set(nolima_runs.keys()) | set(param_runs.keys()))
        for k in all_k:
            nolima_val = f"{nolima_runs[k]:.4f}" if k in nolima_runs else "—"
            if k in param_runs:
                param_val = f"{param_runs[k].get('accuracy', 0):.4f}"
                cc_val = f"{param_runs[k].get('city_country_accuracy', 0):.4f}"
                pq_val = f"{param_runs[k].get('popqa_accuracy', 0):.4f}"
                ar_val = f"{param_runs[k].get('arithmetic_accuracy', 0):.4f}"
            else:
                param_val = cc_val = pq_val = ar_val = "—"

            table.add_row(str(int(k)), "—", nolima_val, param_val, cc_val, pq_val, ar_val)

        console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot parametric/arithmetic ablation vs NoLiMa retrieval ablation",
    )
    parser.add_argument(
        "--nolima",
        nargs="+",
        type=Path,
        default=None,
        help="NoLiMa ablation cache JSON file(s) (optional, same order as --parametric)",
    )
    parser.add_argument(
        "--parametric",
        nargs="+",
        type=Path,
        required=True,
        help="Parametric ablation cache JSON file(s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output SVG path (default: figures/parametric_vs_retrieval.svg)",
    )
    parser.add_argument(
        "--per-source",
        action="store_true",
        help="Also generate per-source breakdown plot",
    )
    parser.add_argument(
        "--bar",
        action="store_true",
        help="Generate grouped bar chart instead of line plots",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=None,
        help="Specific k values for bar chart (default: all available)",
    )
    args = parser.parse_args()

    if args.nolima is not None:
        assert len(args.nolima) == len(args.parametric), (
            f"Must provide same number of --nolima and --parametric files "
            f"(got {len(args.nolima)} vs {len(args.parametric)})"
        )

    nolima_data: list[tuple[str, dict[float, float], float | None]] | None = None
    parametric_data = []

    if args.nolima is not None:
        nolima_data = []
        for nolima_path, param_path in zip(args.nolima, args.parametric):
            nolima_cache = load_cache(nolima_path)
            param_cache = load_cache(param_path)

            label = model_label_from_path(nolima_path, "nolima_ablation")
            nolima_runs, nolima_baseline = extract_nolima_runs(nolima_cache)
            param_runs, param_baseline = extract_parametric_runs(param_cache)

            assert len(nolima_runs) > 0, f"No ablation runs found in {nolima_path}"
            assert len(param_runs) > 0, f"No ablation runs found in {param_path}"

            nolima_data.append((label, nolima_runs, nolima_baseline))
            parametric_data.append((label, param_runs, param_baseline))
    else:
        for param_path in args.parametric:
            param_cache = load_cache(param_path)
            label = model_label_from_path(param_path, "parametric_ablation")
            param_runs, param_baseline = extract_parametric_runs(param_cache)
            assert len(param_runs) > 0, f"No ablation runs found in {param_path}"
            parametric_data.append((label, param_runs, param_baseline))

    out_path = args.out or Path("figures/parametric_vs_retrieval.svg")

    if args.bar:
        bar_path = out_path.with_name(out_path.stem + "_bar" + out_path.suffix) if not args.out else out_path
        make_grouped_bar_plot(parametric_data, bar_path, nolima_data=nolima_data, k_values=args.k_values)
    else:
        assert nolima_data is not None, "Line plots (dissociation/per-source) require --nolima"
        make_dissociation_plot(nolima_data, parametric_data, out_path)

    if args.per_source:
        assert nolima_data is not None, "--per-source requires --nolima"
        per_source_path = out_path.with_name(out_path.stem + "_per_source" + out_path.suffix)
        make_per_source_plot(nolima_data, parametric_data, per_source_path)


if __name__ == "__main__":
    main()
