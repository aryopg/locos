#!/usr/bin/env python3
"""Plot NoLiMa ablation results: performance vs head selection.

Reads cached results from ``run_nolima_ablation.py`` and produces a line plot
with x-axis = k or threshold, y-axis = ROUGE-L.

Usage:
    # Plot from cache file
    python locos/plot_nolima_ablation.py \
        ablation_results/nolima_ablation_Qwen3-8B_nolima.json

    # Multiple models
    python locos/plot_nolima_ablation.py \
        ablation_results/nolima_ablation_Qwen3-8B_nolima.json \
        ablation_results/nolima_ablation_Qwen3-14B_nolima.json

    # Custom output
    python locos/plot_nolima_ablation.py \
        ablation_results/nolima_ablation_Qwen3-8B_nolima.json \
        --out figures/ablation_custom.svg

    # Plot ROUGE-1 instead
    python locos/plot_nolima_ablation.py \
        ablation_results/nolima_ablation_Qwen3-8B_nolima.json \
        --metric rouge_1_mean
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from locos_eval.utils.plotting import (
    FIGURE_SIZE,
    save_figure,
    save_legend,
    setup_plot_style,
)

COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD"]
MARKERS = ["o", "s", "D", "^", "v", "P"]


METRIC_LABELS = {
    "rouge_l_mean": "ROUGE-L",
    "rouge_1_mean": "ROUGE-1",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_ablation_cache(cache_path: Path) -> dict:
    """Load ablation results cache."""
    assert cache_path.exists(), f"Cache file not found: {cache_path}"
    with open(cache_path) as f:
        return json.load(f)


def extract_runs(
    cache: dict,
) -> tuple[str, dict[float, dict], float | None]:
    """Parse cache into structured run data.

    Returns:
        (mode, {value: metrics_dict}, baseline_score_or_None)
    """
    baseline_score = None
    runs: dict[float, dict] = {}
    mode = None

    for _, metrics in cache.items():
        m = metrics.get("mode", "")
        if m in ("greedy", "baseline"):
            baseline_score = metrics
            continue

        if mode is None:
            mode = m
        value = metrics["value"]
        runs[value] = metrics

    return mode or "top-k", runs, baseline_score


def model_label(cache_path: Path) -> str:
    """Derive display label from cache filename."""
    name = cache_path.stem.removeprefix("nolima_ablation_").removeprefix("niah_ablation_")
    return name


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_ablation_plot(
    all_data: dict[str, tuple[str, dict[float, dict], float | None]],
    metric: str,
    out_path: Path,
) -> None:
    """Create line plot of metric vs head selection value.

    Args:
        all_data: {label: (mode, {value: metrics}, greedy_or_None)}
        metric: Metric key to plot (e.g. "rouge_l_mean").
        out_path: Output SVG path.
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    handles = []
    labels = []

    for i, (label, (_, runs, greedy)) in enumerate(all_data.items()):
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        # Sort by value
        sorted_values = sorted(runs.keys())
        x = sorted_values
        y = [runs[v][metric] for v in sorted_values]

        (line,) = ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2,
            zorder=3,
        )
        handles.append(line)
        labels.append(label)

        # Add baseline as horizontal dashed line
        if greedy is not None and metric in greedy:
            ax.axhline(
                greedy[metric],
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
                zorder=2,
            )
            # Add a second entry in the legend for the baseline
            baseline_line = plt.Line2D(
                [0],
                [0],
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
            )
            handles.append(baseline_line)
            labels.append(f"{label} (baseline)")

    # Determine x-axis label from mode
    mode = next(iter(all_data.values()))[0]
    if mode == "top-k":
        ax.set_xlabel("Number of Masked Heads ($k$)")
        # Integer x-ticks
        all_x = sorted({v for _, (_, runs, _) in all_data.items() for v in runs})
        ax.set_xticks(all_x)
        ax.set_xticklabels([str(int(v)) for v in all_x])
    else:
        ax.set_xlabel("Score Threshold")

    metric_label = METRIC_LABELS.get(metric, metric)
    ax.set_ylabel(metric_label)

    fig.tight_layout()
    save_figure(fig, out_path)

    # Legend
    legend_path = out_path.with_name(out_path.stem + "_legend" + out_path.suffix)
    save_legend(handles, labels, legend_path, ncol=min(len(labels), 3))

    print(f"Saved figure: {out_path}")
    print(f"Saved legend: {legend_path}")

    _print_results(all_data, metric)


def _print_results(
    all_data: dict[str, tuple[str, dict[float, dict], float | None]],
    metric: str,
) -> None:
    """Print results table to console."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    for label, (mode, runs, greedy) in all_data.items():
        metric_label = METRIC_LABELS.get(metric, metric)
        table = Table(title=f"{label} — NoLiMa Ablation")

        table.add_column("Value", justify="right", style="bold")
        table.add_column("Heads", justify="right")
        table.add_column(metric_label, justify="right")
        if greedy:
            table.add_column(r"Delta vs Greedy", justify="right")

        if greedy and metric in greedy:
            table.add_row(
                "Greedy",
                "—",
                f"{greedy[metric]:.4f}",
                *([" "] if greedy else []),
            )

        baseline = greedy[metric] if greedy and metric in greedy else None

        for v in sorted(runs.keys()):
            m = runs[v]
            val_display = str(int(v)) if mode == "top-k" else f"{v:.4f}"
            score = m[metric]
            row = [val_display, str(m["n_heads"]), f"{score:.4f}"]

            if baseline is not None:
                delta = score - baseline
                sign = "+" if delta > 0 else ""
                color = "green" if delta > 0 else "red" if delta < 0 else ""
                row.append(f"[{color}]{sign}{delta:.4f}[/{color}]")

            table.add_row(*row)

        console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Plot NoLiMa ablation results from cached data")
    parser.add_argument(
        "cache_files",
        nargs="+",
        type=Path,
        help="One or more ablation cache JSON files",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="rouge_l_mean",
        choices=["rouge_l_mean", "rouge_1_mean"],
        help="Metric to plot (default: rouge_l_mean)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output SVG path (default: figures/nolima_ablation_{label}.svg)",
    )
    args = parser.parse_args()

    all_data: dict[str, tuple[str, dict[float, dict], float | None]] = {}
    for path in args.cache_files:
        label = model_label(path)
        cache = load_ablation_cache(path)
        mode, runs, greedy = extract_runs(cache)
        assert len(runs) > 0, f"No ablation runs found in {path}"
        all_data[label] = (mode, runs, greedy)

    dataset = "niah" if args.cache_files[0].stem.startswith("niah_ablation_") else "nolima"

    if args.out is not None:
        out_path = args.out
    elif len(args.cache_files) == 1:
        label = model_label(args.cache_files[0])
        out_path = Path(f"figures/{dataset}_ablation_{label}.svg")
    else:
        out_path = Path(f"figures/{dataset}_ablation_multi.svg")

    make_ablation_plot(all_data, args.metric, out_path)


if __name__ == "__main__":
    main()
