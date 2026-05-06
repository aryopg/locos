#!/usr/bin/env python3
"""Plot retrieval score distribution: mean score per layer-head pair.

X-axis shows layer-head identifiers sorted by descending score,
Y-axis shows the mean retrieval score.  Use ``--top-k`` to restrict
to the top-k heads only.

Usage:
    # All heads for a single model
    python locos/plot_retrieval_score_dist.py \
        retrieval_heads/Qwen3-8B_nolima.json

    # Top-20 heads only
    python locos/plot_retrieval_score_dist.py \
        retrieval_heads/Qwen3-8B_nolima.json --top-k 20

    # Multiple models overlaid
    python locos/plot_retrieval_score_dist.py \
        retrieval_heads/Qwen3-8B_nolima.json \
        retrieval_heads/Qwen3-14B_nolima.json \
        --top-k 50

    # Custom output path
    python locos/plot_retrieval_score_dist.py \
        retrieval_heads/Qwen3-8B_nolima.json --out figures/dist.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from locos_eval.utils.plotting import (
    save_figure,
    save_legend,
    setup_plot_style,
)

COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

from locos.utils.common import load_head_scores


def model_label(path: Path) -> str:
    """Derive a short display label from a retrieval heads filename."""
    return path.stem  # e.g. "Qwen3-8B_nolima"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_dist_plot(
    all_scores: dict[str, dict[str, float]],
    top_k: int | None,
    out_path: Path,
) -> None:
    """Create retrieval score distribution plot.

    Args:
        all_scores: {model_label: {layer-head: mean_score}}
        top_k: If set, show only top-k heads per model.
        out_path: Output SVG path.
    """
    setup_plot_style()

    single_model = len(all_scores) == 1

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    handles = []
    labels = []

    for i, (label, scores) in enumerate(all_scores.items()):
        # Sort by descending score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if top_k is not None:
            sorted_items = sorted_items[:top_k]

        head_labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        x = np.arange(len(values))

        color = COLORS[i % len(COLORS)]

        if single_model:
            # Bar plot for single model — more readable
            bars = ax.bar(
                x,
                values,
                width=0.8,
                color=color,
                edgecolor="black",
                linewidth=1,
                zorder=3,
            )
            handles.append(bars)
            # Show head labels on x-axis (sparse for readability)
            n_ticks = min(len(head_labels), 30)
            tick_step = max(1, len(head_labels) // n_ticks)
            tick_positions = list(range(0, len(head_labels), tick_step))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                [head_labels[j] for j in tick_positions],
                rotation=90,
                fontsize=7,
            )
        else:
            # Line plot for multiple models — overlay comparison
            (line,) = ax.plot(
                x,
                values,
                color=color,
                linewidth=1,
                zorder=3,
                marker="o" if (top_k is not None and top_k <= 30) else None,
                markersize=4,
            )
            handles.append(line)

        labels.append(label)

    ax.set_xlabel("Layer-Head (ranked by score)")
    ax.set_ylabel("Mean Retrieval Score")
    ax.set_xlim(-0.5, max(1, len(sorted_items) - 0.5))

    if not single_model:
        ax.set_xticks([])

    fig.tight_layout()
    save_figure(fig, out_path)

    # Save legend
    legend_handles = [h[0] if hasattr(h, "__getitem__") else h for h in handles]
    legend_path = out_path.with_name(out_path.stem + "_legend" + out_path.suffix)
    save_legend(legend_handles, labels, legend_path, ncol=min(len(labels), 3))

    _print_table(all_scores, top_k)


def _print_table(
    all_scores: dict[str, dict[str, float]],
    top_k: int | None,
) -> None:
    """Print a summary table to the console."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    for label, scores in all_scores.items():
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        display_k = top_k or 20
        top_items = sorted_items[:display_k]

        table = Table(title=f"{label} — Top-{display_k} Retrieval Heads")
        table.add_column("Rank", justify="right", style="dim")
        table.add_column("Layer-Head", style="bold")
        table.add_column("Mean Score", justify="right")

        for rank, (key, score) in enumerate(top_items, 1):
            table.add_row(str(rank), key, f"{score:.4f}")

        # Summary stats
        all_vals = [v for v in scores.values()]
        nonzero = [v for v in all_vals if v > 0]
        table.add_section()
        table.add_row("", "[dim]Total heads[/dim]", f"[dim]{len(all_vals)}[/dim]")
        table.add_row("", "[dim]Non-zero heads[/dim]", f"[dim]{len(nonzero)}[/dim]")
        if nonzero:
            table.add_row("", "[dim]Max score[/dim]", f"[dim]{max(nonzero):.4f}[/dim]")
            table.add_row("", "[dim]Median (non-zero)[/dim]", f"[dim]{sorted(nonzero)[len(nonzero)//2]:.4f}[/dim]")

        console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Plot retrieval score distribution per layer-head pair")
    parser.add_argument(
        "heads_json",
        nargs="+",
        type=Path,
        help="One or more retrieval heads JSON files",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Show only the top-k heads by mean score (default: 20)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output SVG path (default: figures/retrieval_score_dist_{name}.svg)",
    )
    args = parser.parse_args()

    all_scores: dict[str, dict[str, float]] = {}
    for path in args.heads_json:
        assert path.exists(), f"File not found: {path}"
        label = model_label(path)
        all_scores[label] = load_head_scores(path)

    if args.out is not None:
        out_path = args.out
    elif len(args.heads_json) == 1:
        name = model_label(args.heads_json[0])
        suffix = f"_top{args.top_k}" if args.top_k is not None else ""
        out_path = Path(f"figures/retrieval_score_dist_{name}{suffix}.svg")
    else:
        suffix = f"_top{args.top_k}" if args.top_k is not None else ""
        out_path = Path(f"figures/retrieval_score_dist_multi{suffix}.svg")

    make_dist_plot(all_scores, args.top_k, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
