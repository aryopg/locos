#!/usr/bin/env python3
"""Plot top-k and bottom-k heads by mean S score as a split bar chart.

Shows the highest-scoring heads on the left and lowest-scoring heads on the
right, with an ellipsis in the middle indicating truncated middle ranks.

Usage:
    python -m locos.plotting.score_tails \
        /path/to/model_logit_contrib_nolima.json

    # Custom k
    python -m locos.plotting.score_tails \
        /path/to/model_logit_contrib_nolima.json --k 30

    # Multiple models
    python -m locos.plotting.score_tails \
        /path/to/modelA.json /path/to/modelB.json --k 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from locos.utils.common import load_head_scores
from locos_eval.utils.plotting import (
    LINE_WIDTH,
    MODEL_PRETTY_NAMES,
    save_figure,
    setup_plot_style,
)

COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD"]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_tails_plot(
    scores: dict[str, float],
    k: int,
    model_label: str,
    out_path: Path,
    model_name: str | None = None,
) -> None:
    """Create a split bar chart showing top-k and bottom-k heads."""
    setup_plot_style()

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_k = sorted_items[:k]
    bottom_k = sorted_items[-k:]

    # Show the two tails with a compact gap for omitted middle ranks.
    gap = 1.6

    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)

    # Top-k bars (left side, descending)
    top_labels = [item[0] for item in top_k]
    top_vals = [item[1] for item in top_k]
    x_top = np.arange(len(top_k))

    # Bottom-k bars continue the same descending score order after the break.
    bottom_labels = [item[0] for item in bottom_k]
    bottom_vals = [item[1] for item in bottom_k]
    x_bottom = np.arange(len(bottom_k)) + len(top_k) + gap

    # Color: negative = red, positive = blue
    top_colors = ["#C44E52" if v < 0 else "#4C72B0" for v in top_vals]
    bottom_colors = ["#C44E52" if v < 0 else "#4C72B0" for v in bottom_vals]

    bar_width = 0.8
    ax.bar(x_top, top_vals, width=bar_width, color=top_colors, edgecolor="black", linewidth=0.5, zorder=3)
    ax.bar(x_bottom, bottom_vals, width=bar_width, color=bottom_colors, edgecolor="black", linewidth=0.5, zorder=3)

    # Zero line, broken across the omitted middle ranks.
    left_zero_start = x_top[0] - bar_width / 2
    left_zero_end = x_top[-1] + bar_width / 2
    right_zero_start = x_bottom[0] - bar_width / 2
    right_zero_end = x_bottom[-1] + bar_width / 2
    ax.plot([left_zero_start, left_zero_end], [0, 0], color="black", linewidth=LINE_WIDTH * 0.5, zorder=2)
    ax.plot([right_zero_start, right_zero_end], [0, 0], color="black", linewidth=LINE_WIDTH * 0.5, zorder=2)

    y_lo, y_hi = ax.get_ylim()
    gap_center = len(top_k) + gap / 2 - 0.5
    ax.text(
        gap_center,
        0,
        r"$\cdots$",
        ha="center",
        va="center",
        fontsize=12,
        color="black",
        zorder=5,
    )
    # ax.text(
    #     gap_center,
    #     -0.18,
    #     f"{n_omitted} omitted",
    #     ha="center",
    #     va="top",
    #     fontsize=8,
    #     color="dimgray",
    #     transform=ax.get_xaxis_transform(),
    # )

    # X-axis: show only the visible endpoints of each tail plus the center ellipsis.
    tick_pos = [x_top[0], x_top[-1], gap_center, x_bottom[0], x_bottom[-1]]
    tick_labels = [top_labels[0], top_labels[-1], r"$\cdots$", bottom_labels[0], bottom_labels[-1]]
    ax.set_xticks(tick_pos)
    xticklabels = ax.set_xticklabels(tick_labels, fontsize=6)
    for idx, label in enumerate(xticklabels):
        label.set_rotation(0 if tick_labels[idx] == r"$\cdots$" else 90)

    ax.text(
        np.mean(x_top),
        y_hi - (y_hi - y_lo) * 0.04,
        f"Top-{k}",
        ha="center",
        va="top",
        fontsize=9,
        fontstyle="italic",
        color="dimgray",
        zorder=5,
    )
    ax.text(
        np.mean(x_bottom),
        y_hi - (y_hi - y_lo) * 0.04,
        f"Bottom-{k}",
        ha="center",
        va="top",
        fontsize=9,
        fontstyle="italic",
        color="dimgray",
        zorder=5,
    )

    ax.set_xlabel(r"Layer-Head")
    ax.set_ylabel(r"Mean $S$ score")
    ax.set_xlim(-0.8, x_bottom[-1] + 0.8)

    keep_title = False
    if model_name:
        ax.set_title(MODEL_PRETTY_NAMES.get(model_name, model_name))
        keep_title = True

    fig.subplots_adjust(bottom=0.28)
    fig.tight_layout()
    save_figure(fig, out_path, keep_title=keep_title)

    # Print summary
    _print_summary(bottom_k, top_k, k, model_label)


def _print_summary(
    bottom_k: list[tuple[str, float]],
    top_k: list[tuple[str, float]],
    k: int,
    model_label: str,
) -> None:
    """Print summary statistics to console."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    bottom_vals = [v for _, v in bottom_k]
    top_vals = [v for _, v in top_k]
    neg_in_bottom = sum(1 for _, v in bottom_k if v < 0)
    pos_in_top = sum(1 for _, v in top_k if v > 0)

    table = Table(title=f"{model_label} — Score Tails (k={k})")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row(f"Bottom-{k} negative", f"{neg_in_bottom}/{k}")
    table.add_row(f"Bottom-{k} min", f"{min(bottom_vals):+.6f}")
    table.add_row(f"Bottom-{k} max", f"{max(bottom_vals):+.6f}")
    table.add_section()
    table.add_row(f"Top-{k} positive", f"{pos_in_top}/{k}")
    table.add_row(f"Top-{k} min", f"{min(top_vals):+.6f}")
    table.add_row(f"Top-{k} max", f"{max(top_vals):+.6f}")

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Plot top-k and bottom-k heads by mean S score")
    parser.add_argument(
        "heads_json",
        nargs="+",
        type=Path,
        help="One or more retrieval heads JSON files",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="Number of heads to show on each tail (default: 50)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output SVG path (default: figures/score_tails_{name}.svg)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for figure title (looked up in MODEL_PRETTY_NAMES)",
    )
    args = parser.parse_args()

    for path in args.heads_json:
        assert path.exists(), f"File not found: {path}"

        scores = load_head_scores(path)
        total_heads = len(scores)
        assert args.k <= total_heads // 2, f"k={args.k} too large for {total_heads} heads"

        label = path.stem
        if args.out is not None:
            out_path = args.out
        else:
            out_path = Path(f"figures/score_tails_{label}.svg")

        make_tails_plot(scores, args.k, label, out_path, model_name=args.model)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
