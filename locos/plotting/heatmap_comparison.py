"""Compare retrieval head heatmaps across detection methods.

Accepts any number of heatmaps via CLI, auto-detects JSON format
(flat Wu/behavioral or envelope with "scores" key), and infers grid
dimensions from the data.

Example usage:
    python -m locos.plotting.heatmap_comparison \
        --heatmap "Wu NiaH" retrieval_heads/Llama3-8B_nolima.json \
        --heatmap "Logit Contrib" retrieval_heads/Llama3-8B_logit_contrib_nolima.json \
        --model Meta-Llama-3-8B-Instruct \
        --top-k 10 --output figures/heatmap_comparison.svg
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from locos_eval.utils.plotting import LINE_WIDTH, MODEL_PRETTY_NAMES, save_figure, setup_plot_style

DEFAULT_TOP_K = 10
MODEL_TITLE_FONT_SIZE = 18
METHOD_TITLE_FONT_SIZE = 12
MODEL_TITLE_Y = 0.95
METHOD_TITLE_Y = 1.005
FIGURE_HEIGHT = 3.0
LEFT_MARGIN = 0.75
RIGHT_MARGIN = 0.2
BOTTOM_MARGIN = 0.6
TOP_MARGIN = 0.55
PANEL_GAP = 0.6
KDE_SIZE = 0.5
KDE_PAD = 0.05
CBAR_SIZE = 0.12
CBAR_PAD = 0.03


def _parse_scores_dict(scores_dict: dict) -> tuple[np.ndarray, int, int]:
    """Convert {"layer-head": [values...]} dict to a 2-D grid.

    Returns (grid, num_layers, num_heads) with dimensions inferred from keys.
    """
    max_layer, max_head = 0, 0
    for key in scores_dict:
        layer, head = map(int, key.split("-"))
        max_layer = max(max_layer, layer)
        max_head = max(max_head, head)
    num_layers = max_layer + 1
    num_heads = max_head + 1

    grid = np.zeros((num_layers, num_heads))
    for key, vals in scores_dict.items():
        layer, head = map(int, key.split("-"))
        grid[layer, head] = np.mean(vals) if vals else 0.0
    return grid, num_layers, num_heads


def load_scores(path: str) -> tuple[np.ndarray, int, int]:
    """Load scores from any detector JSON format (auto-detected).

    Supports:
    - Flat (Wu/behavioral): {"layer-head": [scores...]}
    - Envelope (logit_contrib, contrastive, CRI): {"meta": ..., "scores": {"layer-head": [scores...]}}

    Returns (grid, num_layers, num_heads).
    """
    with open(path) as f:
        data = json.load(f)
    if "scores" in data and isinstance(data["scores"], dict):
        return _parse_scores_dict(data["scores"])
    return _parse_scores_dict(data)


def top_k_indices(grid: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Return (layer, head) of the top-k cells by value."""
    num_heads = grid.shape[1]
    flat = grid.flatten()
    top_idx = np.argsort(flat)[-k:]
    return [divmod(int(i), num_heads) for i in top_idx]


def draw_bounding_boxes(ax, top_cells, color="#BE1E2D"):
    """Draw a rectangle around each top-k cell on a heatmap axis."""
    for layer, head in top_cells:
        rect = mpatches.FancyBboxPatch(
            (head - 0.5, layer - 0.5),
            1,
            1,
            boxstyle="square,pad=0",
            linewidth=LINE_WIDTH,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare retrieval head heatmaps across detection methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--heatmap",
        nargs=2,
        action="append",
        metavar=("TITLE", "PATH"),
        required=True,
        help='Subplot title and JSON path (repeat for each heatmap). E.g. --heatmap "Wu NiaH" file.json',
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model name (used in output filename if --output omitted)"
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help=f"Number of top heads to highlight (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument("--output", type=str, default=None, help="Output path (default: figures/heatmap_<model>.svg)")
    return parser


def build_comparison_figure(
    heatmaps: list[tuple[str, np.ndarray]],
    model_name: str | None = None,
    top_k: int = DEFAULT_TOP_K,
):
    """Build the comparison figure and return the Matplotlib artists."""
    setup_plot_style()

    n = len(heatmaps)
    assert n >= 1, "Need at least one --heatmap"

    top_cells_per = [top_k_indices(grid, top_k) for _, grid in heatmaps]

    heatmap_height = FIGURE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN
    heatmap_widths = [heatmap_height * (grid.shape[1] / grid.shape[0]) for _, grid in heatmaps]
    panel_widths = [width + KDE_PAD + KDE_SIZE + CBAR_PAD + CBAR_SIZE for width in heatmap_widths]
    figure_width = LEFT_MARGIN + RIGHT_MARGIN + sum(panel_widths) + PANEL_GAP * max(0, n - 1)

    fig = plt.figure(figsize=(figure_width, FIGURE_HEIGHT))
    artists = {"panels": [], "model_text": None, "top_cells_per": top_cells_per}
    x_cursor = LEFT_MARGIN

    for i, ((title, grid), top_cells) in enumerate(zip(heatmaps, top_cells_per)):
        num_layers = grid.shape[0]
        heatmap_width = heatmap_widths[i]
        ax_hm = fig.add_axes(
            [
                x_cursor / figure_width,
                BOTTOM_MARGIN / FIGURE_HEIGHT,
                heatmap_width / figure_width,
                heatmap_height / FIGURE_HEIGHT,
            ]
        )
        ax_kde = fig.add_axes(
            [
                (x_cursor + heatmap_width + KDE_PAD) / figure_width,
                BOTTOM_MARGIN / FIGURE_HEIGHT,
                KDE_SIZE / figure_width,
                heatmap_height / FIGURE_HEIGHT,
            ],
            sharey=ax_hm,
        )
        ax_cb = fig.add_axes(
            [
                (x_cursor + heatmap_width + KDE_PAD + KDE_SIZE + CBAR_PAD) / figure_width,
                BOTTOM_MARGIN / FIGURE_HEIGHT,
                CBAR_SIZE / figure_width,
                heatmap_height / FIGURE_HEIGHT,
            ]
        )

        # Per-panel symmetric colour scale
        vabs = max(abs(grid.min()), abs(grid.max()))

        ax_hm.set_facecolor("white")
        ax_hm.grid(False)

        im = ax_hm.imshow(
            grid,
            cmap="coolwarm",
            aspect="auto",
            vmin=-vabs,
            vmax=vabs,
            origin="lower",
        )
        draw_bounding_boxes(ax_hm, top_cells)
        ax_hm.set_xlabel("Head")
        ax_hm.tick_params(axis="both", pad=-1.5)
        if i == 0:
            ax_hm.set_ylabel("Layer")

        panel_left = ax_hm.get_position().x0
        panel_right = ax_kde.get_position().x1
        method_y = ax_hm.get_position().y1 + (METHOD_TITLE_Y - 1.0) * (heatmap_height / FIGURE_HEIGHT)

        # Method label above the combined heatmap + KDE panel (not title — save_figure strips titles)
        method_text = fig.text(
            (panel_left + panel_right) / 2,
            method_y,
            title,
            ha="center",
            va="bottom",
            fontsize=METHOD_TITLE_FONT_SIZE,
            fontweight="bold",
        )

        # KDE marginal: layer distribution weighted by positive scores
        layers, weights = [], []
        for layer_idx in range(num_layers):
            for head_idx in range(grid.shape[1]):
                score = grid[layer_idx, head_idx]
                if score > 0:
                    layers.append(layer_idx)
                    weights.append(score)

        ax_kde.set_facecolor("white")
        ax_kde.grid(False)
        ax_kde.set_aspect("auto")  # fill allocated space — match heatmap height exactly
        kde_line = None
        if layers:
            sns.kdeplot(
                y=layers,
                weights=weights,
                ax=ax_kde,
                fill=True,
                color="steelblue",
                alpha=0.5,
                linewidth=0,
                clip=(0, num_layers - 1),
                bw_adjust=0.8,
            )
            sns.kdeplot(
                y=layers,
                weights=weights,
                ax=ax_kde,
                fill=False,
                color="steelblue",
                alpha=1.0,
                linewidth=LINE_WIDTH,
                clip=(0, num_layers - 1),
                bw_adjust=0.8,
            )
            kde_line = ax_kde.lines[-1]
        ax_kde.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
        ax_kde.set_xticks([])
        ax_kde.set_xlabel("")
        ax_kde.set_ylabel("")

        # Per-panel colorbar
        fig.colorbar(im, cax=ax_cb)
        ax_cb.tick_params(
            axis="y", labelleft=False, labelright=True, left=False, right=False, length=0, pad=1, labelsize=8
        )
        artists["panels"].append(
            {"ax_hm": ax_hm, "ax_kde": ax_kde, "ax_cb": ax_cb, "method_text": method_text, "kde_line": kde_line}
        )
        x_cursor += panel_widths[i] + PANEL_GAP

    # Model name as overarching title (fig.text, not suptitle — save_figure strips suptitle)
    if model_name:
        pretty = MODEL_PRETTY_NAMES.get(model_name, model_name)
        artists["model_text"] = fig.text(
            0.5,
            MODEL_TITLE_Y,
            pretty,
            ha="center",
            va="top",
            fontsize=MODEL_TITLE_FONT_SIZE,
            fontweight="bold",
        )

    return fig, artists


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    heatmaps: list[tuple[str, np.ndarray]] = []
    for title, path in args.heatmap:
        grid, _, _ = load_scores(path)
        heatmaps.append((title, grid))

    fig, artists = build_comparison_figure(heatmaps, model_name=args.model, top_k=args.top_k)
    top_cells_per = artists["top_cells_per"]

    # Output path
    if args.output:
        out = Path(args.output)
    else:
        slug = args.model or "comparison"
        out = Path(f"figures/heatmap_{slug}.svg")

    save_figure(fig, out)
    print(f"Saved to {out}")

    # Print overlap info for all pairs
    for i in range(len(heatmaps)):
        title_i, _ = heatmaps[i]
        print(f"{title_i} top-{args.top_k}: {sorted(top_cells_per[i])}")
    if len(heatmaps) >= 2:
        from itertools import combinations

        for i, j in combinations(range(len(heatmaps)), 2):
            title_i, _ = heatmaps[i]
            title_j, _ = heatmaps[j]
            overlap = set(top_cells_per[i]) & set(top_cells_per[j])
            print(f"Overlap ({title_i} ∩ {title_j}): {len(overlap)} heads — {sorted(overlap)}")


if __name__ == "__main__":
    main()
