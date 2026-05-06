"""KV-group x layer heatmap of group-aggregated retrieval scores.

Companion to ``locos.plotting.kv_group_coverage`` (top-k
KV-group coverage curve). This script answers the spatial question:
*which layers do the salient KV groups live in?*

Consumes the envelope JSONs emitted by
``locos.analysis.kv_group_analysis`` (``*__kvgroup.json``),
which already contain per-(layer, kv_group) score lists averaged over the
Q-heads in each group.

Layout: KV-group on y (few, ~8) and layer on x (many, ~36-64), so the
figure spreads horizontally — well-suited to an appendix page. Multiple
``--heatmap`` panels stack vertically with a shared x-axis convention.

Example usage:
    python -m locos.plotting.kv_group_heatmap \\
        --heatmap "Qwen3-8B" analysis_out/kv_group/Qwen3-8B/Qwen3-8B_logit_contrib_nolima__kvgroup.json \\
        --top-k 10 \\
        --output figures/kv_group_heatmap_Qwen3-8B.svg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from locos_eval.utils.plotting import LINE_WIDTH, save_figure, setup_plot_style

DEFAULT_TOP_K = 10
PANEL_TITLE_FONT_SIZE = 12
PANEL_TITLE_Y = 1.02


def load_kvgroup_envelope(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load a ``*__kvgroup.json`` envelope into a (num_layers, num_kv_groups) grid.

    Returns (grid, meta). Cells without an entry are filled with 0.0 (the
    aggregator emits one entry per (layer, kv_group) pair seen in the source
    detection JSON, but missing trial data degrades to an empty list).
    """
    with open(path) as f:
        data = json.load(f)
    assert "scores" in data and isinstance(
        data["scores"], dict
    ), f"{path} is not an envelope JSON; expected a 'scores' dict produced by kv_group_analysis."
    meta = data.get("meta", {})
    scores = data["scores"]

    max_layer, max_group = 0, 0
    for key in scores:
        layer, group = map(int, key.split("-"))
        max_layer = max(max_layer, layer)
        max_group = max(max_group, group)

    num_layers = max_layer + 1
    num_kv_groups = max_group + 1
    if "num_kv_heads" in meta:
        num_kv_groups = max(num_kv_groups, int(meta["num_kv_heads"]))

    grid = np.zeros((num_layers, num_kv_groups), dtype=float)
    for key, vals in scores.items():
        layer, group = map(int, key.split("-"))
        grid[layer, group] = float(np.mean(vals)) if vals else 0.0

    return grid, meta


def top_k_indices(grid: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Return (layer, kv_group) of the top-k cells by raw value (descending)."""
    if k <= 0:
        return []
    num_groups = grid.shape[1]
    k = min(k, grid.size)
    flat = grid.flatten()
    top_idx = np.argsort(flat)[-k:]
    return [divmod(int(i), num_groups) for i in top_idx]


def draw_top_k_boxes(ax, top_cells, color="#BE1E2D"):
    """Outline each top-k cell on a (x=layer, y=kv_group) axis."""
    for layer, group in top_cells:
        rect = mpatches.FancyBboxPatch(
            (layer - 0.5, group - 0.5),
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
        description="Plot KV-group x layer score heatmaps from kv_group_analysis envelopes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--heatmap",
        nargs=2,
        action="append",
        metavar=("TITLE", "PATH"),
        required=True,
        help="Subplot title and __kvgroup.json path (repeat to stack panels vertically).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Highlight the top-k KV-group cells with bounding boxes (default: {DEFAULT_TOP_K}; pass 0 to disable).",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Use a symmetric (-|max|, +|max|) colour scale. Default uses (min, max).",
    )
    parser.add_argument("--output", type=str, required=True, help="Output SVG path.")
    return parser


def build_figure(
    heatmaps: list[tuple[str, np.ndarray]],
    top_k: int = DEFAULT_TOP_K,
    symmetric: bool = False,
):
    """Render a horizontal layer x KV-group heatmap (one row per panel).

    Each panel: layer on x-axis, KV-group on y-axis. Cells use
    ``aspect='auto'`` so the figure stays compact when the layer count
    dwarfs the KV-group count (typical: ~36-64 layers x 8 groups).
    """
    setup_plot_style()

    n = len(heatmaps)
    assert n >= 1, "Need at least one --heatmap"

    # Fixed panel size: all panels share the same figsize; models with more
    # KV groups or layers simply get smaller cells (aspect="auto" handles this).
    PANEL_W = 6.4  # inches per panel (width)
    PANEL_H = 2.5  # inches per panel (height)
    fig_w = PANEL_W
    fig_h = PANEL_H * n + 0.6 * (n - 1)

    fig, axes = plt.subplots(
        n,
        1,
        figsize=(fig_w, fig_h),
        squeeze=False,
        layout="constrained",
    )
    axes_col = axes[:, 0]

    top_cells_per: list[list[tuple[int, int]]] = []

    for (title, grid), ax in zip(heatmaps, axes_col):
        # imshow plots rows on y; we want layer on x and kv_group on y, so transpose.
        display = grid.T  # shape: (num_kv_groups, num_layers)
        num_groups, num_layers = display.shape

        if symmetric:
            vabs = max(abs(grid.min()), abs(grid.max()))
            vmin, vmax, cmap = -vabs, vabs, "coolwarm"
        else:
            vmin, vmax, cmap = float(grid.min()), float(grid.max()), "viridis"

        ax.set_facecolor("white")
        ax.grid(False)
        im = ax.imshow(
            display,
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )

        top_cells = top_k_indices(grid, top_k) if top_k > 0 else []
        top_cells_per.append(top_cells)
        if top_cells:
            draw_top_k_boxes(ax, top_cells)

        ax.set_xlabel("Layer")
        ax.set_ylabel("KV group")
        ax.set_yticks(np.arange(num_groups))
        # Layer axis: pick a sensible tick stride so labels don't crowd.
        stride = 1 if num_layers <= 16 else 2 if num_layers <= 40 else 4
        ax.set_xticks(np.arange(0, num_layers, stride))
        ax.tick_params(axis="both", pad=-1.5)

        ax.text(
            0.5,
            PANEL_TITLE_Y,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=PANEL_TITLE_FONT_SIZE,
            fontweight="bold",
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    return fig, top_cells_per


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    heatmaps: list[tuple[str, np.ndarray]] = []
    metas: list[dict] = []
    for title, path in args.heatmap:
        grid, meta = load_kvgroup_envelope(path)
        heatmaps.append((title, grid))
        metas.append(meta)

    fig, top_cells_per = build_figure(
        heatmaps,
        top_k=args.top_k,
        symmetric=args.symmetric,
    )

    out = Path(args.output)
    save_figure(fig, out)
    print(f"Saved to {out}")

    for (title, grid), top_cells, meta in zip(heatmaps, top_cells_per, metas):
        layers = meta.get("num_layers") if "num_layers" in meta else grid.shape[0]
        groups = meta.get("num_kv_heads") if "num_kv_heads" in meta else grid.shape[1]
        print(f"{title}: grid={grid.shape}, layers={layers}, kv_groups={groups}")
        if top_cells:
            print(f"  top-{args.top_k} (layer, kv_group): {sorted(top_cells)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
