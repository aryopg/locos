#!/usr/bin/env python3
"""Bucketed retrieval score analysis: stacked bars + per-model heatmaps.

Produces two groups of plots:
1. **Stacked bar chart**: one bar per model, segments show fraction of heads
   falling into each score bucket.
2. **Heatmaps**: one per model, with layer on y-axis, head on x-axis, each
   cell colored by the bucket its mean retrieval score falls into.

Usage:
    # All _nolima.json files with default buckets (0, 0-0.1, 0.1-0.5, >0.5)
    python locos/plot_retrieval_score_buckets.py \
        retrieval_heads/*_nolima.json

    # Custom buckets
    python locos/plot_retrieval_score_buckets.py \
        retrieval_heads/*_nolima.json --buckets 0 0.05 0.2 0.5

    # Only NIAH heads
    python locos/plot_retrieval_score_buckets.py \
        retrieval_heads/Qwen3-*.json --exclude-nolima
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from locos_eval.utils.plotting import (
    FIGURE_SIZE,
    FONT_SIZE_AXIS_TICKS,
    save_figure,
    save_legend,
    setup_plot_style,
)

# Default bucket boundaries: [=0, (0, 0.1], (0.1, 0.5], >0.5]
DEFAULT_BUCKET_EDGES = [0.1, 0.5]

# Heatmap palette: one color per bucket, light to dark
# First color (=0 bucket) is a very light blue to distinguish from white background
BUCKET_COLORS = ["#dce9f5", "#a6d4f2", "#4c9ed9", "#1a5276"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

from locos.utils.common import load_head_scores


def model_label(path: Path) -> str:
    """Short display label from filename, stripping _nolima suffix."""
    stem = path.stem
    # Strip common suffixes for cleaner axis labels
    for suffix in ("_nolima", "_niah", "_cri"):
        stem = stem.removesuffix(suffix)
    return stem


def _model_family(label: str) -> str:
    """Group model labels into families for stacked bar ordering."""
    lower = label.lower()
    if "llama" in lower:
        return "Llama-3"
    if "qwen" in lower:
        return "Qwen3"
    if "gemma" in lower:
        return "Gemma-3"
    return label


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------


def bucketize(score: float, edges: list[float]) -> int:
    """Return bucket index for a score given sorted bucket edges.

    Buckets:
        0  → score == 0
        1  → 0 < score <= edges[0]
        2  → edges[0] < score <= edges[1]
        ...
        N  → score > edges[-1]
    """
    if score == 0.0:
        return 0
    for i, edge in enumerate(edges):
        if score <= edge:
            return i + 1
    return len(edges) + 1


def bucket_labels(edges: list[float]) -> list[str]:
    """Generate human-readable bucket labels."""
    labels = [r"$= 0$"]
    for i, edge in enumerate(edges):
        lo = 0.0 if i == 0 else edges[i - 1]
        labels.append(f"({lo:.4g}, {edge:.4g}]")
    labels.append(r"$>$ " + f"{edges[-1]:.4g}")
    return labels


def compute_quantile_edges(
    all_scores: dict[str, dict[str, float]],
    n_buckets: int,
) -> list[float]:
    """Compute bucket edges from quantiles of non-zero scores across all models.

    Args:
        all_scores: {model: {layer-head: mean_score}}
        n_buckets: Number of non-zero buckets (total buckets = n_buckets + 1
            because the zero bucket is always separate).

    Returns:
        Sorted list of n_buckets - 1 edges (the last bucket is "> last_edge").
    """
    nonzero = [s for scores in all_scores.values() for s in scores.values() if s > 0.0]
    assert len(nonzero) > 0, "No non-zero scores found for quantile computation"

    # Compute n_buckets - 1 quantile edges that split non-zero scores evenly
    quantiles = np.linspace(0, 1, n_buckets + 1)[1:-1]  # exclude 0% and 100%
    edges = np.quantile(nonzero, quantiles).tolist()

    # Deduplicate (can happen when many scores are identical)
    edges = sorted(set(edges))
    assert len(edges) > 0, "All non-zero scores are identical; cannot form quantile buckets"

    return edges


# ---------------------------------------------------------------------------
# Stacked bar plot
# ---------------------------------------------------------------------------


def make_stacked_bar(
    all_scores: dict[str, dict[str, float]],
    edges: list[float],
    out_path: Path,
) -> None:
    """Stacked horizontal bar chart: fraction of heads per bucket per model."""
    setup_plot_style()

    n_buckets = len(edges) + 2  # zero + between edges + above last
    blabels = bucket_labels(edges)

    # Compute fractions per model
    model_names = list(all_scores.keys())
    fractions = np.zeros((len(model_names), n_buckets))

    for i, (name, scores) in enumerate(all_scores.items()):
        total = len(scores)
        assert total > 0, f"No heads in {name}"
        counts = [0] * n_buckets
        for s in scores.values():
            counts[bucketize(s, edges)] += 1
        fractions[i] = [c / total for c in counts]

    # Sort by model family then name
    order = sorted(range(len(model_names)), key=lambda i: (_model_family(model_names[i]), model_names[i]))
    model_names = [model_names[i] for i in order]
    fractions = fractions[order]

    fig, ax = plt.subplots(figsize=(FIGURE_SIZE[0], max(2.5, len(model_names) * 0.55)))

    y = np.arange(len(model_names))
    left = np.zeros(len(model_names))
    handles = []

    colors = BUCKET_COLORS[:n_buckets]
    # Extend palette if we have more buckets than default colors
    while len(colors) < n_buckets:
        colors.append(plt.cm.Blues(0.3 + 0.7 * len(colors) / n_buckets))

    for b in range(n_buckets):
        bars = ax.barh(
            y,
            fractions[:, b],
            left=left,
            height=0.6,
            color=colors[b],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        handles.append(bars)
        left += fractions[:, b]

    ax.set_yticks(y)
    ax.set_yticklabels(model_names, fontsize=FONT_SIZE_AXIS_TICKS)
    ax.set_xlabel("Fraction of Heads")
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.invert_yaxis()

    fig.tight_layout()
    save_figure(fig, out_path)

    # Legend
    legend_path = out_path.with_name(out_path.stem + "_legend" + out_path.suffix)
    save_legend(
        [h[0] for h in handles],
        blabels,
        legend_path,
        ncol=min(n_buckets, 4),
    )

    print(f"Saved stacked bar: {out_path}")
    print(f"Saved legend:      {legend_path}")

    _print_bucket_table(all_scores, edges, model_names, fractions)


def _print_bucket_table(
    all_scores: dict[str, dict[str, float]],
    edges: list[float],
    model_names: list[str],
    fractions: np.ndarray,
) -> None:
    """Print bucket distribution table."""
    from rich.console import Console
    from rich.table import Table

    blabels = bucket_labels(edges)
    console = Console()
    table = Table(title="Retrieval Score Bucket Distribution")
    table.add_column("Model", style="bold")
    for bl in blabels:
        table.add_column(bl, justify="right")
    table.add_column("Total", justify="right", style="dim")

    for i, name in enumerate(model_names):
        total = len(all_scores[name])
        row = [name]
        for b in range(len(blabels)):
            count = round(fractions[i, b] * total)
            pct = fractions[i, b]
            row.append(f"{count} ({pct:.1%})")
        row.append(str(total))
        table.add_row(*row)

    console.print(table)


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------


def make_heatmaps(
    all_scores: dict[str, dict[str, float]],
    edges: list[float],
    out_dir: Path,
) -> None:
    """Create one heatmap per model: layer × head colored by bucket."""
    setup_plot_style()

    n_buckets = len(edges) + 2
    blabels = bucket_labels(edges)

    colors = BUCKET_COLORS[:n_buckets]
    while len(colors) < n_buckets:
        colors.append(plt.cm.Blues(0.3 + 0.7 * len(colors) / n_buckets))

    cmap = mcolors.ListedColormap(colors)
    # Boundaries for BoundaryNorm: map bucket index 0..n_buckets-1
    bounds = list(range(n_buckets + 1))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for label, scores in all_scores.items():
        # Determine grid dimensions
        layers = set()
        heads = set()
        for key in scores:
            l, h = key.split("-")
            layers.add(int(l))
            heads.add(int(h))

        n_layers = max(layers) + 1
        n_heads = max(heads) + 1

        grid = np.full((n_layers, n_heads), 0, dtype=int)
        for key, score in scores.items():
            l, h = key.split("-")
            grid[int(l), int(h)] = bucketize(score, edges)

        # Compact figure size for NeurIPS multi-panel layouts
        aspect_ratio = n_heads / n_layers
        fig_h = 3.0
        fig_w = max(2.5, fig_h * aspect_ratio + 0.6)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # Flip grid so layer 0 is at the bottom
        grid_flipped = grid[::-1]
        ax.imshow(
            grid_flipped,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )

        # Remove grid lines from heatmap
        ax.grid(False)

        # Axis labels
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")

        # Sparse tick labels for large grids (with flipped y-axis labels)
        _set_sparse_ticks(ax, n_layers, n_heads, flip_y=True)

        fig.tight_layout()

        out_path = out_dir / f"retrieval_heatmap_{label}.svg"
        save_figure(fig, out_path)
        print(f"Saved heatmap: {out_path}")

    # Save shared legend for heatmaps
    _save_heatmap_legend(colors, blabels, out_dir / "retrieval_heatmap_legend.svg")


def _set_sparse_ticks(ax, n_layers: int, n_heads: int, flip_y: bool = False) -> None:
    """Set tick labels, thinning them out for large grids."""
    if n_layers <= 48:
        step_y = max(1, n_layers // 16)
    else:
        step_y = max(1, n_layers // 12)

    if n_heads <= 32:
        step_x = max(1, n_heads // 16)
    else:
        step_x = max(1, n_heads // 12)

    # Layer ticks: image row 0 is top, so for flip_y we label top→bottom
    # as (n_layers-1)→0
    layer_indices = list(range(0, n_layers, step_y))
    if flip_y:
        # Image row i maps to layer (n_layers - 1 - i)
        ytick_positions = [n_layers - 1 - li for li in layer_indices]
        ytick_labels = [str(li) for li in layer_indices]
    else:
        ytick_positions = layer_indices
        ytick_labels = [str(li) for li in layer_indices]

    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=7)

    xticks = list(range(0, n_heads, step_x))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], fontsize=7)


def _save_heatmap_legend(colors: list, labels: list[str], path: Path) -> None:
    """Save a separate legend for the heatmap color scheme."""
    import matplotlib.patches as mpatches

    handles = [mpatches.Patch(facecolor=c, edgecolor="black", linewidth=1, label=l) for c, l in zip(colors, labels)]
    save_legend(handles, labels, path, ncol=min(len(labels), 4))
    print(f"Saved heatmap legend: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Bucketed retrieval score analysis: stacked bars + heatmaps")
    parser.add_argument(
        "heads_json",
        nargs="+",
        type=Path,
        help="One or more retrieval heads JSON files",
    )
    parser.add_argument(
        "--buckets",
        nargs="+",
        type=float,
        default=DEFAULT_BUCKET_EDGES,
        help="Bucket edges (excluding 0). Default: 0.1 0.5 " "(creates buckets: =0, 0-0.1, 0.1-0.5, >0.5)",
    )
    parser.add_argument(
        "--quantile",
        type=int,
        default=None,
        metavar="N",
        help="Use N quantile-based buckets for non-zero scores instead of "
        "fixed edges. Overrides --buckets. (e.g. --quantile 3 creates "
        "3 non-zero buckets plus the zero bucket)",
    )
    parser.add_argument(
        "--exclude-nolima",
        action="store_true",
        help="Skip files with '_nolima' in the name",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for all figures (default: figures/)",
    )
    args = parser.parse_args()

    # Load all models first (needed for quantile computation)
    all_scores: dict[str, dict[str, float]] = {}
    for path in args.heads_json:
        if args.exclude_nolima and "_nolima" in path.stem:
            continue
        assert path.exists(), f"File not found: {path}"
        label = model_label(path)
        all_scores[label] = load_head_scores(path)

    assert len(all_scores) > 0, "No valid retrieval head files found"

    # Determine bucket edges
    if args.quantile is not None:
        assert args.quantile >= 2, "--quantile must be >= 2"
        edges = compute_quantile_edges(all_scores, args.quantile)
        from rich.console import Console

        Console().print(
            f"[bold]Quantile edges ({args.quantile} non-zero buckets):[/bold] {[f'{e:.4g}' for e in edges]}"
        )
    else:
        edges = sorted(args.buckets)
        assert all(e > 0 for e in edges), "Bucket edges must be positive (zero bucket is automatic)"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stacked bar chart
    stacked_path = args.out_dir / "retrieval_score_buckets.svg"
    make_stacked_bar(all_scores, edges, stacked_path)

    # 2. Per-model heatmaps
    make_heatmaps(all_scores, edges, args.out_dir)


if __name__ == "__main__":
    main()
