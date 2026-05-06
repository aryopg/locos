#!/usr/bin/env python3
"""Overview figure for the logit-contribution retrieval head detection method.

Two-panel figure for NeurIPS submission:

  (a) Spatial contrast schematic: context bar with needle highlighted,
      showing the computation flow from per-position logit contribution
      to Phi+/Phi- to the final S score, illustrated for three head
      archetypes (retrieval, parametric, irrelevant).
  (b) Head-map comparison: two square layer x head heatmaps stacked
      vertically — attention-based R on top, logit-contribution S below,
      with top-k markers showing the methods identify different heads.

Usage:
    # Synthetic demo (no data files needed)
    python locos/plot_logit_contrib_overview.py

    # With real attention-based scores
    python locos/plot_logit_contrib_overview.py \\
        --attn-json retrieval_heads/Meta-Llama-3-8B-Instruct_contrastive_nolima_topk10_contrastive.json

    # With both real score files
    python locos/plot_logit_contrib_overview.py \\
        --attn-json retrieval_heads/..._contrastive.json \\
        --logit-json retrieval_heads/..._logit_contrib.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from locos_eval.utils.plotting import (
    FONT_SIZE_AXIS_LABEL,
    save_figure,
    save_legend,
    setup_plot_style,
)

# ---------------------------------------------------------------------------
# Color palette (colorblind-friendly, consistent with project)
# ---------------------------------------------------------------------------

C_NEEDLE = "#DE8F05"  # orange — needle region
C_OFFNEEDLE = "#0173B2"  # blue — off-needle
C_LOGIT = "#029E73"  # green — logit-contribution / retrieval head
C_NEGATIVE = "#D55E00"  # red-orange — negative S (parametric heads)
C_NEUTRAL = "#777777"  # gray — neutral / irrelevant


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _synthetic_head_scores(
    n_layers: int = 32,
    n_heads: int = 32,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic attention-based R and logit-contribution S head scores.

    Returns (R, S) each shape (n_layers, n_heads).
    Designed so that top heads in R and S are largely different.
    """
    rng = np.random.RandomState(seed)

    # Attention-based R: mostly zero, a few heads with high scores
    R = np.maximum(rng.exponential(0.0005, size=(n_layers, n_heads)), 0)

    # Attention heads cluster in layers 12-20 (mid layers)
    attn_hot_heads = [
        (13, 5),
        (14, 2),
        (15, 8),
        (16, 1),
        (17, 12),
        (18, 3),
        (18, 20),
        (19, 7),
        (19, 25),
        (20, 1),
        (20, 14),
        (21, 9),
        (22, 14),
        (23, 6),
        (24, 27),
    ]
    for l, h in attn_hot_heads:
        R[l, h] = rng.uniform(0.005, 0.012)

    R[16, 1] = 0.0086
    R[20, 1] = 0.0082
    R[20, 14] = 0.0079

    # Logit-contribution S: different distribution, some overlap
    S = rng.normal(0, 0.05, size=(n_layers, n_heads))

    # Logit-contribution heads cluster in layers 20-28 (later layers)
    logit_hot_heads = [
        (20, 1),
        (22, 14),
        (24, 27),
        (25, 3),
        (25, 18),
        (26, 7),
        (26, 15),
        (27, 2),
        (27, 11),
        (27, 22),
        (28, 5),
        (28, 14),
        (29, 1),
        (29, 8),
        (30, 3),
    ]
    for l, h in logit_hot_heads:
        S[l, h] = rng.uniform(2.0, 5.0)

    S[27, 11] = 5.8
    S[28, 14] = 5.2
    S[29, 1] = 4.9

    # Shared heads (found by both methods)
    S[20, 1] = 3.5
    S[22, 14] = 3.1
    S[24, 27] = 2.8

    # A few parametric heads (negative S)
    S[10, 4] = -1.8
    S[12, 7] = -1.5
    S[8, 15] = -1.2

    return R, S


# ---------------------------------------------------------------------------
# Panel (a): Spatial contrast schematic
# ---------------------------------------------------------------------------


def _plot_panel_a(ax) -> None:
    """Schematic: context bar + computation flow + three head archetypes."""
    # Use a unitless coordinate system; axis is turned off
    W = 100  # total width of the drawing area

    # ---- Context bar at top ----
    ns, ne = 35, 50  # needle span in drawing coords
    bar_y = 0.92
    bar_h = 0.055

    ax.barh(bar_y, ns, left=0, height=bar_h, color=C_OFFNEEDLE, alpha=0.25, edgecolor="black", linewidth=0.8)
    ax.barh(bar_y, ne - ns, left=ns, height=bar_h, color=C_NEEDLE, edgecolor="black", linewidth=0.8)
    ax.barh(bar_y, W - ne, left=ne, height=bar_h, color=C_OFFNEEDLE, alpha=0.25, edgecolor="black", linewidth=0.8)

    ax.text(ns / 2, bar_y, "context", ha="center", va="center", fontsize=8, color="#444444")
    ax.text((ns + ne) / 2, bar_y, "needle", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
    ax.text((ne + W) / 2, bar_y, "context", ha="center", va="center", fontsize=8, color="#444444")

    # ---- Equation summary below the context bar ----
    eq_y = 0.82
    ax.text(
        W / 2,
        eq_y,
        r"$\phi_{t,j}^{(l,h)} = \alpha_{t,j} \cdot \mathbf{u}_{y_t}^\top " r"W_O \, \mathbf{v}_{t,j}$",
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f0", edgecolor="#cccccc", linewidth=0.6),
    )

    # ---- Downward arrow ----
    ax.annotate("", xy=(W / 2, 0.73), xytext=(W / 2, 0.78), arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.2))
    ax.text(
        W / 2,
        0.705,
        r"sum over needle ($\Phi^+$) vs off-needle ($\Phi^-$), subtract",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#666666",
        fontstyle="italic",
    )

    # ---- Three example heads ----
    col_bar_start = 32
    max_val = 5.5
    bar_scale = (W - col_bar_start - 18) / max_val

    head_data = [
        ("Retrieval head\n(25, 3)", 4.7, 0.23, C_LOGIT),
        ("Parametric head\n(10, 4)", 0.4, 1.8, C_NEGATIVE),
        ("Irrelevant head\n(3, 12)", 0.15, 0.12, C_NEUTRAL),
    ]

    y_positions = [0.53, 0.35, 0.17]
    bar_h_small = 0.035

    for (label, phi_plus, phi_minus, color), y_pos in zip(head_data, y_positions):
        # Phi+ bar (needle contribution)
        w_plus = phi_plus * bar_scale
        ax.barh(
            y_pos + bar_h_small * 0.65,
            w_plus,
            left=col_bar_start,
            height=bar_h_small,
            color=C_NEEDLE,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            zorder=3,
        )
        ax.text(
            col_bar_start + w_plus + 0.8,
            y_pos + bar_h_small * 0.65,
            f"{phi_plus:.1f}",
            fontsize=6.5,
            va="center",
            ha="left",
            color="#555555",
        )

        # Phi- bar (off-needle, rescaled)
        w_minus = phi_minus * bar_scale
        ax.barh(
            y_pos - bar_h_small * 0.65,
            w_minus,
            left=col_bar_start,
            height=bar_h_small,
            color=C_OFFNEEDLE,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.45,
            zorder=3,
        )
        ax.text(
            col_bar_start + w_minus + 0.8,
            y_pos - bar_h_small * 0.65,
            f"{phi_minus:.2f}",
            fontsize=6.5,
            va="center",
            ha="left",
            color="#555555",
        )

        # Head label (left)
        ax.text(
            col_bar_start - 1.5,
            y_pos,
            label,
            ha="right",
            va="center",
            fontsize=7.5,
            color=color,
            fontweight="bold",
            linespacing=1.15,
        )

        # Score badge (right)
        s_val = phi_plus - phi_minus
        sign = "+" if s_val > 0 else ""
        badge_x = col_bar_start + max(w_plus, w_minus) + 7
        ax.text(
            badge_x,
            y_pos,
            f"$S = {sign}{s_val:.1f}$",
            ha="left",
            va="center",
            fontsize=8.5,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, linewidth=0.7, alpha=0.85),
        )

    # ---- Bar type key (bottom) ----
    key_y = 0.05
    key_x = col_bar_start
    ax.barh(key_y, 5, left=key_x, height=0.022, color=C_NEEDLE, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.text(
        key_x + 6.5,
        key_y,
        r"$\Phi^+$ (needle)",
        fontsize=7.5,
        va="center",
        ha="left",
        color=C_NEEDLE,
        fontweight="bold",
    )

    ax.barh(key_y, 5, left=key_x + 30, height=0.022, color=C_OFFNEEDLE, edgecolor="black", linewidth=0.5, alpha=0.45)
    ax.text(
        key_x + 37.5,
        key_y,
        r"$\Phi^-$ (off-needle, rescaled)",
        fontsize=7.5,
        va="center",
        ha="left",
        color=C_OFFNEEDLE,
        fontweight="bold",
    )

    ax.set_xlim(-3, W + 5)
    ax.set_ylim(-0.02, 1.0)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Panel (b): Stacked square heatmaps
# ---------------------------------------------------------------------------


def _load_real_scores(json_path: str | Path) -> np.ndarray:
    """Load scores from JSON and return (n_layers, n_heads) array."""
    from locos.utils.common import load_head_scores

    scores = load_head_scores(json_path)

    layers = set()
    heads_set = set()
    for key in scores:
        l, h = key.split("-")
        layers.add(int(l))
        heads_set.add(int(h))

    n_layers = max(layers) + 1
    n_heads = max(heads_set) + 1
    grid = np.zeros((n_layers, n_heads), dtype=np.float64)
    for key, val in scores.items():
        l, h = key.split("-")
        grid[int(l), int(h)] = val
    return grid


def _set_sparse_ticks(
    ax, n_layers: int, n_heads: int, flip_y: bool = True, show_xlabel: bool = True, show_ylabel: bool = True
) -> None:
    """Set sparse tick labels for a heatmap axes."""
    step_y = max(1, n_layers // 8)
    step_x = max(1, n_heads // 8)

    layer_indices = list(range(0, n_layers, step_y))
    if flip_y:
        ytick_positions = [n_layers - 1 - li for li in layer_indices]
    else:
        ytick_positions = layer_indices
    ytick_labels = [str(li) for li in layer_indices]

    xticks = list(range(0, n_heads, step_x))
    xlabels = [str(x) for x in xticks]

    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels if show_ylabel else [], fontsize=7)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels if show_xlabel else [], fontsize=7)


def _plot_panel_b(
    ax_R,
    ax_S,
    R: np.ndarray,
    S: np.ndarray,
    top_k: int = 10,
) -> None:
    """Vertically stacked square heatmaps: attention R (top) and logit S (bottom)."""
    n_layers, n_heads = R.shape
    assert S.shape == R.shape

    # --- Attention-based R heatmap (top) ---
    R_display = R.copy()
    vmax_R = np.percentile(R_display[R_display > 0], 95) if (R_display > 0).any() else 0.01
    ax_R.imshow(
        R_display[::-1],
        cmap="Blues",
        aspect="equal",
        interpolation="nearest",
        vmin=0,
        vmax=max(vmax_R, 1e-6),
    )
    ax_R.grid(False)
    ax_R.set_ylabel("Layer", fontsize=FONT_SIZE_AXIS_LABEL - 1)

    # Mark top-k heads
    flat_R = R.flatten()
    top_k_idx_R = np.argsort(flat_R)[-top_k:][::-1]
    for rank, idx in enumerate(top_k_idx_R):
        l, h = divmod(idx, n_heads)
        row = n_layers - 1 - l
        ms = 6 if rank < 3 else 4.5
        ax_R.plot(
            h, row, "o", markersize=ms, markeredgecolor="red", markerfacecolor="none", markeredgewidth=1.5, zorder=5
        )

    _set_sparse_ticks(ax_R, n_layers, n_heads, show_xlabel=False)

    # --- Logit-contribution S heatmap (bottom) ---
    # Power-law stretch: compress near-zero, emphasize signal
    S_display = np.sign(S) * np.abs(S) ** 0.5
    abs_max_display = np.abs(S_display).max()
    norm_S = mcolors.TwoSlopeNorm(
        vmin=-abs_max_display,
        vcenter=0,
        vmax=abs_max_display,
    )
    ax_S.imshow(
        S_display[::-1],
        cmap="RdBu",
        norm=norm_S,
        aspect="equal",
        interpolation="nearest",
    )
    ax_S.grid(False)
    ax_S.set_ylabel("Layer", fontsize=FONT_SIZE_AXIS_LABEL - 1)
    ax_S.set_xlabel("Head", fontsize=FONT_SIZE_AXIS_LABEL - 1)

    # Mark top-k heads
    flat_S = S.flatten()
    top_k_idx_S = np.argsort(flat_S)[-top_k:][::-1]
    for rank, idx in enumerate(top_k_idx_S):
        l, h = divmod(idx, n_heads)
        row = n_layers - 1 - l
        ms = 6 if rank < 3 else 4.5
        ax_S.plot(
            h, row, "o", markersize=ms, markeredgecolor="red", markerfacecolor="none", markeredgewidth=1.5, zorder=5
        )

    # Mark bottom-3 (most negative S, parametric heads)
    bottom_3_idx = np.argsort(flat_S)[:3]
    for idx in bottom_3_idx:
        l, h = divmod(idx, n_heads)
        row = n_layers - 1 - l
        ax_S.plot(
            h, row, "v", markersize=5, markeredgecolor=C_NEGATIVE, markerfacecolor="none", markeredgewidth=1.5, zorder=5
        )

    _set_sparse_ticks(ax_S, n_layers, n_heads, show_xlabel=True)

    # --- Overlap annotation ---
    # Compute overlap between top-k of R and S
    top_R_set = set(top_k_idx_R.tolist())
    top_S_set = set(top_k_idx_S.tolist())
    overlap = len(top_R_set & top_S_set)
    ax_S.text(
        0.5,
        -0.18,
        f"Top-{top_k} overlap: {overlap}/{top_k}",
        transform=ax_S.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#555555",
        fontstyle="italic",
    )


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------


def make_overview_figure(
    out_path: Path,
    attn_json: Path | None = None,
    logit_json: Path | None = None,
    seed: int = 42,
) -> None:
    """Assemble the two-panel overview figure."""
    setup_plot_style()

    # ---- Data ----
    if attn_json is not None:
        R = _load_real_scores(attn_json)
        n_layers, n_heads = R.shape
    else:
        R = None

    if logit_json is not None:
        S = _load_real_scores(logit_json)
        n_layers_s, n_heads_s = S.shape
        if R is not None:
            assert (n_layers, n_heads) == (n_layers_s, n_heads_s), f"Grid mismatch: R={R.shape} vs S={S.shape}"
        else:
            n_layers, n_heads = n_layers_s, n_heads_s
    else:
        S = None

    if R is None or S is None:
        R_syn, S_syn = _synthetic_head_scores(seed=seed)
        if R is None:
            R = R_syn
        if S is None:
            S = S_syn
        n_layers, n_heads = R.shape

    # ---- Layout: 2 columns ----
    # Left: schematic (panel a)
    # Right: 2 stacked square heatmaps (panel b)
    fig = plt.figure(figsize=(11, 5.5))

    gs_top = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.1, 0.9],
        wspace=0.30,
        left=0.02,
        right=0.98,
        top=0.93,
        bottom=0.08,
    )

    # Panel (a): single axes for the schematic
    ax_a = fig.add_subplot(gs_top[0])

    # Panel (b): 2 rows for stacked heatmaps
    gs_b = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=gs_top[1],
        hspace=0.22,
    )
    ax_R = fig.add_subplot(gs_b[0])
    ax_S = fig.add_subplot(gs_b[1])

    # ---- Draw panels ----
    _plot_panel_a(ax_a)
    _plot_panel_b(ax_R, ax_S, R, S)

    # ---- Panel labels ----
    label_kw = dict(fontsize=14, fontweight="bold", ha="left", va="top")
    ax_a.text(-0.03, 1.04, "(a)", transform=ax_a.transAxes, **label_kw)
    ax_R.text(-0.16, 1.12, "(b)", transform=ax_R.transAxes, **label_kw)

    # Sub-labels for (b) heatmaps
    ax_R.text(
        0.5,
        1.06,
        "Attention-based $R$",
        transform=ax_R.transAxes,
        fontsize=9,
        ha="center",
        va="bottom",
        fontstyle="italic",
    )
    ax_S.text(
        0.5,
        1.06,
        "Logit-contribution $S$",
        transform=ax_S.transAxes,
        fontsize=9,
        ha="center",
        va="bottom",
        fontstyle="italic",
    )

    # ---- Save ----
    save_figure(fig, out_path)

    # Shared legend
    legend_handles = [
        mpatches.Patch(facecolor=C_NEEDLE, edgecolor="black", linewidth=0.8, label="Needle span"),
        mpatches.Patch(facecolor=C_OFFNEEDLE, edgecolor="black", linewidth=0.8, label="Off-needle"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markeredgecolor="red",
            markerfacecolor="none",
            markersize=6,
            markeredgewidth=1.2,
            label="Top-$k$ retrieval",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markeredgecolor=C_NEGATIVE,
            markerfacecolor="none",
            markersize=6,
            markeredgewidth=1.2,
            label="Parametric (neg.\\ $S$)",
        ),
    ]
    legend_path = out_path.with_name(f"{out_path.stem}_legend{out_path.suffix}")
    save_legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        legend_path,
        ncol=4,
    )

    from rich.console import Console

    Console().print(f"[green]Saved overview figure:[/green] {out_path}")
    Console().print(f"[green]Saved legend:[/green] {legend_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate logit-contribution method overview figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--attn-json",
        type=Path,
        default=None,
        help="Attention-based contrastive scores JSON (for panel b). " "If omitted, uses synthetic data.",
    )
    parser.add_argument(
        "--logit-json",
        type=Path,
        default=None,
        help="Logit-contribution scores JSON (for panel b). " "If omitted, uses synthetic data.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/logit_contrib_overview.svg"),
        help="Output path for the figure.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    make_overview_figure(
        out_path=args.out,
        attn_json=args.attn_json,
        logit_json=args.logit_json,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
