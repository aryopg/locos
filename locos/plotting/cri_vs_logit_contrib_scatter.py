#!/usr/bin/env python3
"""Paper figure for H3: CRI vs logit-contribution scatter coloured by layer depth.

Consumes one or more ``direct_path_bias.csv`` files produced by
``locos.analysis.direct_path_bias`` and plots a small-
multiples scatter of CRI score vs LC score, with points coloured by
relative layer depth. Spearman ρ / Kendall τ are printed per panel.

If mid-depth points cluster in the "high CRI, low LC" quadrant, direct-path
bias is real. Uniform colour mixing = bias concern is closed.

Usage:
    python -m locos.plotting.cri_vs_logit_contrib_scatter \\
        --bias-csv analysis_out/direct_path_bias/*/direct_path_bias.csv \\
        --agreement-csv analysis_out/direct_path_bias/*/agreement_summary.csv \\
        --output figures/cri_vs_logit_contrib_scatter.svg
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from locos_eval.utils.plotting import MODEL_PRETTY_NAMES, save_figure, setup_plot_style


def load_bias_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append(r)
    for r in rows:
        r["cri_score"] = float(r["cri_score"])
        r["lc_score"] = float(r["lc_score"])
        r["layer_depth"] = float(r["layer_depth"])
        r["layer"] = int(r["layer"])
        r["head"] = int(r["head"])
    return rows


def load_agreement_rows(paths: list[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                out[r["model"]] = {
                    "spearman": float(r["spearman"]),
                    "kendall": float(r["kendall"]),
                }
    return out


def zscore(values: np.ndarray) -> np.ndarray:
    std = values.std()
    if std == 0:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def short_model_name(model: str) -> str:
    slug = model.split("/")[-1]
    return MODEL_PRETTY_NAMES.get(slug, slug)


def plot(
    bias_rows: list[dict],
    agreement: dict[str, dict],
    out_path: Path,
) -> None:
    if not bias_rows:
        raise ValueError("No rows; check --bias-csv paths")

    setup_plot_style()

    models = sorted({r["model"] for r in bias_rows}, key=short_model_name)
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.3 * n_cols, 3.3 * n_rows), sharex=False, sharey=False, squeeze=False
    )
    axes_flat = axes.flatten()

    scatter_ref = None
    for ax_idx, model in enumerate(models):
        ax = axes_flat[ax_idx]
        pts = [r for r in bias_rows if r["model"] == model]
        cri_z = zscore(np.array([p["cri_score"] for p in pts]))
        lc_z = zscore(np.array([p["lc_score"] for p in pts]))
        depths = np.array([p["layer_depth"] for p in pts])

        sc = ax.scatter(
            lc_z,
            cri_z,
            c=depths,
            cmap="viridis",
            s=18,
            edgecolors="black",
            linewidths=0.3,
            vmin=0.0,
            vmax=1.0,
        )
        if scatter_ref is None:
            scatter_ref = sc

        lims = [
            min(float(lc_z.min()), float(cri_z.min())) - 0.2,
            max(float(lc_z.max()), float(cri_z.max())) + 0.2,
        ]
        ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axhline(0.0, color="#888888", linewidth=0.6, alpha=0.5)
        ax.axvline(0.0, color="#888888", linewidth=0.6, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ag = agreement.get(model, {})
        spearman = ag.get("spearman")
        kendall = ag.get("kendall")
        if spearman is not None and kendall is not None:
            ax.text(
                0.03,
                0.97,
                f"$\\rho$={spearman:.2f}\n$\\tau$={kendall:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8, "pad": 2.0},
            )
        ax.text(
            0.97,
            0.03,
            short_model_name(model),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8, "pad": 2.0},
        )

    for row_idx in range(n_rows):
        axes[row_idx, 0].set_ylabel("CRI score (z)")
    for col_idx in range(n_cols):
        axes[n_rows - 1, col_idx].set_xlabel("Logit-contribution (z)")

    for ax_idx in range(n_models, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    if scatter_ref is not None:
        cbar = fig.colorbar(
            scatter_ref,
            ax=axes,
            fraction=0.02,
            pad=0.04,
            orientation="vertical",
        )
        cbar.set_label("relative layer depth (0 = early, 1 = late)")

    # NOTE: per the project plotting convention, titles belong in LaTeX.
    # save_figure strips any in-figure titles before saving. We rely on the
    # per-panel annotation for model identity and on the LaTeX caption for
    # the figure-level heading.
    save_figure(fig, out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bias-csv", nargs="+", required=True, help="direct_path_bias.csv files")
    parser.add_argument("--agreement-csv", nargs="+", default=[], help="agreement_summary.csv files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bias_rows = load_bias_rows([Path(p) for p in args.bias_csv])
    agreement = load_agreement_rows([Path(p) for p in args.agreement_csv]) if args.agreement_csv else {}
    plot(bias_rows, agreement, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
