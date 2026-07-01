#!/usr/bin/env python3
"""Paper figure for H2: KV-group coverage of top-k retrieval-head rankings.

For each model, plots ``unique_kv_groups(k) / min(k, total_kv_groups)`` as a
function of top-k on a log-x axis, with one line per detection method.
Small-multiples layout across models. A reader sees at a glance whether a
method over-concentrates on a few KV groups (flat curve) or spreads across
groups (near 1.0).

Consumes the tidy CSVs emitted by
``locos.analysis.kv_group_analysis``.

Usage:
    python -m locos.plotting.kv_group_coverage \\
        --csv analysis_out/kv_group/**/*__kv_group_stats.csv \\
        --output figures/kv_group_coverage.svg
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from locos_eval.utils.plotting import LINE_WIDTH, MODEL_PRETTY_NAMES, save_figure, setup_plot_style

METHOD_COLOURS = {
    "wu": "#4C72B0",
    "behavioral": "#4C72B0",
    "logit_contrib": "#C44E52",
    "logit_contribution": "#C44E52",
    "contrastive": "#55A868",
    "cri": "#8172B2",
    "tuned_lens": "#CCB974",
}

METHOD_PRETTY = {
    "wu": "Wu (behavioral)",
    "behavioral": "Wu (behavioral)",
    "logit_contrib": "Logit contribution",
    "logit_contribution": "Logit contribution",
    "contrastive": "Contrastive",
    "cri": "CRI",
    "tuned_lens": "Tuned-lens LC",
}

REFERENCE_K = (10, 20, 50)


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p) as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    # Coerce numeric columns
    for r in rows:
        r["k"] = int(r["k"])
        r["unique_groups"] = int(r["unique_groups"])
        r["coverage"] = float(r["coverage"])
        r["concentration"] = float(r["concentration"]) if r["concentration"] not in ("", "nan") else float("nan")
    return rows


def short_model_name(model: str) -> str:
    # Accept full HF names ("Qwen/Qwen3-8B") or bare slugs ("Qwen3-8B")
    slug = model.split("/")[-1]
    return MODEL_PRETTY_NAMES.get(slug, slug)


def plot_coverage(rows: list[dict], out_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to plot — check --csv inputs")

    setup_plot_style()

    # Group by (model, method)
    models = sorted({r["model"] for r in rows}, key=short_model_name)
    methods = sorted({r["method"] for r in rows})

    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.1 * n_rows), sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    for ax_idx, model in enumerate(models):
        ax = axes_flat[ax_idx]
        for method in methods:
            pts = sorted(
                [r for r in rows if r["model"] == model and r["method"] == method],
                key=lambda r: r["k"],
            )
            if not pts:
                continue
            ks = np.array([r["k"] for r in pts])
            cov = np.array([r["coverage"] for r in pts])
            ax.plot(
                ks,
                cov,
                marker="o",
                linewidth=LINE_WIDTH,
                color=METHOD_COLOURS.get(method),
                label=METHOD_PRETTY.get(method, method),
            )
        # Reference k markers
        for ref in REFERENCE_K:
            ax.axvline(ref, color="#888888", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("top-$k$ heads")
        # Subplot "title" as annotation in-plot (we cannot use suptitles per
        # the project plotting conventions — titles are stripped by save_figure)
        ax.text(
            0.97,
            0.04,
            short_model_name(model),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8, "pad": 2.0},
        )
    # Y-label once on left column
    for row_idx in range(n_rows):
        axes[row_idx, 0].set_ylabel("unique KV groups / $\\min(k, G)$")

    # Hide any unused axes
    for ax_idx in range(n_models, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    # One legend on the first axis (save_figure strips and re-saves separately)
    axes_flat[0].legend(loc="upper left", frameon=True)

    fig.tight_layout()
    save_figure(fig, out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="One or more kv_group_stats.csv files (glob via shell)",
    )
    parser.add_argument("--output", required=True, help="Output SVG path")
    args = parser.parse_args()

    paths = [Path(p) for p in args.csv]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing CSVs: {missing}")

    rows = load_rows(paths)
    plot_coverage(rows, Path(args.output))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
