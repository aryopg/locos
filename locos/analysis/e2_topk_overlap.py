#!/usr/bin/env python3
"""E2 — Top-k overlap curves across detectors.

Hypothesis: LOCOS and token-matching select substantially disjoint head sets;
quantify how much of that disjointness the OV term is responsible for.

No new compute required — uses existing score files.

Outputs:
    analysis/outputs/e2/e2_jaccard_curves.svg
    analysis/outputs/e2/e2_jaccard_curves_legend.svg
    analysis/outputs/e2/e2_jaccard_data.csv

Usage:
    python locos/analysis/e2_topk_overlap.py
    python locos/analysis/e2_topk_overlap.py --no-download
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from locos_eval.utils.plotting import LINE_WIDTH, save_figure, setup_plot_style
from locos.analysis._utils import (
    ALL_MODELS,
    JACCARD_K_VALUES,
    MODEL_LABELS,
    get_output_dir,
    jaccard,
    load_score_file,
    top_k_heads,
)

console = Console()

PAIRS = [
    ("LOCOS", "locos", "nolima", "Wu/NIAH", "wu", "niah"),
    ("LOCOS", "locos", "nolima", "Wu/NoLiMa", "wu", "nolima"),
    ("LOCOS", "locos", "nolima", "α-spatial", "alpha_spatial", "nolima"),
    ("Wu/NIAH", "wu", "niah", "Wu/NoLiMa", "wu", "nolima"),
]

PAIR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]


def _expected_random_jaccard(k: int, total_heads: int) -> float:
    """Expected Jaccard when two top-k sets drawn uniformly without replacement."""
    if total_heads <= 0 or k <= 0:
        return 0.0
    return k / (2 * total_heads - k)


def _bootstrap_jaccard_ci(
    scores_a: dict[str, list[float]],
    scores_b: dict[str, list[float]],
    k: int,
    B: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap Jaccard@k over trial subsamples.

    Returns (mean, lo, hi).
    """
    rng = np.random.default_rng(seed)
    heads = sorted(set(scores_a) & set(scores_b))
    if not heads:
        return 0.0, 0.0, 0.0

    # Per-trial scores for each head
    trial_scores_a = np.array([scores_a.get(h, [0.0]) for h in heads])  # (H, T_a)
    trial_scores_b = np.array([scores_b.get(h, [0.0]) for h in heads])  # (H, T_b)

    n_trials_a = trial_scores_a.shape[1]
    n_trials_b = trial_scores_b.shape[1]

    j_boots = []
    for _ in range(B):
        # Resample trials for each detector independently
        idx_a = rng.integers(0, n_trials_a, size=n_trials_a)
        idx_b = rng.integers(0, n_trials_b, size=n_trials_b)
        means_a = {h: float(trial_scores_a[i, idx_a].mean()) for i, h in enumerate(heads)}
        means_b = {h: float(trial_scores_b[i, idx_b].mean()) for i, h in enumerate(heads)}
        set_a = top_k_heads(means_a, k)
        set_b = top_k_heads(means_b, k)
        j_boots.append(jaccard(set_a, set_b))

    j_boots_arr = np.array(j_boots)
    return (
        float(j_boots_arr.mean()),
        float(np.percentile(j_boots_arr, 2.5)),
        float(np.percentile(j_boots_arr, 97.5)),
    )


def run(download: bool = True, B: int = 1000) -> None:
    setup_plot_style()
    out_dir = get_output_dir("e2")

    # Load all needed score files
    score_cache: dict[tuple[str, str, str], dict[str, list[float]] | None] = {}
    for model in ALL_MODELS:
        for _, ma, da, _, mb, db in PAIRS:
            for method, dataset in [(ma, da), (mb, db)]:
                key = (model, method, dataset)
                if key not in score_cache:
                    sf = load_score_file(model, method, dataset, download=download)
                    score_cache[key] = sf.scores if sf is not None else None

    models_to_plot = [m for m in ALL_MODELS if score_cache[(m, "locos", "nolima")] is not None]
    assert models_to_plot, "No models with LOCOS score files found."

    n_models = len(models_to_plot)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten() if n_models > 1 else [axes]

    csv_rows: list[dict] = []

    for ax_idx, (model, ax) in enumerate(zip(models_to_plot, axes_flat)):
        label = MODEL_LABELS[model]
        total_heads = len(score_cache[(model, "locos", "nolima")] or {})

        for pair_idx, (name_a, ma, da, name_b, mb, db) in enumerate(PAIRS):
            scores_a = score_cache.get((model, ma, da))
            scores_b = score_cache.get((model, mb, db))
            if scores_a is None or scores_b is None:
                continue

            j_means, j_los, j_his = [], [], []
            for k in JACCARD_K_VALUES:
                j_mean, j_lo, j_hi = _bootstrap_jaccard_ci(scores_a, scores_b, k, B=B)
                j_means.append(j_mean)
                j_los.append(j_lo)
                j_his.append(j_hi)
                csv_rows.append(
                    {
                        "model": label,
                        "pair": f"{name_a} vs {name_b}",
                        "k": k,
                        "jaccard_mean": f"{j_mean:.4f}",
                        "jaccard_lo": f"{j_lo:.4f}",
                        "jaccard_hi": f"{j_hi:.4f}",
                    }
                )

            color = PAIR_COLORS[pair_idx % len(PAIR_COLORS)]
            ax.plot(
                JACCARD_K_VALUES,
                j_means,
                color=color,
                lw=LINE_WIDTH,
                label=f"{name_a} vs {name_b}",
            )
            ax.fill_between(JACCARD_K_VALUES, j_los, j_his, color=color, alpha=0.2)

        # Expected random baseline
        random_j = [_expected_random_jaccard(k, total_heads) for k in JACCARD_K_VALUES]
        ax.plot(JACCARD_K_VALUES, random_j, color="gray", lw=1.0, ls="--", alpha=0.6, label="Random")

        ax.set_xlabel("k", fontsize=10)
        ax.set_ylabel("Jaccard", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xscale("log")
        ax.text(0.05, 0.95, label, transform=ax.transAxes, va="top", ha="left", fontsize=9, fontweight="bold")
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    for ax in axes_flat[len(models_to_plot) :]:
        ax.set_visible(False)

    fig.tight_layout()
    save_figure(fig, out_dir / "e2_jaccard_curves.svg")

    # CSV
    csv_path = out_dir / "e2_jaccard_data.csv"
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)

    # Summary table
    table = Table(title="E2 — Jaccard@50 Summary (mean across bootstrap)")
    table.add_column("Model")
    for _, _ma, _da, name_b, _mb, _db in PAIRS:
        table.add_column(f"vs {name_b}", justify="right")
    for model in models_to_plot:
        label = MODEL_LABELS[model]
        row_vals = [label]
        for _, ma, da, _name_b, mb, db in PAIRS:
            scores_a = score_cache.get((model, ma, da))
            scores_b = score_cache.get((model, mb, db))
            if scores_a is None or scores_b is None:
                row_vals.append("—")
            else:
                j_mean, _, _ = _bootstrap_jaccard_ci(scores_a, scores_b, 50, B=100)
                row_vals.append(f"{j_mean:.3f}")
        table.add_row(*row_vals)
    console.print(table)
    console.print(f"\n[dim]Saved:[/dim] {out_dir}/e2_jaccard_curves.svg, e2_jaccard_data.csv")


def mean_scores_from_dict(scores: dict[str, list[float]]) -> dict[str, float]:
    return {k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2 — Top-k overlap curves across detectors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples.")
    args = parser.parse_args()
    run(download=not args.no_download, B=args.bootstrap)


if __name__ == "__main__":
    main()
