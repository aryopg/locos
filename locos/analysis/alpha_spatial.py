#!/usr/bin/env python3
"""E1 — α-spatial baseline score + rank diagnostics.

Hypothesis: The OV projection contributes head-selection information beyond
attention spatial contrast. Null: rankings from α alone match LOCOS.

No new compute required — uses existing LOCOS and α-spatial score files.

Outputs:
    analysis/outputs/e1/e1_rankscatter.svg   (6-panel hexbin)
    analysis/outputs/e1/e1_rankscatter_legend.svg
    analysis/outputs/e1/e1_stats.csv

Decision rule printed to console:
    Median Jaccard@50 < 0.4  → proceed to E7 expecting separation
    Median Jaccard@50 > 0.7  → central claim endangered
    0.4–0.7                   → E7 decides

Usage:
    python locos/analysis/alpha_spatial.py
    python locos/analysis/alpha_spatial.py --no-download
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

from locos.analysis._utils import (
    ALL_MODELS,
    MODEL_LABELS,
    get_output_dir,
    jaccard,
    load_score_file,
    mean_scores,
    rbo,
    top_k_heads,
)
from locos_eval.utils.plotting import save_figure, setup_plot_style

console = Console()


def _rank_vector(scores: dict[str, float]) -> dict[str, int]:
    """Map head key → rank (1 = highest score)."""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {key: i + 1 for i, (key, _) in enumerate(ranked)}


def run(download: bool = True) -> None:
    setup_plot_style()

    out_dir = get_output_dir("e1")
    rows: list[dict] = []

    models_ok = []
    model_data: list[tuple[str, dict, dict]] = []

    for model in ALL_MODELS:
        label = MODEL_LABELS[model]
        locos_sf = load_score_file(model, "locos", "nolima", download=download)
        alpha_sf = load_score_file(model, "alpha_spatial", "nolima", download=download)
        if locos_sf is None:
            console.print(f"[yellow]SKIP {label}: LOCOS score file missing[/yellow]")
            continue
        if alpha_sf is None:
            console.print(f"[yellow]SKIP {label}: α-spatial score file missing[/yellow]")
            continue
        models_ok.append(model)
        model_data.append((model, mean_scores(locos_sf), mean_scores(alpha_sf)))

    assert models_ok, "No models with both LOCOS and α-spatial score files. Run inventory.py first."

    n_models = len(models_ok)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows))
    axes_flat = np.array(axes).flatten() if n_models > 1 else [axes]

    j50_values: list[float] = []

    for ax, (model, locos_means, alpha_means) in zip(axes_flat, model_data):
        label = MODEL_LABELS[model]
        all_heads = sorted(set(locos_means) & set(alpha_means))
        assert all_heads, f"No common heads between LOCOS and α-spatial for {label}"

        locos_rank = _rank_vector(locos_means)
        alpha_rank = _rank_vector(alpha_means)

        x = [alpha_rank[h] for h in all_heads]
        y = [locos_rank[h] for h in all_heads]
        n_heads = len(all_heads)

        ax.hexbin(x, y, gridsize=30, cmap="Blues", mincnt=1, linewidths=0.3)
        ax.plot([1, n_heads], [1, n_heads], color="gray", lw=1.0, ls="--", alpha=0.5)
        ax.set_xlabel(r"rank($\alpha$-spatial)", fontsize=10)
        ax.set_ylabel(r"rank(LOCOS)", fontsize=10)
        ax.text(0.05, 0.95, label, transform=ax.transAxes, va="top", ha="left", fontsize=9, fontweight="bold")

        # Jaccard@50 and RBO
        locos_top50 = top_k_heads(locos_means, 50)
        alpha_top50 = top_k_heads(alpha_means, 50)
        j50 = jaccard(locos_top50, alpha_top50)
        j50_values.append(j50)

        locos_ranked_keys = sorted(locos_means, key=lambda h: locos_means[h], reverse=True)
        alpha_ranked_keys = sorted(alpha_means, key=lambda h: alpha_means[h], reverse=True)
        rbo_score = rbo(locos_ranked_keys, alpha_ranked_keys, p=0.9)

        ax.text(
            0.97,
            0.05,
            f"J@50={j50:.2f}\nRBO={rbo_score:.2f}",
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=8,
            bbox=dict(fc="white", ec="none", alpha=0.7),
        )

        rows.append(
            {
                "model": label,
                "n_heads": n_heads,
                "jaccard_at_50": f"{j50:.4f}",
                "rbo_p09": f"{rbo_score:.4f}",
            }
        )

    # Hide unused axes
    for ax in axes_flat[len(models_ok) :]:
        ax.set_visible(False)

    fig.tight_layout()
    save_figure(fig, out_dir / "e1_rankscatter.svg")

    # Write CSV
    csv_path = out_dir / "e1_stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "n_heads", "jaccard_at_50", "rbo_p09"])
        writer.writeheader()
        writer.writerows(rows)

    # Console summary
    table = Table(title="E1 — α-spatial vs LOCOS Rank Agreement")
    table.add_column("Model")
    table.add_column("Heads", justify="right")
    table.add_column("Jaccard@50", justify="right")
    table.add_column("RBO (p=0.9)", justify="right")
    for r in rows:
        table.add_row(r["model"], str(r["n_heads"]), r["jaccard_at_50"], r["rbo_p09"])
    console.print(table)

    median_j50 = float(np.median(j50_values))
    console.print(f"\nMedian Jaccard@50 = [bold]{median_j50:.3f}[/bold]")
    if median_j50 < 0.4:
        console.print("[green]→ E1 DECISION: Proceed to E7 expecting OV separation.[/green]")
    elif median_j50 > 0.7:
        console.print("[red]→ E1 DECISION: Central claim endangered — α-spatial ranks match LOCOS.[/red]")
    else:
        console.print("[yellow]→ E1 DECISION: Ambiguous (0.4–0.7). E7 ablation decides.[/yellow]")

    console.print(f"\n[dim]Saved:[/dim] {out_dir}/e1_rankscatter.svg, e1_stats.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E1 — α-spatial vs LOCOS rank diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--no-download", action="store_true", help="Use local files only.")
    args = parser.parse_args()
    run(download=not args.no_download)


if __name__ == "__main__":
    main()
