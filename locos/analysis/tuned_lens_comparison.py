#!/usr/bin/env python3
"""Compare standard vs tuned-lens-corrected logit-contribution scores.

Usage:
    python -m locos.analysis.tuned_lens_comparison \
        --standard retrieval_heads/gemma-3-27b-it_logit_contrib_nolima.json \
        --corrected retrieval_heads/gemma-3-27b-it_logit_contrib_nolima_tuned_lens.json \
        --top-k 50 \
        --output figures/tuned_lens_comparison.svg

    # With debug output (score distributions, largest rank shifts, per-head details):
    python -m locos.analysis.tuned_lens_comparison \
        --standard ... --corrected ... --top-k 50 --debug
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from locos_eval.utils.plotting import save_figure, setup_plot_style

console = Console()


def load_scores(path: str) -> tuple[dict[str, float], dict]:
    """Load JSON and compute per-head mean scores. Return (scores, metadata)."""
    with open(path) as f:
        data = json.load(f)
    meta = data.get("meta", {})
    raw_scores = data.get("scores", data)
    scores = {k: float(np.mean(v)) if v else 0.0 for k, v in raw_scores.items()}
    return scores, meta


def scores_to_grid(scores: dict[str, float]) -> np.ndarray:
    """Convert {layer-head: score} to (num_layers, num_heads) array."""
    max_l, max_h = 0, 0
    for k in scores:
        l, h = map(int, k.split("-"))
        max_l, max_h = max(max_l, l), max(max_h, h)
    grid = np.zeros((max_l + 1, max_h + 1))
    for k, v in scores.items():
        l, h = map(int, k.split("-"))
        grid[l, h] = v
    return grid


def layer_quartile(layer: int, num_layers: int) -> int:
    """Return quartile index (0-3) for a given layer."""
    return min(int(layer / num_layers * 4), 3)


def quartile_label(q: int) -> str:
    return ["Q1 (early)", "Q2 (mid-early)", "Q3 (mid-late)", "Q4 (late)"][q]


def main():
    parser = argparse.ArgumentParser(description="Tuned-lens bias comparison")
    parser.add_argument("--standard", required=True, help="Standard logit-contrib JSON")
    parser.add_argument("--corrected", required=True, help="Tuned-lens-corrected JSON")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed diagnostics: file metadata, score distributions, "
        "per-layer stats, largest rank shifts, and per-head top-k comparison.",
    )
    args = parser.parse_args()

    std_scores, std_meta = load_scores(args.standard)
    tl_scores, tl_meta = load_scores(args.corrected)

    # --- Debug: file metadata ---
    if args.debug:
        console.rule("[dim]Debug: File Metadata[/dim]")
        for label, meta, path in [("Standard", std_meta, args.standard), ("Corrected", tl_meta, args.corrected)]:
            console.print(f"[bold]{label}:[/bold] {path}")
            if meta:
                for k, v in meta.items():
                    console.print(f"  {k}: {v}")
            else:
                console.print("  [dim](no metadata)[/dim]")
        console.print()

    # Ensure same head set
    common_keys = sorted(set(std_scores) & set(tl_scores))
    assert len(common_keys) == len(std_scores) == len(tl_scores), "Head sets differ"

    std_vals = np.array([std_scores[k] for k in common_keys])
    tl_vals = np.array([tl_scores[k] for k in common_keys])

    std_grid = scores_to_grid(std_scores)
    num_layers, _num_heads = std_grid.shape

    # --- Debug: score distributions ---
    if args.debug:
        console.rule("[dim]Debug: Score Distributions[/dim]")
        dist_table = Table(show_header=True)
        dist_table.add_column("Statistic")
        dist_table.add_column("Standard", justify="right")
        dist_table.add_column("Tuned-Lens", justify="right")
        dist_table.add_column("Delta (TL - Std)", justify="right")
        for name, fn in [("min", np.min), ("max", np.max), ("mean", np.mean), ("median", np.median), ("std", np.std)]:
            s, t = fn(std_vals), fn(tl_vals)
            dist_table.add_row(name, f"{s:.6f}", f"{t:.6f}", f"{t - s:+.6f}")
        dist_table.add_row("num_heads", str(len(common_keys)), str(len(common_keys)), "")
        dist_table.add_row("num_layers", str(num_layers), str(num_layers), "")
        console.print(dist_table)

        # Per-layer mean scores
        tl_grid = scores_to_grid(tl_scores)
        console.print()
        layer_table = Table(title="Per-Layer Mean Scores", show_header=True)
        layer_table.add_column("Layer")
        layer_table.add_column("Std mean", justify="right")
        layer_table.add_column("TL mean", justify="right")
        layer_table.add_column("Delta", justify="right")
        layer_table.add_column("Q")
        for l_idx in range(num_layers):
            s_mean = float(std_grid[l_idx].mean())
            t_mean = float(tl_grid[l_idx].mean())
            delta = t_mean - s_mean
            q = quartile_label(layer_quartile(l_idx, num_layers))
            layer_table.add_row(str(l_idx), f"{s_mean:.6f}", f"{t_mean:.6f}", f"{delta:+.6f}", q)
        console.print(layer_table)
        console.print()

    # --- Rank correlation ---
    tau_global, p_global = stats.kendalltau(std_vals, tl_vals)

    # Per-quartile rank correlation
    quartile_taus = {}
    for q in range(4):
        mask = [layer_quartile(int(k.split("-")[0]), num_layers) == q for k in common_keys]
        if sum(mask) > 1:
            s = std_vals[mask]
            t = tl_vals[mask]
            tau_q, _ = stats.kendalltau(s, t)
            quartile_taus[q] = tau_q

    # --- Top-k overlap ---
    std_ranked = sorted(common_keys, key=lambda k: std_scores[k], reverse=True)
    tl_ranked = sorted(common_keys, key=lambda k: tl_scores[k], reverse=True)
    std_topk = set(std_ranked[: args.top_k])
    tl_topk = set(tl_ranked[: args.top_k])
    overlap = std_topk & tl_topk
    jaccard = len(overlap) / len(std_topk | tl_topk)

    # --- Layer distribution of top-k ---
    std_quartile_counts = [0, 0, 0, 0]
    tl_quartile_counts = [0, 0, 0, 0]
    for k in std_topk:
        q = layer_quartile(int(k.split("-")[0]), num_layers)
        std_quartile_counts[q] += 1
    for k in tl_topk:
        q = layer_quartile(int(k.split("-")[0]), num_layers)
        tl_quartile_counts[q] += 1

    # --- Print results ---
    console.rule("[bold]Tuned-Lens Direct-Path Bias Analysis[/bold]")

    table = Table(title="Rank Correlation (Kendall's tau)")
    table.add_column("Scope")
    table.add_column("tau", justify="right")
    table.add_row("Global (all heads)", f"{tau_global:.4f} (p={p_global:.2e})")
    for q, tau in quartile_taus.items():
        table.add_row(quartile_label(q), f"{tau:.4f}")
    console.print(table)

    table2 = Table(title=f"Top-{args.top_k} Head Overlap")
    table2.add_column("Metric", style="bold")
    table2.add_column("Value", justify="right")
    table2.add_row("Overlap", f"{len(overlap)}/{args.top_k}")
    table2.add_row("Jaccard", f"{jaccard:.3f}")
    console.print(table2)

    table3 = Table(title=f"Layer Quartile Distribution of Top-{args.top_k}")
    table3.add_column("Quartile")
    table3.add_column("Standard", justify="right")
    table3.add_column("Tuned-Lens", justify="right")
    table3.add_column("Delta", justify="right")
    for q in range(4):
        delta = tl_quartile_counts[q] - std_quartile_counts[q]
        sign = "+" if delta > 0 else ""
        table3.add_row(
            quartile_label(q),
            str(std_quartile_counts[q]),
            str(tl_quartile_counts[q]),
            f"{sign}{delta}",
        )
    console.print(table3)

    # --- Debug: top-k head details and rank shifts ---
    if args.debug:
        console.print()
        console.rule("[dim]Debug: Top-k Head Comparison[/dim]")
        std_rank_map = {k: i for i, k in enumerate(std_ranked)}
        tl_rank_map = {k: i for i, k in enumerate(tl_ranked)}

        head_table = Table(title=f"Top-{args.top_k} Heads (by standard ranking)")
        head_table.add_column("Std Rank", justify="right")
        head_table.add_column("Head")
        head_table.add_column("Std Score", justify="right")
        head_table.add_column("TL Score", justify="right")
        head_table.add_column("TL Rank", justify="right")
        head_table.add_column("Rank Shift", justify="right")
        head_table.add_column("In TL top-k?")
        for i, k in enumerate(std_ranked[: args.top_k]):
            tl_rank = tl_rank_map[k]
            shift = tl_rank - i
            in_tl = "yes" if k in tl_topk else f"[red]no (#{tl_rank})[/red]"
            head_table.add_row(
                str(i),
                k,
                f"{std_scores[k]:.6f}",
                f"{tl_scores[k]:.6f}",
                str(tl_rank),
                f"{shift:+d}",
                in_tl,
            )
        console.print(head_table)

        # Heads promoted into TL top-k but not in standard top-k
        promoted = tl_topk - std_topk
        if promoted:
            console.print()
            promo_table = Table(title=f"Heads promoted into TL top-{args.top_k} (not in standard top-{args.top_k})")
            promo_table.add_column("Head")
            promo_table.add_column("TL Rank", justify="right")
            promo_table.add_column("Std Rank", justify="right")
            promo_table.add_column("TL Score", justify="right")
            promo_table.add_column("Std Score", justify="right")
            for k in sorted(promoted, key=lambda k: tl_rank_map[k]):
                promo_table.add_row(
                    k,
                    str(tl_rank_map[k]),
                    str(std_rank_map[k]),
                    f"{tl_scores[k]:.6f}",
                    f"{std_scores[k]:.6f}",
                )
            console.print(promo_table)

        # Largest absolute rank shifts across ALL heads
        console.print()
        all_shifts = [(k, tl_rank_map[k] - std_rank_map[k]) for k in common_keys]
        all_shifts.sort(key=lambda x: abs(x[1]), reverse=True)
        shift_table = Table(title="Top 20 Largest Rank Shifts (any head)")
        shift_table.add_column("Head")
        shift_table.add_column("Std Rank", justify="right")
        shift_table.add_column("TL Rank", justify="right")
        shift_table.add_column("Shift", justify="right")
        shift_table.add_column("Direction")
        for k, shift in all_shifts[:20]:
            direction = "[green]promoted[/green]" if shift < 0 else "[red]demoted[/red]"
            shift_table.add_row(k, str(std_rank_map[k]), str(tl_rank_map[k]), f"{shift:+d}", direction)
        console.print(shift_table)

    # --- Interpretation ---
    if tau_global > 0.8 and jaccard > 0.7:
        console.print(
            "\n[green]Interpretation:[/green] High rank correlation and top-k overlap. "
            "The direct-path bias does not meaningfully change head selection. "
            "Late-layer concentration is genuine."
        )
    elif tl_quartile_counts[0] + tl_quartile_counts[1] > std_quartile_counts[0] + std_quartile_counts[1] + 5:
        console.print(
            "\n[yellow]Interpretation:[/yellow] Tuned-lens correction promotes early/mid-layer "
            "heads into the top-k. The direct-path bias may hide important retrieval heads."
        )
    else:
        console.print("\n[cyan]Interpretation:[/cyan] Mixed results. Check the plots for detailed picture.")

    # --- Plot ---
    if args.output:
        import matplotlib.pyplot as plt

        setup_plot_style()
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Panel 1: Scatter of mean scores
        ax = axes[0]
        colors = [layer_quartile(int(k.split("-")[0]), num_layers) for k in common_keys]
        scatter = ax.scatter(std_vals, tl_vals, c=colors, cmap="viridis", alpha=0.5, s=10, vmin=0, vmax=3)
        lims = [min(std_vals.min(), tl_vals.min()), max(std_vals.max(), tl_vals.max())]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel(r"Standard $S^{\tau}$")
        ax.set_ylabel(r"Tuned-lens $S^{\tau}_{\mathrm{TL}}$")
        cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(["Q1", "Q2", "Q3", "Q4"])

        # Panel 2: Layer distribution histogram
        ax = axes[1]
        x = np.arange(4)
        width = 0.35
        ax.bar(x - width / 2, std_quartile_counts, width, label="Standard")
        ax.bar(x + width / 2, tl_quartile_counts, width, label="Tuned-lens")
        ax.set_xticks(x)
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_ylabel(f"Count in top-{args.top_k}")
        ax.legend()

        # Panel 3: Rank difference for top-k heads
        ax = axes[2]
        std_rank_map = {k: i for i, k in enumerate(std_ranked)}
        tl_rank_map = {k: i for i, k in enumerate(tl_ranked)}
        rank_diffs = [tl_rank_map[k] - std_rank_map[k] for k in std_ranked[: args.top_k]]
        ax.barh(range(args.top_k), rank_diffs, height=0.8, color="steelblue")
        ax.set_xlabel("Rank shift (positive = demoted by TL)")
        ax.set_ylabel(f"Head (standard top-{args.top_k} order)")
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.5)

        fig.tight_layout()
        save_figure(fig, args.output)
        console.print(f"\n[bold green]Saved figure to {args.output}[/bold green]")


if __name__ == "__main__":
    main()
