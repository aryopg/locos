#!/usr/bin/env python3
"""Compare two retrieval head JSON files.

Compares generated retrieval-head JSON against a reference JSON (e.g., the
original from nightdessert/Retrieval_Head) to validate consistency.

Usage:
    python -m locos.analysis.compare_heads \\
        --reference retrieval_heads/Meta-Llama-3-8B-Instruct.json \\
        --generated retrieval_heads/Meta-Llama-3-8B-Instruct_new.json

    # With custom top-K for ranking comparison
    python -m locos.analysis.compare_heads \\
        --reference retrieval_heads/Meta-Llama-3-8B-Instruct.json \\
        --generated retrieval_heads/Meta-Llama-3-8B-Instruct_new.json \\
        --top-k 50
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from locos_eval.utils.plotting import facecolor_alpha, save_figure, setup_plot_style

console = Console()


def load_scores(path: str) -> dict[str, list[float]]:
    with open(path) as f:
        data = json.load(f)
    # Support envelope format ({"meta": {...}, "scores": {...}})
    # used by CRI and contrastive scoring output.
    if "scores" in data and isinstance(data["scores"], dict):
        return data["scores"]
    return data


def rank_heads(data: dict[str, list[float]]) -> list[tuple[str, float]]:
    """Return heads sorted by mean score descending."""
    scored = [(k, float(np.mean(v)) if v else 0.0) for k, v in data.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def plot_comparison_story(
    ref_ranked: list[tuple[str, float]],
    gen_ranked: list[tuple[str, float]],
    ref_means: dict[str, float],
    gen_means: dict[str, float],
    common: set[str],
    spearman: float | None,
    all_diffs: np.ndarray,
    top_k: int,
    out_path: str,
    scatter_xlabel: str = "Reference mean score",
    scatter_ylabel: str = "Generated mean score",
    overlap_xlabel: str = "k",
    overlap_ylabel: str = "Top-k overlap (%)",
) -> None:
    """Create comparison plots saved as separate subfigures."""
    import matplotlib.pyplot as plt

    setup_plot_style()
    out = Path(out_path)

    # --- Scatter: per-head score agreement ---
    common_sorted = sorted(common)
    ref_vals = np.array([ref_means[k] for k in common_sorted], dtype=float)
    gen_vals = np.array([gen_means[k] for k in common_sorted], dtype=float)

    fig_scatter, ax_scatter = plt.subplots()
    if ref_vals.size > 0:
        ax_scatter.scatter(
            ref_vals,
            gen_vals,
            s=14,
            color=facecolor_alpha("#4C72B0", 0.65),
            edgecolors="black",
            linewidths=0.4,
        )
        lo = float(min(ref_vals.min(), gen_vals.min()))
        hi = float(max(ref_vals.max(), gen_vals.max()))
        if hi > lo:
            ax_scatter.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, linestyle="--")

    ax_scatter.set_xlabel(scatter_xlabel)
    ax_scatter.set_ylabel(scatter_ylabel)
    mae = float(np.mean(all_diffs)) if all_diffs.size > 0 else float("nan")
    rho_txt = "n/a" if spearman is None else f"{spearman:.3f}"
    mae_txt = "n/a" if not np.isfinite(mae) else f"{mae:.4f}"
    ax_scatter.text(
        0.03,
        0.97,
        r"$\rho$ = " + rho_txt + f"\nMAE = {mae_txt}",
        transform=ax_scatter.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    scatter_path = out.with_name(f"{out.stem}_scatter{out.suffix}")
    save_figure(fig_scatter, scatter_path)

    # --- Overlap@k curve ---
    fig_overlap, ax_overlap = plt.subplots()
    max_k = min(top_k, len(ref_ranked), len(gen_ranked))
    if max_k > 0:
        ks = np.arange(1, max_k + 1)
        ref_top = [k for k, _ in ref_ranked[:max_k]]
        gen_top = [k for k, _ in gen_ranked[:max_k]]
        overlaps_pct = np.array(
            [len(set(ref_top[:k]) & set(gen_top[:k])) / k * 100.0 for k in ks],
            dtype=float,
        )
        ax_overlap.plot(
            ks,
            overlaps_pct,
            color="#55A868",
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            markeredgecolor="black",
            markeredgewidth=0.4,
            markerfacecolor=facecolor_alpha("#55A868", 0.8),
        )
        ax_overlap.set_xlim(1, max_k)
    ax_overlap.set_ylim(0, 100)
    ax_overlap.set_xlabel(overlap_xlabel)
    ax_overlap.set_ylabel(overlap_ylabel)

    overlap_path = out.with_name(f"{out.stem}_overlap{out.suffix}")
    save_figure(fig_overlap, overlap_path)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two retrieval head JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference", required=True, help="Reference JSON (ground truth)")
    parser.add_argument("--generated", required=True, help="Generated JSON to validate")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top heads to compare for ranking metrics")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Tolerance for per-head mean score difference")
    parser.add_argument(
        "--plot-out",
        default="figures/retrieval_heads_comparison.svg",
        help="Output path for comparison plot",
    )
    parser.add_argument("--scatter-xlabel", default="Reference mean score", help="X-axis label for scatter plot")
    parser.add_argument("--scatter-ylabel", default="Generated mean score", help="Y-axis label for scatter plot")
    parser.add_argument("--overlap-xlabel", default="k", help="X-axis label for overlap plot")
    parser.add_argument("--overlap-ylabel", default="Top-k overlap (%%)", help="Y-axis label for overlap plot")
    args = parser.parse_args()

    console.rule("[bold]Retrieval Head Comparison[/bold]")

    ref_data = load_scores(args.reference)
    gen_data = load_scores(args.generated)

    console.print(f"Reference: {args.reference} ({len(ref_data)} heads)")
    console.print(f"Generated: {args.generated} ({len(gen_data)} heads)")

    # -----------------------------------------------------------------------
    # 1. Structural comparison
    # -----------------------------------------------------------------------
    console.print("\n[bold]1. Structure[/bold]")

    ref_keys = set(ref_data.keys())
    gen_keys = set(gen_data.keys())

    missing = ref_keys - gen_keys
    extra = gen_keys - ref_keys
    common = ref_keys & gen_keys

    if missing:
        console.print(f"  [red]Missing heads in generated: {len(missing)}[/red]")
        if len(missing) <= 10:
            console.print(f"    {sorted(missing)}")
    if extra:
        console.print(f"  [yellow]Extra heads in generated: {len(extra)}[/yellow]")
        if len(extra) <= 10:
            console.print(f"    {sorted(extra)}")
    if not missing and not extra:
        console.print(f"  [green]All {len(ref_keys)} heads present in both files[/green]")

    # Score list lengths
    ref_lengths = set(len(v) for v in ref_data.values())
    gen_lengths = set(len(v) for v in gen_data.values())
    console.print(f"  Reference score lengths: {ref_lengths}")
    console.print(f"  Generated score lengths: {gen_lengths}")

    # -----------------------------------------------------------------------
    # 2. Ranking comparison
    # -----------------------------------------------------------------------
    console.print(f"\n[bold]2. Top-{args.top_k} Ranking[/bold]")

    ref_ranked = rank_heads(ref_data)
    gen_ranked = rank_heads(gen_data)

    ref_top_keys = [k for k, _ in ref_ranked[: args.top_k]]
    gen_top_keys = [k for k, _ in gen_ranked[: args.top_k]]

    ref_top_set = set(ref_top_keys)
    gen_top_set = set(gen_top_keys)

    overlap = ref_top_set & gen_top_set
    overlap_denom = max(1, min(args.top_k, len(ref_ranked), len(gen_ranked)))
    overlap_pct = len(overlap) / overlap_denom * 100

    console.print(f"  Overlap: {len(overlap)}/{args.top_k} ({overlap_pct:.1f}%)")

    # Rank correlation (Spearman) over top-K heads only.
    # All-heads Spearman is misleadingly high because retrieval heads are
    # sparse: the vast majority of heads have near-zero scores in both files,
    # so zeros agreeing with zeros inflates the correlation.
    spearman = None
    if common:
        ref_rank_map = {k: i for i, (k, _) in enumerate(ref_ranked)}
        gen_rank_map = {k: i for i, (k, _) in enumerate(gen_ranked)}

        # Top-K Spearman (primary metric)
        # Compute Spearman on mean scores of the common heads within the
        # reference top-K. Uses scipy.stats.spearmanr which returns a
        # bounded [-1, 1] correlation coefficient.
        from scipy.stats import spearmanr as _spearmanr

        ref_mean_map = dict(ref_ranked)
        gen_mean_map = dict(gen_ranked)
        top_common = [k for k in ref_top_keys if k in gen_mean_map]
        if len(top_common) >= 2:
            ref_scores_common = np.array([ref_mean_map[k] for k in top_common])
            gen_scores_common = np.array([gen_mean_map[k] for k in top_common])
            spearman, _p = _spearmanr(ref_scores_common, gen_scores_common)
            console.print(f"  Spearman rank correlation (top-{args.top_k}): {spearman:.4f}")
        else:
            console.print(f"  Spearman rank correlation (top-{args.top_k}): n/a (need >=2 common heads)")

    # -----------------------------------------------------------------------
    # 3. Per-head score comparison
    # -----------------------------------------------------------------------
    console.print("\n[bold]3. Per-Head Mean Score Differences[/bold]")

    ref_means = {k: np.mean(v) if v else 0.0 for k, v in ref_data.items()}
    gen_means = {k: np.mean(v) if v else 0.0 for k, v in gen_data.items()}

    diffs = []
    for k in common:
        diff = abs(ref_means[k] - gen_means[k])
        diffs.append((k, ref_means[k], gen_means[k], diff))

    diffs.sort(key=lambda x: x[3], reverse=True)
    all_diffs = np.array([d[3] for d in diffs], dtype=float)
    if all_diffs.size > 0:
        console.print(f"  Mean absolute difference: {np.mean(all_diffs):.6f}")
        console.print(f"  Max absolute difference:  {np.max(all_diffs):.6f}")
        console.print(f"  Median absolute difference: {np.median(all_diffs):.6f}")

        within_tol = np.sum(all_diffs <= args.tolerance)
        console.print(
            f"  Within tolerance ({args.tolerance}): "
            f"{within_tol}/{len(all_diffs)} ({within_tol/len(all_diffs)*100:.1f}%)"
        )
    else:
        console.print("  No common heads available for score-difference statistics.")

    # -----------------------------------------------------------------------
    # 4. Side-by-side top heads table
    # -----------------------------------------------------------------------
    console.print(f"\n[bold]4. Top-{min(args.top_k, 30)} Heads Side-by-Side[/bold]")

    table = Table()
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Reference", width=8)
    table.add_column("Ref Score", justify="right", width=10)
    table.add_column("Generated", width=8)
    table.add_column("Gen Score", justify="right", width=10)
    table.add_column("Match", width=5)

    show_n = min(args.top_k, 30)
    for i in range(show_n):
        ref_key, ref_score = ref_ranked[i] if i < len(ref_ranked) else ("—", 0)
        gen_key, gen_score = gen_ranked[i] if i < len(gen_ranked) else ("—", 0)
        match = "[green]Y[/green]" if ref_key == gen_key else "[red]N[/red]"
        table.add_row(
            str(i + 1),
            str(ref_key),
            f"{ref_score:.4f}",
            str(gen_key),
            f"{gen_score:.4f}",
            match,
        )
    console.print(table)

    # -----------------------------------------------------------------------
    # 5. Largest disagreements table
    # -----------------------------------------------------------------------
    console.print("\n[bold]5. Largest Score Disagreements[/bold]")

    table2 = Table()
    table2.add_column("Head", width=8)
    table2.add_column("Ref Mean", justify="right", width=10)
    table2.add_column("Gen Mean", justify="right", width=10)
    table2.add_column("Abs Diff", justify="right", width=10)
    table2.add_column("Ref Rank", justify="right", width=8)
    table2.add_column("Gen Rank", justify="right", width=8)

    ref_rank_map = {k: i + 1 for i, (k, _) in enumerate(ref_ranked)}
    gen_rank_map = {k: i + 1 for i, (k, _) in enumerate(gen_ranked)}

    for k, ref_m, gen_m, diff in diffs[:15]:
        style = "red" if diff > args.tolerance else ""
        table2.add_row(
            k,
            f"{ref_m:.4f}",
            f"{gen_m:.4f}",
            f"{diff:.4f}",
            str(ref_rank_map.get(k, "—")),
            str(gen_rank_map.get(k, "—")),
            style=style,
        )
    console.print(table2)

    # -----------------------------------------------------------------------
    # 6. Plot summary
    # -----------------------------------------------------------------------
    plot_comparison_story(
        ref_ranked=ref_ranked,
        gen_ranked=gen_ranked,
        ref_means=ref_means,
        gen_means=gen_means,
        common=common,
        spearman=spearman,
        all_diffs=all_diffs,
        top_k=args.top_k,
        out_path=args.plot_out,
        scatter_xlabel=args.scatter_xlabel,
        scatter_ylabel=args.scatter_ylabel,
        overlap_xlabel=args.overlap_xlabel,
        overlap_ylabel=args.overlap_ylabel,
    )
    out = Path(args.plot_out)
    console.print(f"\nPlots saved to {out.parent / out.stem}_scatter/overlap{out.suffix}")

    # -----------------------------------------------------------------------
    # Summary verdict
    # -----------------------------------------------------------------------
    console.print()
    issues = []
    if missing:
        issues.append(f"{len(missing)} missing heads")
    if overlap_pct < 80:
        issues.append(f"low top-{args.top_k} overlap ({overlap_pct:.0f}%)")
    if spearman is not None and spearman < 0.8:
        issues.append(f"low top-{args.top_k} Spearman correlation ({spearman:.3f})")
    if all_diffs.size > 0 and np.mean(all_diffs) > args.tolerance:
        issues.append(f"high mean score diff ({np.mean(all_diffs):.4f})")

    mean_diff_text = f"{np.mean(all_diffs):.6f}" if all_diffs.size > 0 else "n/a"
    spearman_text = f"{spearman:.4f}" if spearman is not None else "n/a"

    if not issues:
        console.print(
            Panel(
                f"[bold green]PASS[/bold green] — Rankings and scores are consistent.\n"
                f"Top-{args.top_k} overlap: {overlap_pct:.0f}%, "
                f"Spearman: {spearman_text}, "
                f"Mean diff: {mean_diff_text}",
                title="Verdict",
            )
        )
    else:
        console.print(
            Panel(
                "[bold yellow]DIFFERENCES FOUND[/bold yellow]\n"
                + "\n".join(f"  - {i}" for i in issues)
                + "\n\nNote: Some difference is expected due to non-determinism in "
                "attention patterns, different tokenizer behavior, and haystack "
                "content differences.",
                title="Verdict",
            )
        )


if __name__ == "__main__":
    main()
