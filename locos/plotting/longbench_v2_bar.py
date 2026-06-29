#!/usr/bin/env python3
"""LongBench-v2 accuracy bar charts across decoding variants.

Two kinds of figures:
- Per-model: one SVG per model under ``<out_dir>/<ModelPretty>.svg``, showing
  ``accuracy_compensated`` per domain.
- Overall: one cross-model SVG (``Overall.svg``) with x = model, hue = variant.

In both, the ``Long Structured Data Understanding`` domain is dropped (n=4 is
too small to be informative). Bars are hued by decoding variant
(Greedy / Ablate LOCOS (NoLiMa) / Ablate Wu (NIAH) / Ablate random) and error
bars show ±1 std across seeds.

Discovery and aggregation are shared with ``longbench_v2_radar.py``.

Usage:
    python locos/plotting/longbench_v2_bar.py \
        --results-root ../locos-results/downstream_results \
        --task longbench_v2_short \
        --out-dir figures/longbench_v2_bar
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console

from locos.plotting._paths import default_downstream_results_root
from locos.plotting.longbench_v2_radar import (
    METRIC,
    MODEL_ORDER,
    VARIANT_FAMILIES,
    discover,
    domain_sample_counts,
    load_results_jsonl,
)
from locos_eval.utils.plotting import (
    LINE_WIDTH,
    MODEL_PRETTY_NAMES,
    facecolor_alpha,
    save_figure,
    setup_plot_style,
)

console = Console()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


_DOMAIN_SHORT = {
    "Single-Document QA": "Single-Doc QA",
    "Multi-Document QA": "Multi-Doc QA",
    "Long-dialogue History Understanding": "Long-Dialogue",
    "Long Structured Data Understanding": "Long Structured",
    "Code Repository Understanding": "Code Repo",
    "Long In-context Learning": "Long ICL",
}

# Domains excluded from both per-model figures and the Overall computation.
# Long Structured has only n=4 in LongBench-v2 short, far too few for
# stable per-seed accuracy.
EXCLUDED_DOMAINS: set[str] = {"Long Structured Data Understanding"}


def build_long_form(
    discovered: dict[str, dict[str, list[tuple[int, dict[str, float]]]]],
    overall: dict[str, dict[str, list[tuple[int, float]]]],
) -> pd.DataFrame:
    """One row per (model, variant, seed, domain). The ``Overall`` group is
    treated as a synthetic domain so it can be plotted alongside the rest."""
    records = []
    for model, by_family in discovered.items():
        for family, seed_results in by_family.items():
            for seed, scores in seed_results:
                for domain, acc in scores.items():
                    records.append(
                        {
                            "model": model,
                            "variant": family,
                            "seed": seed,
                            "domain": domain,
                            "accuracy_compensated": acc,
                        }
                    )
    for model, by_family in overall.items():
        for family, seed_scores in by_family.items():
            for seed, acc in seed_scores:
                records.append(
                    {
                        "model": model,
                        "variant": family,
                        "seed": seed,
                        "domain": "Overall",
                        "accuracy_compensated": acc,
                    }
                )
    return pd.DataFrame.from_records(records)


def overall_per_seed(
    results_root: Path, task: str
) -> tuple[dict[str, dict[str, list[tuple[int, float]]]], dict[str, int]]:
    """Compute per-seed overall accuracy (mean of ``accuracy_compensated``
    across all scored samples) for each (model, variant). Also returns
    domain sample counts gathered from the first scored variant per model."""
    from locos.plotting.longbench_v2_radar import (
        family_to_canonical,
        parse_variant_dir,
    )

    overall: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    counts: dict[str, int] = {}

    task_dir = results_root / task
    for model_dir in sorted(task_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        short = model_dir.name.split("_", 1)[-1]
        for variant_dir in sorted(model_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            parsed = parse_variant_dir(variant_dir.name)
            if parsed is None:
                continue
            family, seed = parsed
            canonical = family_to_canonical(family)
            if canonical is None:
                continue
            rows = load_results_jsonl(variant_dir)
            if not rows:
                continue
            kept = [
                r
                for r in rows
                if METRIC in r.get("scores", {}) and r.get("metadata", {}).get("domain") not in EXCLUDED_DOMAINS
            ]
            if not kept:
                continue
            scores = [r["scores"][METRIC] for r in kept]
            overall[short][canonical].append((seed, float(np.mean(scores))))
            if not counts:
                counts = domain_sample_counts(rows)
    assert counts, "Failed to collect domain sample counts"
    return overall, counts


def aggregate(long_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        long_df.groupby(["model", "variant", "domain"])["accuracy_compensated"]
        .agg(["mean", "std", "min", "max", "count"])
        .rename(columns={"count": "n_seeds"})
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _compute_ytop(df: pd.DataFrame) -> float:
    """Round mean+std up to next 0.05 step, with a small headroom margin."""
    if df.empty:
        return 1.0
    upper = (df["mean"].fillna(0.0) + df["std"].fillna(0.0)).max()
    raw = float(upper) + 0.05
    step = 0.05
    return min(1.0, max(0.3, np.ceil(raw / step) * step))


def _yticks(y_top: float) -> np.ndarray:
    """Pick reasonable y-tick spacing for the chosen y_top."""
    spacing = 0.1 if y_top <= 0.6 else 0.2
    return np.arange(0.0, y_top + 1e-9, spacing)


def make_overall_figure(
    summary: pd.DataFrame,
    model_order: list[str],
    total_samples: int,
    out_path: Path,
) -> None:
    """Cross-model overall-accuracy figure. x-axis = models, hue = variant."""
    setup_plot_style()
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"

    overall = summary[summary["domain"] == "Overall"]
    present_models = [m for m in model_order if m in overall["model"].unique()]
    if not present_models:
        console.print("[yellow]No overall data — skipping overall figure[/yellow]")
        return

    n_groups = len(present_models)
    variants = list(VARIANT_FAMILIES)
    n_variants = len(variants)
    bar_width = 0.8 / n_variants
    x_centers = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(10, 1.5 * n_groups + 1.5), 3))

    for v_idx, (key, label, color) in enumerate(variants):
        means = []
        stds = []
        n_seeds_arr = []
        for model in present_models:
            row = overall[(overall["model"] == model) & (overall["variant"] == key)]
            if row.empty:
                means.append(np.nan)
                stds.append(0.0)
                n_seeds_arr.append(0)
            else:
                means.append(float(row["mean"].iloc[0]))
                stds.append(float(row["std"].iloc[0]))
                n_seeds_arr.append(int(row["n_seeds"].iloc[0]))
        means_arr = np.array(means, dtype=float)
        stds_arr = np.array(stds, dtype=float)
        n_seeds_np = np.array(n_seeds_arr, dtype=int)
        means_plot = np.where(n_seeds_np > 0, means_arr, 0.0)
        stds_plot = np.where(n_seeds_np > 1, stds_arr, 0.0)

        offsets = (v_idx - (n_variants - 1) / 2) * bar_width
        bars = ax.bar(
            x_centers + offsets,
            means_plot,
            width=bar_width,
            color=facecolor_alpha(color, 0.85),
            edgecolor="black",
            linewidth=LINE_WIDTH * 0.5,
            label=label,
            zorder=2,
        )
        for bar, has_data in zip(bars, n_seeds_np > 0):
            if not has_data:
                bar.set_visible(False)
        if np.any(stds_plot > 0):
            ax.errorbar(
                x_centers + offsets,
                means_plot,
                yerr=stds_plot,
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=2.5,
                zorder=3,
            )

    pretty_labels = [MODEL_PRETTY_NAMES.get(m, m) for m in present_models]
    ax.set_xticks(x_centers)
    ax.set_xticklabels(pretty_labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Overall accuracy (compensated)")
    y_top = _compute_ytop(overall)
    ax.set_ylim(0.0, y_top)
    ax.set_yticks(_yticks(y_top))
    ax.axhline(0.25, color="black", linestyle=":", linewidth=1.0, alpha=0.4, zorder=1)
    ax.grid(False)
    ax.set_axisbelow(True)

    ax.set_xlabel(f"Model (LongBench-v2 short, n={total_samples})")
    ax.legend(loc="upper right", ncol=n_variants, fontsize=9, frameon=False)

    fig.tight_layout()
    save_figure(fig, out_path)
    console.print(f"[green]Saved:[/green] {out_path}")


def make_bar_figure(
    model: str,
    summary: pd.DataFrame,
    domain_order: list[str],
    domain_counts: dict[str, int],
    out_path: Path,
) -> None:
    """One bar chart for one model. Groups = domains (incl. Overall),
    hue = decoding variant. Error bars = ±1 std across seeds."""
    setup_plot_style()
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"

    model_data = summary[summary["model"] == model]
    if model_data.empty:
        console.print(f"[yellow]No data for {model}, skipping[/yellow]")
        return

    n_groups = len(domain_order)
    variants = [(k, lab, col) for (k, lab, col) in VARIANT_FAMILIES]
    n_variants = len(variants)
    bar_width = 0.8 / n_variants
    x_centers = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(10, 1.5 * n_groups + 1.5), 3))

    for v_idx, (key, label, color) in enumerate(variants):
        sub = model_data[model_data["variant"] == key].set_index("domain")
        means = np.array([sub.loc[d, "mean"] if d in sub.index else np.nan for d in domain_order], dtype=float)
        stds = np.array([sub.loc[d, "std"] if d in sub.index else 0.0 for d in domain_order], dtype=float)
        n_seeds = np.array([int(sub.loc[d, "n_seeds"]) if d in sub.index else 0 for d in domain_order], dtype=int)
        # Hide bars where the variant is missing entirely (n_seeds==0).
        means_plot = np.where(n_seeds > 0, means, 0.0)
        stds_plot = np.where(n_seeds > 1, stds, 0.0)

        offsets = (v_idx - (n_variants - 1) / 2) * bar_width
        bars = ax.bar(
            x_centers + offsets,
            means_plot,
            width=bar_width,
            color=facecolor_alpha(color, 0.85),
            edgecolor="black",
            linewidth=LINE_WIDTH * 0.5,
            label=label,
            zorder=2,
        )
        # Hide bars with no data (avoid drawing a 0.0 spike).
        for bar, has_data in zip(bars, n_seeds > 0):
            if not has_data:
                bar.set_visible(False)
        if np.any(stds_plot > 0):
            ax.errorbar(
                x_centers + offsets,
                means_plot,
                yerr=stds_plot,
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=2.5,
                zorder=3,
            )

    # Axis labels with sample counts (Overall has a separate count = total).
    overall_n = sum(domain_counts.values())
    xtick_labels = []
    for d in domain_order:
        n = overall_n if d == "Overall" else domain_counts.get(d, 0)
        short = "Overall" if d == "Overall" else _DOMAIN_SHORT.get(d, d)
        xtick_labels.append(f"{short}\n(n={n})")
    ax.set_xticks(x_centers)
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_ylabel("Accuracy (compensated)")
    y_top = _compute_ytop(model_data)
    ax.set_ylim(0.0, y_top)
    ax.set_yticks(_yticks(y_top))
    ax.axhline(0.25, color="black", linestyle=":", linewidth=1.0, alpha=0.4, zorder=1)  # 4-way random chance
    ax.grid(False)
    ax.set_axisbelow(True)

    # Title with model name (separate file per model — title aids identification).
    pretty = MODEL_PRETTY_NAMES.get(model, model)
    ax.set_title(pretty, fontsize=12, pad=6)

    ax.legend(loc="upper right", ncol=n_variants, fontsize=9, frameon=False)

    fig.tight_layout()
    save_figure(fig, out_path, keep_title=True)
    console.print(f"[green]Saved:[/green] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-model bar charts of LongBench-v2 accuracy by domain (incl. Overall)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_downstream_results_root(),
    )
    parser.add_argument("--task", type=str, default="longbench_v2_short")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/longbench_v2_bar"),
        help="Directory to write per-model SVGs into.",
    )
    args = parser.parse_args()

    discovered = discover(args.results_root, args.task)
    overall, counts = overall_per_seed(args.results_root, args.task)
    long_df = build_long_form(discovered, overall)
    summary = aggregate(long_df)

    # Drop very-small-N domains where per-seed accuracy swings are dominated
    # by single-example noise (see ``EXCLUDED_DOMAINS``).
    plot_domains = [d for d in counts if d not in EXCLUDED_DOMAINS]
    # Per-model figures: domains only (Overall lives in its own cross-model figure).
    domain_order = sorted(plot_domains, key=lambda d: -counts[d])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "summary.csv"
    summary.to_csv(csv_path, index=False)
    console.print(f"[green]Saved CSV:[/green] {csv_path}")

    for model in MODEL_ORDER:
        if model not in summary["model"].unique():
            console.print(f"[yellow]No data for {model} — skipping[/yellow]")
            continue
        out_path = args.out_dir / f"{MODEL_PRETTY_NAMES.get(model, model)}.svg"
        make_bar_figure(model, summary, domain_order, counts, out_path)

    total_samples = sum(n for d, n in counts.items() if d not in EXCLUDED_DOMAINS)
    make_overall_figure(summary, MODEL_ORDER, total_samples, args.out_dir / "Overall.svg")


if __name__ == "__main__":
    main()
