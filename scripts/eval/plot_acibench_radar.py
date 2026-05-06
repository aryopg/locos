#!/usr/bin/env python3
"""Radar plot comparing decoding methods on ACI-Bench metrics.

Discovers results from the structured output directory layout produced by
EvalRunner (``{output_dir}/aci_bench/{model}/``).  Generates one radar plot
per model, with legend saved as a separate file.

Usage:
    # Auto-discover all models under eval_results/aci_bench/
    python scripts/eval/plot_acibench_radar.py --results-dir eval_results

    # Explicit per-file mode (legacy)
    python scripts/eval/plot_acibench_radar.py \
        --decore results_decore.jsonl \
        --greedy results_greedy.jsonl \
        --out figures/acibench_radar.svg

    # With custom labels
    python scripts/eval/plot_acibench_radar.py \
        --results-dir eval_results --labels "DeCoRe" "Greedy"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from locos_eval.utils.plotting import (
    facecolor_alpha,
    save_figure,
    save_legend,
    setup_plot_style,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_mean_scores(path: str | Path) -> dict[str, float]:
    """Load a results JSONL and return mean of each metric."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) > 0, f"No records in {path}"

    all_keys: set[str] = set()
    for r in records:
        all_keys.update(r["scores"].keys())

    means: dict[str, float] = {}
    for key in all_keys:
        vals = [r["scores"][key] for r in records if key in r["scores"] and r["scores"][key] >= 0]
        if vals:
            means[key] = sum(vals) / len(vals)

    return means


def discover_model_results(results_dir: Path) -> dict[str, dict[str, Path]]:
    """Discover results per model from ``{results_dir}/aci_bench/{model}/{variant}/``.

    Supports both the new layout (``{model}/{variant}/results_*.jsonl``) and
    the legacy flat layout (``{model}/{decoding}_*.jsonl``).

    Returns:
        ``{model_name: {"greedy": Path, "decore_niah": Path, ...}}``
    """
    task_dir = results_dir / "aci_bench"
    assert task_dir.is_dir(), f"No aci_bench directory found at {task_dir}"

    models: dict[str, dict[str, Path]] = {}
    for model_dir in sorted(task_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        variant_files: dict[str, Path] = {}

        for sub in sorted(model_dir.iterdir()):
            if sub.is_dir():
                # New layout: {model}/{variant}/results_*.jsonl
                result_files = sorted(
                    sub.glob("results_*.jsonl"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if result_files:
                    variant_files[sub.name] = result_files[0]
            elif sub.suffix == ".jsonl" and not sub.name.endswith("_generations.jsonl"):
                # Legacy flat layout: {model}/{decoding}_{timestamp}.jsonl
                stem = sub.stem
                decoding = stem.rsplit("_", 2)[0]
                if decoding not in variant_files:
                    variant_files[decoding] = sub

        if len(variant_files) >= 1:
            models[model_dir.name] = variant_files

    return models


# ---------------------------------------------------------------------------
# Radar plot
# ---------------------------------------------------------------------------

# Metrics to display and their display names + normalization ranges
METRICS = [
    ("rouge_l", "ROUGE-L", 0.0, 1.0),
    ("bertscore", "BERTScore", 0.0, 1.0),
    ("judge_completeness", "Completeness", 1.0, 5.0),
    ("judge_accuracy", "Accuracy", 1.0, 5.0),
    ("judge_relevance", "Relevance", 1.0, 5.0),
]


def normalize(value: float, lo: float, hi: float) -> float:
    """Normalize value from [lo, hi] to [0, 1]."""
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]


def make_radar(
    runs: list[dict[str, float]],
    labels: list[str],
    out_path: Path,
) -> None:
    setup_plot_style()

    metric_labels = [m[1] for m in METRICS]
    num_metrics = len(METRICS)

    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    ax.set_theta_zero_location("N")  # first vertex at top
    ax.set_theta_direction(-1)  # clockwise

    # Hide all default decorations — we draw everything manually
    ax.set_facecolor("none")
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    ax.set_yticks([])
    ax.set_yticklabels([])

    yticks = [0.2, 0.4, 0.6, 0.8, 1.0]

    # Fill the outer pentagon background
    ax.fill(angles, [1.0] * len(angles), facecolor="#EFEFEAFF", edgecolor="black", linewidth=1.5, zorder=0)

    # Draw polygon gridlines (straight edges between vertices)
    for r in yticks[:-1]:  # skip 1.0 — outer border covers it
        ax.plot(angles, [r] * len(angles), color="white", alpha=0.7, linewidth=1.5, linestyle="--", zorder=1)

    # Draw spoke lines from center to each vertex
    for a in angles[:-1]:
        ax.plot([a, a], [0, 1.0], color="white", alpha=0.7, linewidth=1.5, linestyle="--", zorder=1)

    # Data series
    handles = []
    for i, (run_means, label) in enumerate(zip(runs, labels)):
        values = []
        for key, _, lo, hi in METRICS:
            raw = run_means.get(key, lo)
            values.append(normalize(raw, lo, hi))
        values += values[:1]  # close

        color = COLORS[i % len(COLORS)]
        (line,) = ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=label,
            color=color,
            markersize=5,
            zorder=3,
        )
        ax.fill(angles, values, facecolor=facecolor_alpha(color, 0.15), zorder=2)
        handles.append(line)

    # Metric labels on axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)

    # Radial tick labels (placed along the first spoke: angle=0, i.e. top)
    for r_val, r_label in zip(yticks, ["0.2", "0.4", "0.6", "0.8", "1.0"]):
        ax.text(
            0,
            r_val,
            r_label,
            fontsize=8,
            color="grey",
            ha="center",
            va="bottom",
            zorder=4,
        )

    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    # Save main figure (no legend)
    save_figure(fig, out_path)
    print(f"Saved figure: {out_path}")

    # Save legend separately
    legend_path = out_path.with_name(out_path.stem + "_legend" + out_path.suffix)
    save_legend(handles, labels, legend_path, ncol=len(labels))
    print(f"Saved legend: {legend_path}")


# ---------------------------------------------------------------------------
# Score table (printed to console)
# ---------------------------------------------------------------------------


def print_comparison(runs: list[dict[str, float]], labels: list[str]) -> None:
    """Print a comparison table to the console."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="ACI-Bench Results")
    table.add_column("Metric", style="bold")
    for label in labels:
        table.add_column(label, justify="right")
    if len(labels) == 2:
        table.add_column("Delta", justify="right")

    for key, display, _, hi in METRICS:
        row = [display]
        vals = []
        for run in runs:
            v = run.get(key, -1)
            vals.append(v)
            if hi > 1:
                row.append(f"{v:.2f}" if v >= 0 else "N/A")
            else:
                row.append(f"{v:.4f}" if v >= 0 else "N/A")

        if len(labels) == 2 and all(v >= 0 for v in vals):
            delta = vals[0] - vals[1]
            sign = "+" if delta > 0 else ""
            color = "green" if delta > 0 else "red" if delta < 0 else ""
            if hi > 1:
                row.append(f"[{color}]{sign}{delta:.2f}[/{color}]")
            else:
                row.append(f"[{color}]{sign}{delta:.4f}[/{color}]")
        elif len(labels) == 2:
            row.append("N/A")

        table.add_row(*row)

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Radar plot for ACI-Bench decoding comparison")

    # New: auto-discovery mode
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Root results directory (discovers {dir}/aci_bench/{model}/ structure)",
    )

    # Legacy: explicit file mode
    parser.add_argument("--decore", default=None, help="Path to DeCoRe results JSONL")
    parser.add_argument("--greedy", default=None, help="Path to Greedy results JSONL")

    parser.add_argument(
        "--labels",
        nargs="+",
        default=["DeCoRe", "Greedy"],
        help="Legend labels (default: DeCoRe Greedy)",
    )
    parser.add_argument(
        "--out",
        default="figures/acibench_radar.svg",
        help="Output figure path (auto mode appends model name)",
    )
    args = parser.parse_args()

    if args.results_dir is not None:
        # Auto-discovery mode: one plot per model
        models = discover_model_results(args.results_dir)
        assert len(models) > 0, (
            f"No models with multiple decoding methods found under " f"{args.results_dir}/aci_bench/"
        )

        out_dir = Path(args.out).parent
        out_suffix = Path(args.out).suffix or ".svg"

        for model_name, decoding_files in models.items():
            print(f"\n=== {model_name} ===")
            runs = []
            labels = []
            for decoding, fpath in sorted(decoding_files.items()):
                runs.append(load_mean_scores(fpath))
                labels.append(decoding.capitalize())

            print_comparison(runs, labels)
            out_path = out_dir / f"acibench_radar_{model_name}{out_suffix}"
            make_radar(runs, labels, out_path)

    elif args.decore is not None and args.greedy is not None:
        # Legacy explicit mode
        decore_means = load_mean_scores(args.decore)
        greedy_means = load_mean_scores(args.greedy)

        runs = [decore_means, greedy_means]
        labels = args.labels[:2]

        print_comparison(runs, labels)
        make_radar(runs, labels, Path(args.out))

    else:
        parser.error("Provide either --results-dir or both --decore and --greedy")


if __name__ == "__main__":
    main()
