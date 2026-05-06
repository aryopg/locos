#!/usr/bin/env python3
"""Bar plot comparing decoding methods on MedRAG accuracy per sub-dataset.

Discovers results from the structured output directory layout produced by
EvalRunner (``{output_dir}/medrag_{dataset}/{model}/``).  Generates one
grouped bar plot per model, with legend saved as a separate file.

Usage:
    # Auto-discover all models under eval_results/
    python scripts/eval/plot_medrag_bar.py --results-dir eval_results

    # Explicit per-file mode
    python scripts/eval/plot_medrag_bar.py \
        --decore results_decore.jsonl \
        --greedy results_greedy.jsonl \
        --out figures/medrag_bar.svg
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

# Sub-datasets in display order
SUBDATASETS = ["mmlu_med", "medqa", "medmcqa", "pubmedqa", "supergpqa_med"]
SUBDATASET_LABELS = {
    "mmlu_med": "MMLU-Med",
    "medqa": "MedQA",
    "medmcqa": "MedMCQA",
    "pubmedqa": "PubMedQA",
    "supergpqa_med": "SuperGPQA",
}

COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_accuracy_by_subdataset(path: str | Path) -> dict[str, float]:
    """Load a results JSONL and return mean accuracy per sub-dataset.

    Falls back to a single "all" key when records lack a ``dataset``
    metadata field.
    """
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) > 0, f"No records in {path}"

    # Group by sub-dataset
    groups: dict[str, list[float]] = {}
    for r in records:
        ds = r.get("metadata", {}).get("dataset", "all")
        acc = r.get("scores", {}).get("accuracy")
        if acc is not None:
            groups.setdefault(ds, []).append(acc)

    return {ds: sum(vs) / len(vs) for ds, vs in groups.items()}


def _parse_task_dir(dirname: str) -> tuple[str, int | None]:
    """Parse a medrag task directory name into (subdataset, top_k).

    Examples:
        ``"medrag_medqa_top5"``  → ``("medqa", 5)``
        ``"medrag_medqa"``       → ``("medqa", None)``
        ``"medrag_top10"``       → ``("all", 10)``
        ``"medrag"``             → ``("all", None)``
    """
    import re

    rest = dirname.removeprefix("medrag").lstrip("_")
    # Check for trailing _topN
    m = re.match(r"^(.+?)_top(\d+)$", rest)
    if m:
        return m.group(1) or "all", int(m.group(2))
    m = re.match(r"^top(\d+)$", rest)
    if m:
        return "all", int(m.group(1))
    return rest or "all", None


def discover_model_results(
    results_dir: Path,
    top_k: int | None = None,
) -> dict[str, dict[str, dict[str, Path]]]:
    """Discover results per model from ``{results_dir}/medrag*/{model}/{variant}/``.

    Supports both the new layout (``{model}/{variant}/results_*.jsonl``) and
    the legacy flat layout (``{model}/{decoding}_*.jsonl``).

    Args:
        results_dir: Root results directory.
        top_k: If set, only include results for this top-k value.

    Returns:
        ``{model_name: {variant: {subdataset: Path}}}``
    """
    models: dict[str, dict[str, dict[str, Path]]] = {}

    # Find all medrag task directories (medrag_medqa_top5, medrag_mmlu_med, etc.)
    task_dirs = sorted(results_dir.glob("medrag*"))

    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue

        subdataset, dir_topk = _parse_task_dir(task_dir.name)

        # Filter by top-k if requested
        if top_k is not None and dir_topk is not None and dir_topk != top_k:
            continue

        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            for sub in sorted(model_dir.iterdir()):
                if sub.is_dir():
                    # New layout: {model}/{variant}/results_*.jsonl
                    result_files = sorted(
                        sub.glob("results_*.jsonl"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if result_files:
                        variant = sub.name
                        models.setdefault(model_name, {}).setdefault(variant, {})[subdataset] = result_files[0]
                elif sub.suffix == ".jsonl" and not sub.name.endswith("_generations.jsonl"):
                    # Legacy flat layout: {model}/{decoding}_{timestamp}.jsonl
                    stem = sub.stem
                    variant = stem.rsplit("_", 2)[0]
                    if subdataset not in models.get(model_name, {}).get(variant, {}):
                        models.setdefault(model_name, {}).setdefault(variant, {})[subdataset] = sub

    return models


# ---------------------------------------------------------------------------
# Bar plot
# ---------------------------------------------------------------------------


def make_bar_plot(
    runs: list[dict[str, float]],
    labels: list[str],
    out_path: Path,
    model_name: str = "",
) -> None:
    """Create a grouped bar plot of accuracy per sub-dataset."""
    setup_plot_style()

    # Determine which sub-datasets have data
    all_subdatasets = []
    for ds in SUBDATASETS:
        if any(ds in run for run in runs):
            all_subdatasets.append(ds)

    # Fallback: if records use "all" key (single combined run)
    if not all_subdatasets:
        all_subdatasets = sorted({ds for run in runs for ds in run})

    n_groups = len(all_subdatasets)
    n_methods = len(runs)
    assert n_groups > 0, "No sub-dataset data found"

    x = np.arange(n_groups)
    width = 0.7 / n_methods

    fig, ax = plt.subplots(figsize=(max(5.5, n_groups * 1.2), 3.5))

    handles = []
    for i, (run, label) in enumerate(zip(runs, labels)):
        values = [run.get(ds, 0.0) for ds in all_subdatasets]
        color = COLORS[i % len(COLORS)]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=facecolor_alpha(color, 0.85),
            edgecolor="black",
            linewidth=1.5,
            zorder=3,
        )
        handles.append(bars)

    # Labels and formatting
    display_labels = [SUBDATASET_LABELS.get(ds, ds) for ds in all_subdatasets]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    fig.tight_layout()

    save_figure(fig, out_path)
    print(f"Saved figure: {out_path}")

    # Save legend separately
    legend_path = out_path.with_name(out_path.stem + "_legend" + out_path.suffix)
    save_legend(
        [h[0] for h in handles],
        labels,
        legend_path,
        ncol=len(labels),
    )
    print(f"Saved legend: {legend_path}")


# ---------------------------------------------------------------------------
# Score table
# ---------------------------------------------------------------------------


def print_comparison(
    runs: list[dict[str, float]],
    labels: list[str],
) -> None:
    """Print a comparison table to the console."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="MedRAG Accuracy by Sub-dataset")
    table.add_column("Sub-dataset", style="bold")
    for label in labels:
        table.add_column(label, justify="right")
    if len(labels) == 2:
        table.add_column("Delta", justify="right")

    # Determine which sub-datasets to show
    all_ds = []
    for ds in SUBDATASETS:
        if any(ds in run for run in runs):
            all_ds.append(ds)
    if not all_ds:
        all_ds = sorted({ds for run in runs for ds in run})

    overall_vals = [[] for _ in runs]

    for ds in all_ds:
        display = SUBDATASET_LABELS.get(ds, ds)
        row = [display]
        vals = []
        for j, run in enumerate(runs):
            v = run.get(ds, -1.0)
            vals.append(v)
            if v >= 0:
                row.append(f"{v:.1%}")
                overall_vals[j].append(v)
            else:
                row.append("N/A")

        if len(labels) == 2 and all(v >= 0 for v in vals):
            delta = vals[0] - vals[1]
            sign = "+" if delta > 0 else ""
            color = "green" if delta > 0 else "red" if delta < 0 else ""
            row.append(f"[{color}]{sign}{delta:.1%}[/{color}]")
        elif len(labels) == 2:
            row.append("N/A")

        table.add_row(*row)

    # Overall mean row
    row = ["[bold]Overall[/bold]"]
    mean_vals = []
    for vals in overall_vals:
        if vals:
            m = sum(vals) / len(vals)
            mean_vals.append(m)
            row.append(f"[bold]{m:.1%}[/bold]")
        else:
            mean_vals.append(-1.0)
            row.append("N/A")
    if len(labels) == 2 and all(v >= 0 for v in mean_vals):
        delta = mean_vals[0] - mean_vals[1]
        sign = "+" if delta > 0 else ""
        color = "green" if delta > 0 else "red" if delta < 0 else ""
        row.append(f"[bold][{color}]{sign}{delta:.1%}[/{color}][/bold]")
    elif len(labels) == 2:
        row.append("N/A")
    table.add_row(*row)

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Bar plot for MedRAG accuracy by sub-dataset")

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Root results directory (discovers medrag_*/{model}/ structure)",
    )

    # Explicit file mode
    parser.add_argument("--decore", default=None, help="Path to DeCoRe results JSONL")
    parser.add_argument("--greedy", default=None, help="Path to Greedy results JSONL")

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Filter to a specific top-k value (default: use all found)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["DeCoRe", "Greedy"],
        help="Legend labels (default: DeCoRe Greedy)",
    )
    parser.add_argument(
        "--out",
        default="figures/medrag_bar.svg",
        help="Output figure path (auto mode appends model name)",
    )
    args = parser.parse_args()

    if args.results_dir is not None:
        models = discover_model_results(args.results_dir, top_k=args.top_k)
        assert len(models) > 0, f"No MedRAG results found under {args.results_dir}/medrag_*/"

        out_dir = Path(args.out).parent
        out_suffix = Path(args.out).suffix or ".svg"

        for model_name, decoding_map in models.items():
            print(f"\n=== {model_name} ===")
            runs = []
            labels = []
            for decoding, subdataset_files in sorted(decoding_map.items()):
                # Merge accuracy from all sub-dataset files
                merged: dict[str, float] = {}
                for subdataset, fpath in subdataset_files.items():
                    acc = load_accuracy_by_subdataset(fpath)
                    # If file has per-subdataset breakdown, use that
                    if len(acc) > 1 or "all" not in acc:
                        merged.update(acc)
                    else:
                        # Single file covers one subdataset
                        merged[subdataset] = acc.get("all", acc.get(subdataset, 0.0))
                runs.append(merged)
                labels.append(decoding.capitalize())

            print_comparison(runs, labels)
            out_path = out_dir / f"medrag_bar_{model_name}{out_suffix}"
            make_bar_plot(runs, labels, out_path, model_name=model_name)

    elif args.decore is not None and args.greedy is not None:
        decore_acc = load_accuracy_by_subdataset(args.decore)
        greedy_acc = load_accuracy_by_subdataset(args.greedy)

        runs = [decore_acc, greedy_acc]
        labels = args.labels[:2]

        print_comparison(runs, labels)
        make_bar_plot(runs, labels, Path(args.out))

    else:
        parser.error("Provide either --results-dir or both --decore and --greedy")


if __name__ == "__main__":
    main()
