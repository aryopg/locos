#!/usr/bin/env python3
"""Line plots showing MedRAG accuracy vs. number of retrieved passages (top-k).

Discovers results from the structured output directory layout produced by
EvalRunner (``{output_dir}/medrag_{dataset}_top{k}/{model}/{variant}/``).
Generates one plot per (model, sub-dataset) pair, with each line representing
a decoding variant (greedy, locos_wu_niah, locos_wu_nolima, etc.).

Usage:
    # Auto-discover all models and sub-datasets
    python scripts/eval/plot_medrag_topk.py --results-dir eval_results

    # Filter to specific sub-datasets
    python scripts/eval/plot_medrag_topk.py --results-dir eval_results \
        --datasets medqa mmlu_med supergpqa_med

    # Filter to specific models
    python scripts/eval/plot_medrag_topk.py --results-dir eval_results \
        --models Qwen_Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

from locos_eval.utils.plotting import (
    save_figure,
    save_legend,
    setup_plot_style,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBDATASET_LABELS = {
    "mmlu_med": "MMLU-Med",
    "medqa": "MedQA",
    "medmcqa": "MedMCQA",
    "pubmedqa": "PubMedQA",
    "supergpqa_med": "SuperGPQA",
}

VARIANT_LABELS = {
    "greedy": "Greedy",
    "locos_wu_niah": "LOCOS (Wu NIAH)",
    "locos_wu_nolima": "LOCOS (Wu NoLiMa)",
    "locos_logitcontrib_nolima": "LOCOS (LogitContrib NoLiMa)",
    "locos_ori": "LOCOS (Ori)",
    "locos_cri": "LOCOS (CRI)",
    "ablation_wu_niah": "Ablation (Wu NIAH)",
    "ablation_wu_nolima": "Ablation (Wu NoLiMa)",
    "ablation_logitcontrib_nolima": "Ablation (LogitContrib NoLiMa)",
    "ablation_random": "Ablation (Random)",
}

COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD"]
MARKERS = ["o", "s", "D", "^", "v", "P"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _parse_task_dir(dirname: str) -> tuple[str, int | None]:
    """Parse a medrag task directory name into (subdataset, top_k).

    Examples:
        ``"medrag_medqa_top5"``  -> ``("medqa", 5)``
        ``"medrag_medqa"``       -> ``("medqa", None)``
        ``"medrag_top10"``       -> ``("all", 10)``
        ``"medrag"``             -> ``("all", None)``
    """
    rest = dirname.removeprefix("medrag").lstrip("_")
    m = re.match(r"^(.+?)_top(\d+)$", rest)
    if m:
        return m.group(1) or "all", int(m.group(2))
    m = re.match(r"^top(\d+)$", rest)
    if m:
        return "all", int(m.group(1))
    return rest or "all", None


def load_mean_accuracy(path: str | Path) -> float:
    """Load a results JSONL and return overall mean accuracy."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) > 0, f"No records in {path}"

    accs = [r["scores"]["accuracy"] for r in records if "accuracy" in r.get("scores", {})]
    assert len(accs) > 0, f"No accuracy scores in {path}"
    return sum(accs) / len(accs)


def discover_topk_results(
    results_dir: Path,
) -> dict[str, dict[str, dict[str, dict[int, Path]]]]:
    """Discover results across all top-k values.

    Returns:
        ``{model: {variant: {subdataset: {top_k: Path}}}}``
    """
    data: dict[str, dict[str, dict[str, dict[int, Path]]]] = {}

    task_dirs = sorted(results_dir.glob("medrag*"))

    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue

        subdataset, topk = _parse_task_dir(task_dir.name)
        if topk is None:
            continue  # skip directories without a top-k suffix

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
                        (data.setdefault(model_name, {}).setdefault(variant, {}).setdefault(subdataset, {}))[topk] = (
                            result_files[0]
                        )
                elif sub.suffix == ".jsonl" and not sub.name.endswith("_generations.jsonl"):
                    # Legacy flat layout
                    stem = sub.stem
                    variant = stem.rsplit("_", 2)[0]
                    entry = data.setdefault(model_name, {}).setdefault(variant, {}).setdefault(subdataset, {})
                    if topk not in entry:
                        entry[topk] = sub

    return data


# ---------------------------------------------------------------------------
# Line plot
# ---------------------------------------------------------------------------


def make_topk_plot(
    variant_data: dict[str, dict[int, float]],
    subdataset: str,
    out_path: Path,
) -> None:
    """Create a line plot of accuracy vs top-k for one (model, subdataset).

    Args:
        variant_data: ``{variant_name: {top_k: accuracy}}``
        subdataset: Sub-dataset name (for axis label).
        out_path: Output SVG path.
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    handles = []
    labels = []
    for i, (variant, topk_acc) in enumerate(sorted(variant_data.items())):
        if not topk_acc:
            continue

        ks = sorted(topk_acc.keys())
        accs = [topk_acc[k] for k in ks]

        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        display = VARIANT_LABELS.get(variant, variant.replace("_", " ").title())

        (line,) = ax.plot(
            ks,
            accs,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2,
            zorder=3,
        )
        handles.append(line)
        labels.append(display)

    ax.set_xlabel("Number of retrieved passages ($k$)")
    ax.set_ylabel("Accuracy")

    # Integer x-ticks at the actual top-k values
    all_ks = sorted({k for vd in variant_data.values() for k in vd})
    ax.set_xticks(all_ks)
    ax.set_xticklabels([str(k) for k in all_ks])

    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    fig.tight_layout()

    save_figure(fig, out_path)
    print(f"Saved figure: {out_path}")

    legend_path = out_path.with_name(out_path.stem + "_legend" + out_path.suffix)
    save_legend(handles, labels, legend_path, ncol=min(len(labels), 3))
    print(f"Saved legend: {legend_path}")


# ---------------------------------------------------------------------------
# Score table
# ---------------------------------------------------------------------------


def print_topk_table(
    variant_data: dict[str, dict[int, float]],
    model_name: str,
    subdataset: str,
) -> None:
    """Print a table of accuracy per variant and top-k."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    ds_label = SUBDATASET_LABELS.get(subdataset, subdataset)
    table = Table(title=f"{model_name} — {ds_label}")

    all_ks = sorted({k for vd in variant_data.values() for k in vd})
    table.add_column("Variant", style="bold")
    for k in all_ks:
        table.add_column(f"top-{k}", justify="right")

    for variant in sorted(variant_data):
        display = VARIANT_LABELS.get(variant, variant)
        row = [display]
        for k in all_ks:
            acc = variant_data[variant].get(k)
            row.append(f"{acc:.1%}" if acc is not None else "—")
        table.add_row(*row)

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Line plot of MedRAG accuracy vs number of retrieved passages")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Root results directory (discovers medrag_*_top*/{model}/{variant}/ structure)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Filter to specific sub-datasets (e.g. medqa mmlu_med)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filter to specific model directory names (e.g. Qwen_Qwen3-8B)",
    )
    parser.add_argument(
        "--out",
        default="figures/medrag_topk.svg",
        help="Output figure path (model and dataset names are appended)",
    )
    args = parser.parse_args()

    data = discover_topk_results(args.results_dir)
    assert len(data) > 0, f"No MedRAG top-k results found under {args.results_dir}/medrag_*_top*/"

    # Optional filters
    if args.models:
        data = {m: v for m, v in data.items() if m in args.models}
    assert len(data) > 0, "No models matched --models filter"

    out_dir = Path(args.out).parent
    out_suffix = Path(args.out).suffix or ".svg"

    for model_name, variant_map in sorted(data.items()):
        # Collect all subdatasets across variants
        all_subdatasets = sorted({ds for vdata in variant_map.values() for ds in vdata})

        if args.datasets:
            all_subdatasets = [ds for ds in all_subdatasets if ds in args.datasets]

        for subdataset in all_subdatasets:
            # Build {variant: {topk: accuracy}} for this subdataset
            variant_data: dict[str, dict[int, float]] = {}
            for variant, ds_map in variant_map.items():
                if subdataset not in ds_map:
                    continue
                topk_paths = ds_map[subdataset]
                topk_acc: dict[int, float] = {}
                for k, fpath in topk_paths.items():
                    topk_acc[k] = load_mean_accuracy(fpath)
                if topk_acc:
                    variant_data[variant] = topk_acc

            if not variant_data:
                continue

            print(f"\n=== {model_name} / {subdataset} ===")
            print_topk_table(variant_data, model_name, subdataset)

            out_path = out_dir / f"medrag_topk_{model_name}_{subdataset}{out_suffix}"
            make_topk_plot(variant_data, subdataset, out_path)


if __name__ == "__main__":
    main()
