#!/usr/bin/env python3
"""MedRAG accuracy bar charts across decoding variants.

Unlike the other downstream-bar scripts, MedRAG is split per sub-dataset —
MMLU-Med, MedQA, and SuperGPQA-Med are different test beds with different
accuracy levels and different sample sizes, so combining them into a single
``Overall`` bar would be misleading. Each sub-dataset gets its own subdir
under ``<out-dir>/`` with its own Overall figure (titled with the sub-dataset
name) and the matching ``Overall_3models`` variant.

Per-model figures are skipped: each sub-dataset has only one domain, so a
per-model figure would be a single bar group identical to its Overall.

Only ``accuracy`` is plotted — MedRAG is MCQ so there is no ``tag_present``
sanity score in the results.

Usage:
    python locos/plotting/medrag_bar.py \\
        --results-root ../locos-results/downstream_results \\
        --out-dir figures/medrag_bar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from rich.console import Console

from locos.plotting._downstream_bar_common import (
    THREE_MODEL_FIGSIZE,
    THREE_MODEL_ORDER,
    aggregate,
    discover_long_form,
    make_overall_figure,
    overall_per_seed,
)
from locos.plotting._paths import default_downstream_results_root
from locos.plotting.longbench_v2_radar import MODEL_ORDER

console = Console()


# (task_dir, pretty_title, output_subdir).
SUB_DATASETS: list[tuple[str, str, str]] = [
    ("medrag_mmlu_med_top10", "MMLU-Med", "MMLU-Med"),
    ("medrag_medqa_top10", "MedQA", "MedQA"),
    ("medrag_supergpqa_med_top10", "SuperGPQA-Med", "SuperGPQA-Med"),
]

# Title for the across-sub-datasets figure (macro-averaged over all 3).
ACROSS_TITLE = "Medical RAG"

METRICS = [("accuracy", "Accuracy")]


def domain_for_row(row: dict, task_dir_name: str) -> str | None:
    """Group by sub-dataset name (from metadata, with task-dir fallback)."""
    ds = (row.get("metadata") or {}).get("dataset")
    if ds:
        return ds
    if task_dir_name.startswith("medrag_") and task_dir_name.endswith("_top10"):
        return task_dir_name[len("medrag_") : -len("_top10")]
    return None


def render_one_subdataset(
    results_root: Path,
    out_dir: Path,
    task_dirs: list[str],
    pretty_title: str,
) -> None:
    """Discover, aggregate, and render Overall figures for one or more task
    dirs. With one entry, this renders a single sub-dataset's Overall; with
    multiple, ``overall_per_seed`` macro-averages across the supplied task
    dirs (each treated as a domain) — i.e. the across-sub-datasets view."""
    metric_keys = [m for m, _ in METRICS]
    long_df, counts = discover_long_form(results_root, task_dirs, domain_for_row, metric_keys)
    overall_df = overall_per_seed(results_root, task_dirs, domain_for_row, metric_keys)
    if long_df.empty:
        console.print(f"[yellow]No data for {task_dirs} — skipping[/yellow]")
        return

    full_df = long_df if overall_df.empty else pd.concat([long_df, overall_df], ignore_index=True)
    summary = aggregate(full_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    console.print(f"[green]Saved CSV:[/green] {out_dir / 'summary.csv'}")

    total_samples = sum(counts.values())
    primary_key = METRICS[0][0]
    for metric_key, metric_label in METRICS:
        suffix = "" if metric_key == primary_key else f"_{metric_key}"
        make_overall_figure(
            summary,
            MODEL_ORDER,
            metric_key,
            metric_label,
            total_samples,
            out_dir / f"Overall{suffix}.svg",
            task_name=pretty_title,
        )
        make_overall_figure(
            summary,
            THREE_MODEL_ORDER,
            metric_key,
            metric_label,
            total_samples,
            out_dir / f"Overall{suffix}_3models.svg",
            figsize=THREE_MODEL_FIGSIZE,
            rotate_xticks=False,
            task_name=pretty_title,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-model Overall bar charts for each MedRAG sub-dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_downstream_results_root(),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/medrag_bar"),
        help="Parent directory; one subdir per sub-dataset is created underneath.",
    )
    args = parser.parse_args()

    for task_dir, pretty_title, subdir_name in SUB_DATASETS:
        render_one_subdataset(
            args.results_root,
            args.out_dir / subdir_name,
            [task_dir],
            pretty_title,
        )

    # Across-sub-datasets: macro-average over the 3 sub-datasets, written at
    # the parent ``out_dir`` so it sits next to the per-sub-dataset folders.
    render_one_subdataset(
        args.results_root,
        args.out_dir,
        [td for td, _, _ in SUB_DATASETS],
        ACROSS_TITLE,
    )


if __name__ == "__main__":
    main()
