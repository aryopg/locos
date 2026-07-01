#!/usr/bin/env python3
"""E5 — Per-instance cue-overlap / cue-uniqueness correlation with damage gap.

Original verbatim-literality operationalization failed (all extractive benchmarks
scored 1.000, reducing n=5 to a binary NoLiMa-vs-rest comparison). Replaced with
two per-instance axes that have real within-benchmark variance:

  cue_overlap   — ROUGE-1 F1(question, answer-bearing span)
  cue_uniqueness — 1 / count(answer string in full context)

Per-instance damage gap G_i = wu_acc_i - locos_acc_i on baseline-correct instances.
Positive G = LOCOS ablation hurts more than Wu on this instance.

Correlation run per model family (Qwen / Gemma / Olmo) to expose the
Gemma/Olmo-specific Wu > LOCOS pattern that aggregate mean G masks.

Benchmarks: MuSiQue (answerable), BABILong-qa2-0k, BABILong-qa3-0k.
NIAH/NoLiMa excluded — no per-instance downstream JSONL in the eval runner format.

Outputs:
    analysis/outputs/e5/e5_per_instance.csv
    analysis/outputs/e5/e5_family_corr.csv
    analysis/outputs/e5/e5_cue_scatter.svg
    analysis/outputs/e5/e5_cue_scatter_legend.svg

Usage:
    python locos/analysis/benchmark_literality.py
    python locos/analysis/benchmark_literality.py --no-download
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.stats import spearmanr

from locos.analysis._utils import (
    ALL_MODELS,
    MODEL_LABELS,
    get_output_dir,
    load_downstream_from_hf,
    load_eval_rows,
)
from locos_eval.utils.plotting import save_figure, setup_plot_style

console = Console()

BENCHMARKS = [
    ("MuSiQue", "musique_answerable"),
    ("BABILong-qa2", "babilong_qa2_0k"),
    ("BABILong-qa3", "babilong_qa3_0k"),
]

LOCOS_VARIANTS = ["ablation_logitcontrib_nolima", "ablation_locos_nolima"]
WU_VARIANTS = ["ablation_wu_nolima", "ablation_wu_niah"]
BASELINE_VARIANTS = ["greedy"]

MODEL_FAMILY: dict[str, str] = {
    "Qwen/Qwen3-8B": "Qwen",
    "Qwen/Qwen3-14B": "Qwen",
    "Qwen/Qwen3-32B": "Qwen",
    "google/gemma-3-12b-it": "Gemma",
    "google/gemma-3-27b-it": "Gemma",
    "allenai/Olmo-3.1-32B-Instruct": "Olmo",
}

FAMILY_COLORS: dict[str, str] = {
    "Qwen": "#1f77b4",
    "Gemma": "#ff7f0e",
    "Olmo": "#2ca02c",
}


# ---------------------------------------------------------------------------
# Cue metrics
# ---------------------------------------------------------------------------


def _cue_overlap(question: str, answer_span: str) -> float:
    """ROUGE-1 F1 between question and answer-bearing span."""
    q_toks = set(question.lower().split())
    s_toks = set(answer_span.lower().split())
    if not q_toks or not s_toks:
        return 0.0
    common = q_toks & s_toks
    if not common:
        return 0.0
    p = len(common) / len(s_toks)
    r = len(common) / len(q_toks)
    return 2 * p * r / (p + r)


def _cue_uniqueness(answer: str, context: str) -> float:
    """1 / count(answer in context); 0 if answer absent."""
    ans = answer.strip().lower()
    if not ans:
        return 0.0
    count = context.lower().count(ans)
    return 1.0 / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Dataset index loaders — sample_id (= dataset row index) → instance dict
# ---------------------------------------------------------------------------


def _load_babilong_index(qa_split: str, context_len: str) -> dict[int, dict]:
    from datasets import load_dataset

    ds = load_dataset("RMT-team/babilong", context_len, split=qa_split, trust_remote_code=True)
    index: dict[int, dict] = {}
    for i, row in enumerate(ds):
        story: str = row["input"]
        question: str = row["question"]
        target: str = row["target"]
        # Answer-bearing span = last story line containing the target
        span = ""
        for line in reversed(story.split("\n")):
            if target.strip().lower() in line.strip().lower() and line.strip():
                span = line.strip()
                break
        index[i] = {
            "question": question,
            "answer": target,
            "answer_span": span,
            "context": story,
        }
    return index


def _load_musique_index() -> dict[int, dict]:
    from datasets import load_dataset

    ds = load_dataset("bdsaglam/musique", "answerable", split="validation", trust_remote_code=True)
    index: dict[int, dict] = {}
    for i, row in enumerate(ds):
        span = " ".join(p["paragraph_text"] for p in row["paragraphs"] if p.get("is_supporting"))
        context = " ".join(p["paragraph_text"] for p in row["paragraphs"])
        index[i] = {
            "question": row["question"],
            "answer": row["answer"],
            "answer_span": span,
            "context": context,
        }
    return index


def _get_dataset_index(task_key: str) -> dict[int, dict]:
    if task_key.startswith("babilong"):
        parts = task_key.replace("babilong_", "").split("_")
        qa_split = parts[0]
        context_len = parts[1] if len(parts) > 1 else "0k"
        return _load_babilong_index(qa_split, context_len)
    if task_key.startswith("musique"):
        return _load_musique_index()
    raise ValueError(f"No dataset loader for task_key={task_key!r}")


# ---------------------------------------------------------------------------
# Eval row loading
# ---------------------------------------------------------------------------


def _load_acc_by_id(
    task_key: str,
    model: str,
    variants: list[str],
    download: bool,
) -> dict[int, float] | None:
    rows = load_eval_rows(task_key, model, variants)
    if rows is None and download:
        rows = load_downstream_from_hf(task_key, model, variants)
    if not rows:
        return None
    return {int(r.get("sample_id", i)): float(r.get("scores", {}).get("accuracy", 0.0)) for i, r in enumerate(rows)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(download: bool = True) -> None:
    setup_plot_style()
    import matplotlib.pyplot as plt

    out_dir = get_output_dir("e5")
    all_rows: list[dict] = []

    for bench_name, task_key in BENCHMARKS:
        console.print(f"Loading dataset index for {bench_name}...")
        try:
            ds_index = _get_dataset_index(task_key)
        except Exception as e:
            console.print(f"[yellow]SKIP {bench_name}: cannot load dataset — {e}[/yellow]")
            continue

        for model in ALL_MODELS:
            label = MODEL_LABELS[model]
            family = MODEL_FAMILY[model]

            base_map = _load_acc_by_id(task_key, model, BASELINE_VARIANTS, download)
            locos_map = _load_acc_by_id(task_key, model, LOCOS_VARIANTS, download)
            wu_map = _load_acc_by_id(task_key, model, WU_VARIANTS, download)

            if base_map is None or locos_map is None or wu_map is None:
                console.print(f"[yellow]SKIP {label}/{bench_name}: missing eval results[/yellow]")
                continue

            for sid, b_acc in base_map.items():
                if b_acc < 0.5:
                    continue  # restrict to baseline-correct instances
                l_acc = locos_map.get(sid, 0.0)
                w_acc = wu_map.get(sid, 0.0)
                g = w_acc - l_acc  # positive = LOCOS hurts more than Wu

                inst = ds_index.get(sid)
                if inst is None:
                    continue

                co = _cue_overlap(inst["question"], inst["answer_span"])
                cu = _cue_uniqueness(inst["answer"], inst["context"])

                all_rows.append(
                    {
                        "benchmark": bench_name,
                        "model": label,
                        "family": family,
                        "sample_id": sid,
                        "cue_overlap": f"{co:.4f}",
                        "cue_uniqueness": f"{cu:.4f}",
                        "G": f"{g:.4f}",
                    }
                )

    if not all_rows:
        console.print("[red]No per-instance data collected. Check eval results availability.[/red]")
        return

    csv_path = out_dir / "e5_per_instance.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    families = sorted({r["family"] for r in all_rows})
    benchmarks = sorted({r["benchmark"] for r in all_rows})
    cue_axes = ["cue_overlap", "cue_uniqueness"]

    # Correlations: pooled-by-family AND stratified-by-(family, benchmark).
    # Pooled numbers can be driven entirely by between-benchmark differences on
    # the predictor (e.g. BABILong has low cue_uniqueness by design; MuSiQue
    # has high). Stratified ρ is the primary diagnostic; pooled is context only.
    corr_rows: list[dict] = []

    for family in families:
        for benchmark in ["pooled", *benchmarks]:
            if benchmark == "pooled":
                subset = [r for r in all_rows if r["family"] == family]
            else:
                subset = [r for r in all_rows if r["family"] == family and r["benchmark"] == benchmark]
            if len(subset) < 10:
                continue
            for axis in cue_axes:
                x = np.array([float(r[axis]) for r in subset])
                y = np.array([float(r["G"]) for r in subset])
                if len(np.unique(x)) < 3:
                    continue
                rho, p = spearmanr(x, y)
                corr_rows.append(
                    {
                        "family": family,
                        "benchmark": benchmark,
                        "axis": axis,
                        "n": len(subset),
                        "rho": f"{rho:.4f}",
                        "p": f"{p:.4f}",
                    }
                )

    corr_csv = out_dir / "e5_family_corr.csv"
    if corr_rows:
        with open(corr_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(corr_rows[0]))
            writer.writeheader()
            writer.writerows(corr_rows)

    # Figure: 2 rows (axes) x n_families cols; points colored by benchmark
    bench_colors = {b: c for b, c in zip(benchmarks, ["#1f77b4", "#ff7f0e", "#2ca02c"])}
    ncols = len(families)
    fig, ax_grid = plt.subplots(2, ncols, figsize=(4.0 * ncols, 5.5))
    if ncols == 1:
        ax_grid = [[ax_grid[0]], [ax_grid[1]]]

    for col, family in enumerate(families):
        for row_idx, axis in enumerate(cue_axes):
            ax = ax_grid[row_idx][col]
            for bench in benchmarks:
                pts = [r for r in all_rows if r["family"] == family and r["benchmark"] == bench]
                if pts:
                    ax.scatter(
                        [float(r[axis]) for r in pts],
                        [float(r["G"]) for r in pts],
                        s=5,
                        alpha=0.25,
                        color=bench_colors[bench],
                        linewidths=0.0,
                        label=bench,
                    )
            ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
            ax.set_xlabel(axis.replace("_", " "), fontsize=9)
            if col == 0:
                ax.set_ylabel("G = wu_acc - locos_acc", fontsize=9)
            # Annotate pooled rho (context) and per-benchmark rho (diagnostic)
            annot_lines = []
            pooled_cr = next(
                (r for r in corr_rows if r["family"] == family and r["benchmark"] == "pooled" and r["axis"] == axis),
                None,
            )
            if pooled_cr:
                sign = "+" if float(pooled_cr["rho"]) >= 0 else ""
                annot_lines.append(f"pooled r={sign}{float(pooled_cr['rho']):.2f} p={float(pooled_cr['p']):.3f}")
            for bench in benchmarks:
                bcr = next(
                    (r for r in corr_rows if r["family"] == family and r["benchmark"] == bench and r["axis"] == axis),
                    None,
                )
                if bcr:
                    sign = "+" if float(bcr["rho"]) >= 0 else ""
                    annot_lines.append(f"{bench[:6]} r={sign}{float(bcr['rho']):.2f} p={float(bcr['p']):.3f}")
            if annot_lines:
                ax.text(
                    0.03,
                    0.97,
                    "\n".join(annot_lines),
                    transform=ax.transAxes,
                    va="top",
                    fontsize=7,
                    bbox=dict(fc="white", ec="none", alpha=0.8),
                )
            ax.text(0.97, 0.97, family, transform=ax.transAxes, va="top", ha="right", fontsize=9, fontweight="bold")

    fig.tight_layout()
    save_figure(fig, out_dir / "e5_cue_scatter.svg")

    # Console: pooled table, then stratified table
    pooled_rows = [r for r in corr_rows if r["benchmark"] == "pooled"]
    strat_rows = [r for r in corr_rows if r["benchmark"] != "pooled"]

    table = Table(title="E5 — Pooled-by-family rho (context only; may reflect benchmark composition)")
    table.add_column("Family")
    table.add_column("Axis")
    table.add_column("n", justify="right")
    table.add_column("rho", justify="right")
    table.add_column("p", justify="right")
    for r in pooled_rows:
        table.add_row(r["family"], r["axis"], str(r["n"]), r["rho"], r["p"])
    console.print(table)

    table2 = Table(title="E5 — Within-benchmark rho (stratified; primary diagnostic)")
    table2.add_column("Family")
    table2.add_column("Benchmark")
    table2.add_column("Axis")
    table2.add_column("n", justify="right")
    table2.add_column("rho", justify="right")
    table2.add_column("p", justify="right")
    for r in strat_rows:
        table2.add_row(r["family"], r["benchmark"], r["axis"], str(r["n"]), r["rho"], r["p"])
    console.print(table2)

    within_sig = [r for r in strat_rows if float(r["p"]) < 0.05]
    if within_sig:
        console.print(
            "[green]-> E5 DECISION: Significant within-benchmark correlations. Report per-family in §4.7.[/green]"
        )
    else:
        console.print(
            "[yellow]-> E5 DECISION: No within-benchmark significance. "
            "Pooled significance (if any) is benchmark-composition artifact. Report as tested negative.[/yellow]"
        )

    console.print(f"\n[dim]Saved:[/dim] {out_dir}/e5_per_instance.csv, e5_family_corr.csv, e5_cue_scatter.svg")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E5 — Per-instance cue correlation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    run(download=not args.no_download)


if __name__ == "__main__":
    main()
