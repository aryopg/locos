#!/usr/bin/env python3
"""E7 — 2×2 factorial: observable (α/φ) × contrast axis (spatial/temporal).

Runs cells B (α-spatial) and D (φ-spatial/LOCOS); skips cells A and C.
  Cell A (α-temporal): needs scoring re-run — not yet available.
  Cell C (φ-temporal): gated on V2 (non-answer step arrays) — blocked.

Outputs:
    analysis/outputs/e7/e7_ablation_<model>.csv   (k, cell, rouge, ci_lo, ci_hi)
    analysis/outputs/e7/e7_factorial.svg           (2×3, all 6 models)
    analysis/outputs/e7/e7_factorial_legend.svg

Decision rules printed to console (per headline model):
    D ≈ B (overlapping CIs at k∈{10,20,50}) → OV claim falsified
    D strictly below B (non-overlap, D < B)  → OV component validated
    D strictly above B (non-overlap, D > B)  → adverse result, flag

Usage:
    python locos/analysis/e7_factorial.py
    python locos/analysis/e7_factorial.py --no-download
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from locos_eval.utils.plotting import save_figure, setup_plot_style
from locos.analysis._utils import (
    ABLATION_K_VALUES,
    ALL_MODELS,
    MODEL_LABELS,
    MODEL_SHORT,
    bootstrap_ci,
    get_output_dir,
)

console = Console()

_LOCOS_RESULTS = Path(os.environ.get("LOCOS_RESULTS_DIR", str(_REPO_ROOT.parent / "locos-results")))
_ABLATION_DIRS: list[Path] = [
    _REPO_ROOT / "ablation_results",
    _LOCOS_RESULTS / "ablation_results",
]

HEADLINE_MODELS = ["Qwen/Qwen3-8B", "google/gemma-3-12b-it"]
K_VALUES = [k for k in ABLATION_K_VALUES if k <= 50]  # [1, 5, 10, 20, 50]

CELL_COLORS = {
    "D": "#0763BA",
    "B": "#41A9AC",
    "random": "#DF922D",
}
CELL_LABELS = {
    "D": r"$\phi$-spatial (LOCOS)",
    "B": r"$\alpha$-spatial",
    "random": "Random",
}
CELL_MARKERS = {"D": "o", "B": "P", "random": "^"}


class CellPoint(NamedTuple):
    k: int
    rouge: float
    ci_lo: float
    ci_hi: float


# ---------------------------------------------------------------------------
# File discovery and loading
# ---------------------------------------------------------------------------


def _find_ablation_dir(short: str, stem: str) -> Path | None:
    for d in _ABLATION_DIRS:
        if (d / f"nolima_ablation_{short}_{stem}.json").exists():
            return d
    return None


def _load_cache(abl_dir: Path, short: str, stem: str) -> dict:
    with open(abl_dir / f"nolima_ablation_{short}_{stem}.json") as f:
        return json.load(f)


def _load_trial_rouge(abl_dir: Path, filename: str) -> list[float]:
    p = abl_dir / filename
    if not p.exists():
        return []
    rows: list[dict] = []
    with open(p) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return [r["rouge_l"] for r in rows if "rouge_l" in r]


def _sweep(
    short: str,
    stem: str,
    cache_key_fn,
    trial_fn,
) -> list[CellPoint] | None:
    abl_dir = _find_ablation_dir(short, stem)
    if abl_dir is None:
        return None
    cache = _load_cache(abl_dir, short, stem)
    points: list[CellPoint] = []
    for k in K_VALUES:
        key = cache_key_fn(short, k)
        if key not in cache:
            continue
        rouge_mean = cache[key]["rouge_l_mean"]
        trial_rouges = _load_trial_rouge(abl_dir, trial_fn(short, k))
        if trial_rouges:
            ci_lo, ci_hi = bootstrap_ci(trial_rouges)
        else:
            ci_lo = ci_hi = rouge_mean
        points.append(CellPoint(k=k, rouge=rouge_mean, ci_lo=ci_lo, ci_hi=ci_hi))
    return points or None


def locos_sweep(short: str) -> list[CellPoint] | None:
    stem = "logit_contrib_nolima"
    return _sweep(
        short,
        stem,
        cache_key_fn=lambda s, k: f"{s}__topk_{k}_mean",
        trial_fn=lambda s, k: f"nolima_ablation_{s}_{stem}_top-k_{k}p0_trials.jsonl",
    )


def alpha_spatial_sweep(short: str) -> list[CellPoint] | None:
    stem = "attention_spatial_nolima"
    return _sweep(
        short,
        stem,
        cache_key_fn=lambda s, k: f"{s}__topk_{k}_mean",
        trial_fn=lambda s, k: f"nolima_ablation_{s}_{stem}_top-k_{k}p0_trials.jsonl",
    )


def random_sweep(short: str) -> list[CellPoint] | None:
    stem = "random_seed42"
    return _sweep(
        short,
        stem,
        cache_key_fn=lambda s, k: f"{s}__topk_{k}_random_mean",
        trial_fn=lambda s, k: f"nolima_ablation_{s}_random_seed42_random_{k}p0_trials.jsonl",
    )


def baseline_point(short: str) -> CellPoint | None:
    """Load no-ablation baseline; tries LOCOS cache first, then random cache."""
    sources = [
        ("logit_contrib_nolima", f"nolima_ablation_{short}_logit_contrib_nolima_baseline_trials.jsonl"),
        ("random_seed42", f"nolima_ablation_{short}_random_seed42_baseline_trials.jsonl"),
    ]
    for stem, trial_file in sources:
        abl_dir = _find_ablation_dir(short, stem)
        if abl_dir is None:
            continue
        cache = _load_cache(abl_dir, short, stem)
        bsl_key = f"{short}__baseline"
        if bsl_key not in cache:
            continue
        rouge_mean = cache[bsl_key]["rouge_l_mean"]
        trial_rouges = _load_trial_rouge(abl_dir, trial_file)
        if trial_rouges:
            ci_lo, ci_hi = bootstrap_ci(trial_rouges)
        else:
            ci_lo = ci_hi = rouge_mean
        return CellPoint(k=0, rouge=rouge_mean, ci_lo=ci_lo, ci_hi=ci_hi)
    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def abc_d_minus_b(sweep_d: list[CellPoint], sweep_b: list[CellPoint]) -> float:
    """Area between curves (D minus B), trapezoidal over k, normalized by k range."""
    d_map = {p.k: p.rouge for p in sweep_d}
    b_map = {p.k: p.rouge for p in sweep_b}
    common_ks = sorted(set(d_map) & set(b_map))
    if len(common_ks) < 2:
        return float("nan")
    d_vals = np.array([d_map[k] for k in common_ks])
    b_vals = np.array([b_map[k] for k in common_ks])
    ks = np.array(common_ks, dtype=float)
    return float(np.trapezoid(d_vals - b_vals, ks) / (ks[-1] - ks[0]))


def separation_verdict(sweep_d: list[CellPoint], sweep_b: list[CellPoint]) -> str:
    """Per-model 3-way decision at k∈{10,20,50}."""
    d_map = {p.k: p for p in sweep_d}
    b_map = {p.k: p for p in sweep_b}
    overlap_ks: list[int] = []
    d_below_ks: list[int] = []
    d_above_ks: list[int] = []
    for k in [10, 20, 50]:
        if k not in d_map or k not in b_map:
            continue
        dp, bp = d_map[k], b_map[k]
        bands_overlap = dp.ci_hi >= bp.ci_lo and bp.ci_hi >= dp.ci_lo
        if bands_overlap:
            overlap_ks.append(k)
        elif dp.rouge < bp.rouge:
            d_below_ks.append(k)
        else:
            d_above_ks.append(k)

    n = len(overlap_ks) + len(d_below_ks) + len(d_above_ks)
    if n == 0:
        return "no_data"
    if len(overlap_ks) >= 2:
        return "falsified"
    if len(d_below_ks) >= 2 and len(d_above_ks) == 0:
        return "validated"
    if len(d_above_ks) >= 2 and len(d_below_ks) == 0:
        return "adverse"
    return "mixed"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_panel(
    ax,
    label: str,
    sweep_d: list[CellPoint] | None,
    sweep_b: list[CellPoint] | None,
    sweep_rand: list[CellPoint] | None,
    bsl: CellPoint | None,
) -> None:
    if bsl is not None:
        ax.axhline(bsl.rouge, color="black", lw=1.5, ls=":", label="Baseline (no ablation)")
        if bsl.ci_lo != bsl.ci_hi:
            ax.axhspan(bsl.ci_lo, bsl.ci_hi, color="black", alpha=0.07)

    for cell_id, sweep in [("D", sweep_d), ("B", sweep_b), ("random", sweep_rand)]:
        if sweep is None:
            continue
        ks = [p.k for p in sweep]
        rouge = [p.rouge for p in sweep]
        ci_lo = [p.ci_lo for p in sweep]
        ci_hi = [p.ci_hi for p in sweep]
        ls = "-"
        ax.plot(
            ks,
            rouge,
            color=CELL_COLORS[cell_id],
            lw=2.0,
            marker=CELL_MARKERS[cell_id],
            ms=4,
            ls=ls,
            label=CELL_LABELS[cell_id],
        )
        ax.fill_between(ks, ci_lo, ci_hi, color=CELL_COLORS[cell_id], alpha=0.15)

    ax.set_xlabel(r"$k$ (heads ablated)", fontsize=10)
    ax.set_ylabel("ROUGE-L", fontsize=10)
    ax.set_xticks(K_VALUES)
    ax.set_title(label)
    # ax.text(0.05, 0.95, label, transform=ax.transAxes, va="top", ha="left", fontsize=9, fontweight="bold")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(download: bool = True) -> None:
    setup_plot_style()
    out_dir = get_output_dir("e7")
    k_set = set(K_VALUES)

    all_data: dict[str, dict] = {}

    for model in ALL_MODELS:
        short = MODEL_SHORT[model]
        label = MODEL_LABELS[model]

        sweep_d = locos_sweep(short)
        if sweep_d is None:
            console.print(f"[yellow]SKIP {label}: LOCOS ablation cache missing[/yellow]")
            continue
        sweep_b = alpha_spatial_sweep(short)
        sweep_rand = random_sweep(short)
        bsl = baseline_point(short)

        sweep_d = [p for p in sweep_d if p.k in k_set]
        if sweep_b:
            sweep_b = [p for p in sweep_b if p.k in k_set]
        if sweep_rand:
            sweep_rand = [p for p in sweep_rand if p.k in k_set]

        all_data[model] = {
            "D": sweep_d,
            "B": sweep_b,
            "random": sweep_rand,
            "baseline": bsl,
            "label": label,
            "short": short,
        }

        csv_path = out_dir / f"e7_ablation_{short}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["k", "cell", "rouge", "ci_lo", "ci_hi"])
            writer.writeheader()
            for cell_id, sweep in [("D", sweep_d), ("B", sweep_b), ("random", sweep_rand)]:
                if sweep is None:
                    continue
                for p in sweep:
                    writer.writerow(
                        {
                            "k": p.k,
                            "cell": cell_id,
                            "rouge": f"{p.rouge:.6f}",
                            "ci_lo": f"{p.ci_lo:.6f}",
                            "ci_hi": f"{p.ci_hi:.6f}",
                        }
                    )
            if bsl is not None:
                writer.writerow(
                    {
                        "k": 0,
                        "cell": "baseline",
                        "rouge": f"{bsl.rouge:.6f}",
                        "ci_lo": f"{bsl.ci_lo:.6f}",
                        "ci_hi": f"{bsl.ci_hi:.6f}",
                    }
                )

    # --- All-models figure (2×3) and individual figures ---
    models_ok = [m for m in ALL_MODELS if m in all_data]
    if models_ok:
        # Save individual model figures
        for model in models_ok:
            d = all_data[model]
            fig_ind, ax_ind = plt.subplots(figsize=(3.5, 3))
            _plot_panel(ax_ind, d["label"], d["D"], d["B"], d["random"], d["baseline"])
            ax_ind.legend(ncol=4)
            save_figure(fig_ind, out_dir / f"e7_factorial_{d['short']}.svg", keep_title=True)

        # Save the big 2x3 grid figure
        ncols = 3
        nrows = (len(models_ok) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows))
        axes_flat = np.array(axes).flatten()
        for ax, model in zip(axes_flat, models_ok):
            d = all_data[model]
            _plot_panel(ax, d["label"], d["D"], d["B"], d["random"], d["baseline"])
        for ax in axes_flat[len(models_ok) :]:
            ax.set_visible(False)
        axes_flat[0].legend(ncol=4)
        fig.tight_layout()
        save_figure(fig, out_dir / "e7_factorial.svg", keep_title=True)

    # --- Console summary ---
    table = Table(title="E7 — Cell B vs D: OV-component ablation audit")
    table.add_column("Model")
    table.add_column("ABC(D-B)", justify="right")
    table.add_column("Verdict", justify="right")
    table.add_column("B?", justify="center")

    for model in ALL_MODELS:
        if model not in all_data:
            continue
        d = all_data[model]
        abc = abc_d_minus_b(d["D"], d["B"]) if d["B"] else float("nan")
        verdict = separation_verdict(d["D"], d["B"]) if d["B"] else "no_cell_B"
        color_map = {
            "validated": "green",
            "falsified": "red",
            "adverse": "yellow",
            "mixed": "yellow",
            "no_data": "dim",
            "no_cell_B": "dim",
        }
        color = color_map.get(verdict, "white")
        abc_str = f"{abc:+.4f}" if not np.isnan(abc) else "—"
        table.add_row(
            d["label"],
            abc_str,
            f"[{color}]{verdict}[/{color}]",
            "[green]✓[/green]" if d["B"] else "[red]✗[/red]",
        )
    console.print(table)

    console.print("\n[bold]E7 per-headline-model decision[/bold]")
    for model in HEADLINE_MODELS:
        if model not in all_data:
            console.print(f"  {MODEL_LABELS[model]}: [dim]no data[/dim]")
            continue
        d = all_data[model]
        if not d["B"]:
            console.print(f"  {d['label']}: [dim]cell B missing[/dim]")
            continue
        verdict = separation_verdict(d["D"], d["B"])
        abc = abc_d_minus_b(d["D"], d["B"])
        abc_str = f"ABC(D-B)={abc:+.4f}"
        if verdict == "validated":
            console.print(f"  {d['label']}: [green]D strictly below B ({abc_str}) — OV component validated[/green]")
        elif verdict == "falsified":
            console.print(f"  {d['label']}: [red]D ≈ B ({abc_str}) — OV claim falsified; α-spatial suffices[/red]")
        elif verdict == "adverse":
            console.print(
                f"  {d['label']}: [yellow]D above B ({abc_str}) — adverse result; α-spatial identifies "
                f"more disruptive heads[/yellow]"
            )
        else:
            console.print(f"  {d['label']}: [yellow]mixed ({abc_str}) — check per-k CIs[/yellow]")

    console.print(f"\n[dim]Saved to: {out_dir}/[/dim]")
    console.print("[dim]Cells A, C skipped (A: needs scoring re-run; C: blocked by V2)[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E7 — 2×2 factorial ablation audit (cells B and D).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--no-download", action="store_true", help="Use local files only.")
    args = parser.parse_args()
    run(download=not args.no_download)


if __name__ == "__main__":
    main()
