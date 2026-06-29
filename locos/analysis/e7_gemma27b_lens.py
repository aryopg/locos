#!/usr/bin/env python3
"""E7 appendix — Gemma3-27B tuned-lens diagnostic.

Plots four ablation curves for Gemma3-27B:
  D  : φ-spatial (LOCOS)
  B  : α-spatial
  L  : φ-spatial (lens-corrected LOCOS)
  R  : random baseline

Used to diagnose whether LOCOS's "mixed" verdict on Gemma3-27B is a
direct-path-approximation failure (corrected by tuned-lens) or a genuine
result (attention placement more causally informative than OV-weighted score).

Output:
    analysis/outputs/e7/e7_gemma27b_lens.svg
    analysis/outputs/e7/e7_gemma27b_lens_legend.svg

Usage:
    python locos/analysis/e7_gemma27b_lens.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
from rich.console import Console

from locos_eval.utils.plotting import save_figure, setup_plot_style
from locos.analysis._utils import ABLATION_K_VALUES, bootstrap_ci, get_output_dir

console = Console()

SHORT = "gemma-3-27b-it"
K_VALUES = [k for k in ABLATION_K_VALUES if k <= 50]  # [1, 5, 10, 20, 50]

_LOCOS_RESULTS = Path(os.environ.get("LOCOS_RESULTS_DIR", str(_REPO_ROOT.parent / "locos-results")))
_ABLATION_DIRS: list[Path] = [
    _REPO_ROOT / "ablation_results",
    _LOCOS_RESULTS / "ablation_results",
]

COLORS = {
    "D": "#0763BA",
    "B": "#41A9AC",
    "L": "#9B59B6",
    "random": "#DF922D",
}
LABELS = {
    "D": r"$\phi$-spatial (LOCOS)",
    "B": r"$\alpha$-spatial",
    "L": "LOCOS + Tuned-Lens",
    "random": "Random",
}
MARKERS = {"D": "o", "B": "P", "L": "D", "random": "^"}


class CellPoint(NamedTuple):
    k: int
    rouge: float
    ci_lo: float
    ci_hi: float


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _find_dir(stem: str) -> Path | None:
    for d in _ABLATION_DIRS:
        if (d / f"nolima_ablation_{SHORT}_{stem}.json").exists():
            return d
    return None


def _load_cache(d: Path, stem: str) -> dict:
    with open(d / f"nolima_ablation_{SHORT}_{stem}.json") as f:
        return json.load(f)


def _load_trial_rouge(d: Path, filename: str) -> list[float]:
    p = d / filename
    if not p.exists():
        return []
    rows: list[float] = []
    with open(p) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if "rouge_l" in obj:
                    rows.append(obj["rouge_l"])
    return rows


def _sweep(stem: str, key_fn, trial_fn) -> list[CellPoint] | None:
    d = _find_dir(stem)
    if d is None:
        return None
    cache = _load_cache(d, stem)
    points: list[CellPoint] = []
    for k in K_VALUES:
        key = key_fn(k)
        if key not in cache:
            continue
        mean = cache[key]["rouge_l_mean"]
        trials = _load_trial_rouge(d, trial_fn(k))
        ci_lo, ci_hi = bootstrap_ci(trials) if trials else (mean, mean)
        points.append(CellPoint(k=k, rouge=mean, ci_lo=ci_lo, ci_hi=ci_hi))
    return points or None


def locos_sweep() -> list[CellPoint] | None:
    stem = "logit_contrib_nolima"
    return _sweep(
        stem,
        key_fn=lambda k: f"{SHORT}__topk_{k}_mean",
        trial_fn=lambda k: f"nolima_ablation_{SHORT}_{stem}_top-k_{k}p0_trials.jsonl",
    )


def alpha_spatial_sweep() -> list[CellPoint] | None:
    stem = "attention_spatial_nolima"
    return _sweep(
        stem,
        key_fn=lambda k: f"{SHORT}__topk_{k}_mean",
        trial_fn=lambda k: f"nolima_ablation_{SHORT}_{stem}_top-k_{k}p0_trials.jsonl",
    )


def lens_sweep() -> list[CellPoint] | None:
    stem = "logit_contrib_nolima_tuned_lens"
    return _sweep(
        stem,
        key_fn=lambda k: f"{SHORT}__topk_{k}_mean",
        trial_fn=lambda k: f"nolima_ablation_{SHORT}_{stem}_top-k_{k}p0_trials.jsonl",
    )


def random_sweep() -> list[CellPoint] | None:
    stem = "random_seed42"
    return _sweep(
        stem,
        key_fn=lambda k: f"{SHORT}__topk_{k}_random_mean",
        trial_fn=lambda k: f"nolima_ablation_{SHORT}_random_seed42_random_{k}p0_trials.jsonl",
    )


def baseline_point() -> CellPoint | None:
    for stem, trial_file in [
        ("logit_contrib_nolima", f"nolima_ablation_{SHORT}_logit_contrib_nolima_baseline_trials.jsonl"),
        ("random_seed42", f"nolima_ablation_{SHORT}_random_seed42_baseline_trials.jsonl"),
    ]:
        d = _find_dir(stem)
        if d is None:
            continue
        cache = _load_cache(d, stem)
        key = f"{SHORT}__baseline"
        if key not in cache:
            continue
        mean = cache[key]["rouge_l_mean"]
        trials = _load_trial_rouge(d, trial_file)
        ci_lo, ci_hi = bootstrap_ci(trials) if trials else (mean, mean)
        return CellPoint(k=0, rouge=mean, ci_lo=ci_lo, ci_hi=ci_hi)
    return None


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def run() -> None:
    setup_plot_style()
    out_dir = get_output_dir("e7")

    sweep_d = locos_sweep()
    sweep_b = alpha_spatial_sweep()
    sweep_l = lens_sweep()
    sweep_r = random_sweep()
    bsl = baseline_point()

    if sweep_d is None:
        console.print("[red]LOCOS cache missing for Gemma3-27B — aborting.[/red]")
        return

    fig, ax = plt.subplots(figsize=(3.5, 3))

    if bsl is not None:
        ax.axhline(bsl.rouge, color="black", lw=1.5, ls=":", label="Baseline (no ablation)")
        if bsl.ci_lo != bsl.ci_hi:
            ax.axhspan(bsl.ci_lo, bsl.ci_hi, color="black", alpha=0.07)

    for cell_id, sweep in [("D", sweep_d), ("B", sweep_b), ("L", sweep_l), ("random", sweep_r)]:
        if sweep is None:
            console.print(f"[yellow]  {cell_id} ({LABELS[cell_id]}): missing, skipped[/yellow]")
            continue
        ks = [p.k for p in sweep]
        rouge = [p.rouge for p in sweep]
        ci_lo = [p.ci_lo for p in sweep]
        ci_hi = [p.ci_hi for p in sweep]
        ax.plot(ks, rouge, color=COLORS[cell_id], lw=2.0, marker=MARKERS[cell_id], ms=4, label=LABELS[cell_id])
        ax.fill_between(ks, ci_lo, ci_hi, color=COLORS[cell_id], alpha=0.15)

    ax.set_xlabel(r"$k$ (heads ablated)", fontsize=10)
    ax.set_ylabel("ROUGE-L", fontsize=10)
    ax.set_xticks(K_VALUES)
    ax.set_title("Gemma3-27B")
    # ax.legend(ncol=5)
    ax.legend()

    out_path = out_dir / "e7_gemma27b_lens.svg"
    save_figure(fig, out_path, keep_title=True)
    console.print(f"[green]Saved:[/green] {out_path}")
    console.print(f"[green]Saved:[/green] {out_dir / 'e7_gemma27b_lens_legend.svg'}")


if __name__ == "__main__":
    run()
