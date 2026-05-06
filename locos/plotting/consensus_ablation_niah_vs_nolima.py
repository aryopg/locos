#!/usr/bin/env python3
"""Paper figure for H1: NIAH-vs-NoLiMa paired ablation curves.

Consumes ``ablation_results/{niah,nolima}_ablation_<model>_<set_kind>_k<kpm>.json``
files produced by ``deploy/jobs/ablation_{niah,nolima}_consensus.sh`` and
plots, for each scoped model, ROUGE-L at the consensus / wu-only / lc-only
ablation sets across k ∈ {10, 20, 50} — NIAH as solid lines, NoLiMa as
dashed. The goal is instant visual legibility: if consensus ablation crushes
NIAH but barely moves NoLiMa, H1 is confirmed.

Usage:
    python -m locos.plotting.consensus_ablation_niah_vs_nolima \\
        --results-dir ablation_results \\
        --output figures/consensus_ablation_niah_vs_nolima.svg
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from locos_eval.utils.plotting import LINE_WIDTH, MODEL_PRETTY_NAMES, save_figure, setup_plot_style

# Maps our three set kinds to a visually distinct palette.
SET_COLOURS = {
    "consensus": "#C44E52",  # red — the headline claim
    "wu_only": "#4C72B0",  # blue — method-A specific
    "lc_only": "#55A868",  # green — method-B specific
}
SET_PRETTY = {
    "consensus": r"Consensus (Wu $\cap$ LC)",
    "wu_only": "Wu-only",
    "lc_only": "LC-only",
}
DATASET_LINESTYLE = {"niah": "-.", "nolima": "-"}
DATASET_PRETTY = {"niah": "NIAH", "nolima": "NoLiMa"}

# Filename regex: e.g. "niah_ablation_Qwen3-8B_consensus_k20.json"
CACHE_RE = re.compile(
    r"^(?P<dataset>niah|nolima)_ablation_(?P<model>.+)_(?P<set_kind>consensus|wu_only|lc_only)_k(?P<k>\d+)\.json$"
)


def discover_cells(results_dir: Path) -> list[dict]:
    cells: list[dict] = []
    for p in sorted(results_dir.glob("*.json")):
        m = CACHE_RE.match(p.name)
        if not m:
            continue
        with open(p) as f:
            payload = json.load(f)
        # payload: {run_key: metrics}. Pick the single non-baseline topk entry.
        # FIXME(aryo): if a future run writes multiple topk entries into the
        # same cache file we silently average them. For now each consensus
        # cache file corresponds to exactly one set size.
        rouge_ls = []
        baseline_rl = None
        for run_key, metrics in payload.items():
            if not isinstance(metrics, dict) or "rouge_l_mean" not in metrics:
                continue
            if "baseline" in run_key:
                baseline_rl = float(metrics["rouge_l_mean"])
            else:
                rouge_ls.append(float(metrics["rouge_l_mean"]))
        if not rouge_ls:
            continue
        cells.append(
            {
                "dataset": m.group("dataset"),
                "model": m.group("model"),
                "set_kind": m.group("set_kind"),
                "k_per_method": int(m.group("k")),
                "rouge_l_mean": sum(rouge_ls) / len(rouge_ls),
                "baseline_rouge_l": baseline_rl,
            }
        )
    return cells


def discover_random_cells(results_dir: Path) -> list[dict]:
    """Parse random-head ablation caches for control bands.

    Matches ``{dataset}_ablation_{model}_random_seed{seed}.json``.
    """
    rand_re = re.compile(r"^(?P<dataset>niah|nolima)_ablation_(?P<model>.+)_random_seed(?P<seed>\d+)\.json$")
    out: list[dict] = []
    for p in sorted(results_dir.glob("*.json")):
        m = rand_re.match(p.name)
        if not m:
            continue
        with open(p) as f:
            payload = json.load(f)
        for run_key, metrics in payload.items():
            if not isinstance(metrics, dict) or "rouge_l_mean" not in metrics:
                continue
            if "baseline" in run_key:
                continue
            k = int(metrics.get("n_heads") or 0)
            out.append(
                {
                    "dataset": m.group("dataset"),
                    "model": m.group("model"),
                    "seed": int(m.group("seed")),
                    "n_heads": k,
                    "rouge_l_mean": float(metrics["rouge_l_mean"]),
                }
            )
    return out


def short_model_name(model: str) -> str:
    slug = model.split("/")[-1]
    return MODEL_PRETTY_NAMES.get(slug, slug)


def plot(cells: list[dict], random_cells: list[dict], out_path: Path) -> None:
    if not cells:
        raise ValueError("No consensus ablation cache files found. Did you run the consensus jobs?")

    setup_plot_style()

    models = sorted({c["model"] for c in cells}, key=short_model_name)
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.2 * n_rows), sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    for ax_idx, model in enumerate(models):
        ax = axes_flat[ax_idx]

        # Random-head control band per dataset (min/max across seeds at any k).
        # axhspan does not accept linestyle, and the printed figure needs the
        # band to stay legible, so the two datasets get two different shades
        # instead of solid-vs-dashed.
        for idx, dataset in enumerate(("niah", "nolima")):
            rand_vals = [r["rouge_l_mean"] for r in random_cells if r["model"] == model and r["dataset"] == dataset]
            if rand_vals:
                lo, hi = min(rand_vals), max(rand_vals)
                ax.axhspan(lo, hi, color="#888888", alpha=0.22 if idx == 0 else 0.12)

        # One line per (set_kind, dataset)
        for set_kind in ("consensus", "wu_only", "lc_only"):
            for dataset in ("niah", "nolima"):
                pts = sorted(
                    [c for c in cells if c["model"] == model and c["set_kind"] == set_kind and c["dataset"] == dataset],
                    key=lambda c: c["k_per_method"],
                )
                if not pts:
                    continue
                xs = [c["k_per_method"] for c in pts]
                ys = [c["rouge_l_mean"] for c in pts]
                ax.plot(
                    xs,
                    ys,
                    linestyle=DATASET_LINESTYLE[dataset],
                    color=SET_COLOURS[set_kind],
                    marker="o" if dataset == "niah" else "s",
                    linewidth=LINE_WIDTH,
                    markerfacecolor="white",
                )

        ax.set_xlabel(r"$k$ per method (Wu-NIAH $\cap$ LC-NoLiMa)")
        ax.set_xticks([10, 20, 50])
        ax.text(
            0.03,
            0.04,
            short_model_name(model),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8, "pad": 2.0},
        )

    for row_idx in range(n_rows):
        axes[row_idx, 0].set_ylabel("ROUGE-L")

    for ax_idx in range(n_models, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    # Construct a combined colour (set-kind) × linestyle (dataset) legend on the
    # first axis — save_figure strips and re-saves it as a sibling SVG.
    legend_handles: list = []
    for set_kind, color in SET_COLOURS.items():
        legend_handles.append(mlines.Line2D([], [], color=color, linewidth=LINE_WIDTH, label=SET_PRETTY[set_kind]))
    for dataset, style in DATASET_LINESTYLE.items():
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                linewidth=LINE_WIDTH,
                linestyle=style,
                marker="o" if dataset == "niah" else "s",
                markerfacecolor="white",
                label=DATASET_PRETTY[dataset],
            )
        )
    axes_flat[0].legend(handles=legend_handles, loc="upper right", frameon=True)

    fig.tight_layout()
    save_figure(fig, out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("ablation_results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cells = discover_cells(args.results_dir)
    random_cells = discover_random_cells(args.results_dir)
    plot(cells, random_cells, args.output)
    print(f"Wrote {args.output} ({len(cells)} consensus cells, {len(random_cells)} random controls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
