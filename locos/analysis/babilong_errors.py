#!/usr/bin/env python3
"""E6 — BABILong stale-copy error analysis.

Hypothesis: Wu-head ablation improves BABILong on some models because Wu heads
are copy heads and the dominant error is copying a stale (earlier-valid) location.

Models with Wu ablation improvement: qwen3-14b, gemma3-27b, olmo3.1-32b.
Conditions: {baseline, wu_ablated, locos_ablated, random_ablated}.
Tasks: babilong_qa2_0k, babilong_qa3_0k.

Error categories:
    stale       — output matches an earlier valid location for the entity
    other_entity — location valid for a different entity in context
    hallucinated — location not mentioned in context at all
    format_other — miscellaneous (wrong format, empty, etc.)

Statistical test: two-proportion z-test (wu_ablated vs random_ablated),
Holm-corrected over 3 models, α=0.05.

Outputs:
    analysis/outputs/e6/e6_error_composition.svg
    analysis/outputs/e6/e6_error_composition_legend.svg
    analysis/outputs/e6/e6_errors.csv
    analysis/outputs/e6/e6_ztest.csv

Usage:
    python locos/analysis/babilong_errors.py
    python locos/analysis/babilong_errors.py --no-download
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.stats import norm as scipy_norm

from locos.analysis._utils import (
    E6_AFFECTED_MODELS,
    MODEL_LABELS,
    get_output_dir,
    load_downstream_from_hf,
    load_eval_rows,
)
from locos_eval.utils.plotting import save_figure, setup_plot_style

console = Console()

CONDITIONS = [
    ("baseline", ["greedy"], "Baseline"),
    ("wu_ablated", ["ablation_wu_nolima", "ablation_wu_niah"], "Wu-ablated"),
    ("locos_ablated", ["ablation_logitcontrib_nolima", "ablation_locos_nolima"], "LOCOS-ablated"),
    ("random_ablated", ["ablation_random_s42_n50", "ablation_random"], "Random-ablated"),
]
TASK_SUBSETS = ["babilong_qa2_0k", "babilong_qa3_0k"]
CATEGORIES = ["stale", "other_entity", "hallucinated", "format_other"]
CAT_COLORS = {
    "stale": "#1f77b4",
    "other_entity": "#ff7f0e",
    "hallucinated": "#2ca02c",
    "format_other": "#d62728",
}


# ---------------------------------------------------------------------------
# Entity trajectory parsing (bAbI-style stories)
# ---------------------------------------------------------------------------

_MOVE_PATTERNS = [
    re.compile(r"(\w+)\s+(?:went|moved|journeyed|travelled|went\s+back)\s+to\s+the\s+(\w+)", re.I),
]
_CARRY_PATTERNS = [
    re.compile(r"(\w+)\s+(?:grabbed|picked\s+up|got|took)\s+the\s+(\w+)", re.I),
    re.compile(r"(\w+)\s+(?:dropped|left|put\s+down)\s+the\s+(\w+)", re.I),
]


def _parse_entity_trajectory(story: str) -> dict[str, list[str]]:
    """Extract entity → ordered list of locations from bAbI-style story.

    Returns {entity_lower: [loc1, loc2, ...]} where list is chronological.
    """
    trajectory: dict[str, list[str]] = {}
    for line in story.split("\n"):
        line = line.strip()
        for pat in _MOVE_PATTERNS:
            m = pat.search(line)
            if m:
                entity = m.group(1).lower()
                location = m.group(2).lower()
                trajectory.setdefault(entity, []).append(location)
    return trajectory


def _all_locations_in_story(story: str) -> set[str]:
    """All location tokens that appear in move-pattern matches."""
    locs = set()
    for pat in _MOVE_PATTERNS:
        for m in pat.finditer(story):
            locs.add(m.group(2).lower())
    return locs


def _classify_error(
    output: str,
    target: str,
    story: str,
    question: str,
) -> str:
    """Classify an incorrect BABILong answer into one of 4 error categories."""
    output_clean = output.strip().lower()
    # Extract <answer>...</answer> tag if present
    tag_m = re.search(r"<answer>(.*?)</answer>", output, re.I | re.DOTALL)
    if tag_m:
        output_clean = tag_m.group(1).strip().lower()

    if not output_clean:
        return "format_other"

    # Parse entity from question (e.g. "Where is John?" → "john")
    q_entity_m = re.search(r"where\s+is\s+(\w+)", question, re.I)
    entity_name = q_entity_m.group(1).lower() if q_entity_m else None

    trajectory = _parse_entity_trajectory(story)
    all_locs = _all_locations_in_story(story)

    if entity_name and entity_name in trajectory:
        entity_locs = trajectory[entity_name]
        # Current location = target; prior locations = everything before last
        prior_locs = entity_locs[:-1] if len(entity_locs) > 1 else []
        if any(output_clean in loc or loc in output_clean for loc in prior_locs):
            return "stale"

    # Check if output is a valid location for a *different* entity
    for ent, locs in trajectory.items():
        if ent != entity_name and any(output_clean in loc or loc in output_clean for loc in locs):
            return "other_entity"

    # Check if output mentions any location in the story
    if any(output_clean in loc or loc in output_clean for loc in all_locs):
        return "other_entity"  # location exists but not correct entity/time

    # Output not found in story at all
    if output_clean in story.lower():
        return "hallucinated"  # mentioned in story but not as a location

    return "hallucinated"


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------


def _load_results(task_key: str, model: str, variants: list[str], download: bool) -> list[dict] | None:
    rows = load_eval_rows(task_key, model, variants)
    if rows:
        return rows
    if download:
        return load_downstream_from_hf(task_key, model, variants)
    return None


def _load_babilong_dataset(subset: str, split: str) -> list[dict]:
    """Load original BABILong stories from HF for entity trajectory parsing."""
    try:
        from datasets import load_dataset

        ds = load_dataset("RMT-team/babilong", split, split=subset)
        return [{"input": row["input"], "question": row["question"], "target": row["target"]} for row in ds]
    except Exception as e:
        console.print(f"[yellow]WARNING: Could not load BABILong dataset: {e}[/yellow]")
        return []


# ---------------------------------------------------------------------------
# Two-proportion z-test with Holm correction
# ---------------------------------------------------------------------------


def _two_prop_ztest(n1: int, p1: float, n2: int, p2: float) -> tuple[float, float]:
    """Two-proportion z-test. Returns (z, p_value)."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p_pool = (n1 * p1 + n2 * p2) / (n1 + n2)
    se = (p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) ** 0.5
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    p_val = 2 * (1 - scipy_norm.cdf(abs(z)))
    return z, p_val


def _holm_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni correction. Returns list of reject booleans."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    reject = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (n - rank)
        if p <= threshold:
            reject[orig_idx] = True
        else:
            break
    return reject


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(download: bool = True) -> None:
    setup_plot_style()
    import matplotlib.pyplot as plt

    out_dir = get_output_dir("e6")

    # Load BABILong dataset for story parsing
    story_cache: dict[tuple[str, int], dict] = {}
    for task_key in TASK_SUBSETS:
        parts = task_key.split("_")  # babilong, qa2, 0k
        subset = parts[1]  # qa2 / qa3
        split = parts[2]  # 0k
        stories = _load_babilong_dataset(subset, split)
        for i, s in enumerate(stories):
            story_cache[(task_key, i)] = s

    # Collect per-(model, condition, task) error distributions
    all_data: dict[str, dict[str, dict[str, list[str]]]] = {}
    # all_data[model][condition_key][task_key] = [error_cat, ...]

    for model in E6_AFFECTED_MODELS:
        model_label = MODEL_LABELS[model]
        all_data[model_label] = {}

        for _cond_key, variants, cond_label in CONDITIONS:
            all_data[model_label][cond_label] = {}

            for task_key in TASK_SUBSETS:
                results = _load_results(task_key, model, variants, download)
                if results is None:
                    console.print(f"[yellow]SKIP {model_label}/{cond_label}/{task_key}: not found[/yellow]")
                    all_data[model_label][cond_label][task_key] = []
                    continue

                errors: list[str] = []
                for row in results:
                    acc = row.get("scores", {}).get("accuracy", 1.0)
                    if acc >= 0.5:
                        continue  # correct answer — skip
                    sample_id = row.get("sample_id", 0)
                    story_data = story_cache.get((task_key, sample_id), {})
                    story = story_data.get("input", row.get("metadata", {}).get("story", ""))
                    question = story_data.get("question", row.get("metadata", {}).get("question", ""))
                    target = row.get("target", "")
                    output = row.get("output", "")

                    cat = _classify_error(output, target, story, question)
                    errors.append(cat)

                all_data[model_label][cond_label][task_key] = errors

    # Aggregate across task subsets
    csv_rows: list[dict] = []

    for model_label in all_data:
        for cond_label in all_data[model_label]:
            combined: list[str] = []
            for task_key in TASK_SUBSETS:
                combined.extend(all_data[model_label][cond_label].get(task_key, []))
            total = len(combined)
            cat_counts = {c: combined.count(c) for c in CATEGORIES}
            for cat, cnt in cat_counts.items():
                csv_rows.append(
                    {
                        "model": model_label,
                        "condition": cond_label,
                        "category": cat,
                        "count": cnt,
                        "total_errors": total,
                        "fraction": f"{cnt/total:.4f}" if total > 0 else "0",
                    }
                )

    csv_path = out_dir / "e6_errors.csv"
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)

    # --- Figure: stacked bars per condition × model ---
    n_models = len(E6_AFFECTED_MODELS)
    n_conds = len(CONDITIONS)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4.0))
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, E6_AFFECTED_MODELS):
        model_label = MODEL_LABELS[model]
        cond_labels = [c[2] for c in CONDITIONS]
        bottoms = np.zeros(n_conds)
        for cat in CATEGORIES:
            fracs = []
            for cond_label in cond_labels:
                combined = []
                for task_key in TASK_SUBSETS:
                    combined.extend(all_data.get(model_label, {}).get(cond_label, {}).get(task_key, []))
                total = len(combined)
                fracs.append(combined.count(cat) / total if total > 0 else 0.0)
            fracs_arr = np.array(fracs)
            ax.bar(
                cond_labels,
                fracs_arr,
                bottom=bottoms,
                color=CAT_COLORS[cat],
                label=cat,
                edgecolor="black",
                linewidth=0.8,
            )
            bottoms += fracs_arr

        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of incorrect answers" if model == E6_AFFECTED_MODELS[0] else "")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.text(0.05, 0.97, model_label, transform=ax.transAxes, va="top", fontsize=9, fontweight="bold")

    if len(axes) > 0:
        handles = [plt.Rectangle((0, 0), 1, 1, fc=CAT_COLORS[c]) for c in CATEGORIES]
        axes[0].legend(handles, CATEGORIES, fontsize=7, loc="upper right")

    fig.tight_layout()
    save_figure(fig, out_dir / "e6_error_composition.svg")

    # --- Statistical tests: wu_ablated vs random_ablated on stale errors ---
    p_values_raw = []
    ztest_rows: list[dict] = []
    for model in E6_AFFECTED_MODELS:
        model_label = MODEL_LABELS[model]
        wu_errors, rand_errors = [], []
        for task_key in TASK_SUBSETS:
            wu_errors.extend(all_data.get(model_label, {}).get("Wu-ablated", {}).get(task_key, []))
            rand_errors.extend(all_data.get(model_label, {}).get("Random-ablated", {}).get(task_key, []))
        n_wu = len(wu_errors)
        n_rand = len(rand_errors)
        stale_wu = wu_errors.count("stale") / n_wu if n_wu > 0 else float("nan")
        stale_rand = rand_errors.count("stale") / n_rand if n_rand > 0 else float("nan")
        z, p = _two_prop_ztest(n_wu, stale_wu, n_rand, stale_rand)
        p_values_raw.append(p)
        ztest_rows.append(
            {
                "model": model_label,
                "n_wu_errors": n_wu,
                "stale_rate_wu": f"{stale_wu:.4f}",
                "n_rand_errors": n_rand,
                "stale_rate_rand": f"{stale_rand:.4f}",
                "z_stat": f"{z:.4f}",
                "p_raw": f"{p:.4f}",
            }
        )

    reject = _holm_correction([r for r in p_values_raw if not np.isnan(r)], alpha=0.05)
    for i, row in enumerate(ztest_rows):
        row["reject_holm_005"] = str(reject[i] if i < len(reject) else False)

    ztest_path = out_dir / "e6_ztest.csv"
    if ztest_rows:
        with open(ztest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(ztest_rows[0]))
            writer.writeheader()
            writer.writerows(ztest_rows)

    # Console table
    table = Table(title="E6 — Stale-Error Z-Tests (Wu vs Random ablation)")
    table.add_column("Model")
    table.add_column("Stale/Wu", justify="right")
    table.add_column("Stale/Rand", justify="right")
    table.add_column("z", justify="right")
    table.add_column("p (raw)", justify="right")
    table.add_column("Reject (Holm)", justify="center")
    for row in ztest_rows:
        table.add_row(
            row["model"],
            row["stale_rate_wu"],
            row["stale_rate_rand"],
            row["z_stat"],
            row["p_raw"],
            row["reject_holm_005"],
        )
    console.print(table)

    any_reject = any(r == "True" for r in [row["reject_holm_005"] for row in ztest_rows])
    if any_reject:
        console.print(
            "[green]→ E6 DECISION: Confirmed — Wu ablation suppresses stale-copy errors. "
            "Add one paragraph citing copy-suppression literature.[/green]"
        )
    else:
        console.print("[yellow]→ E6 DECISION: Not confirmed. State explicitly with rejected hypothesis.[/yellow]")

    console.print(f"\n[dim]Saved:[/dim] {out_dir}/e6_error_composition.svg, " f"e6_errors.csv, e6_ztest.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E6 — BABILong stale-copy error analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    run(download=not args.no_download)


if __name__ == "__main__":
    main()
