#!/usr/bin/env python3
"""Radar plot of LongBench-v2 accuracy by domain across decoding variants.

Reads downstream eval results under ``<results-root>/longbench_v2_short/<model>/<variant>_s<seed>/``
and produces a 2×3 grid of radar charts — one subplot per model — with one
polygon per decoding variant (greedy / ablation_wu_niah / ablation_random /
ablation_logitcontrib) over 6 domain axes.

Seed handling: per (model, variant, domain) we mean accuracy_compensated
across seeds. The radar shows means only; per-seed mean/std/min/max are
written to a sibling CSV for downstream use (e.g. bar plot with error bars).

Usage:
    python locos/plotting/longbench_v2_radar.py \
        --results-root ../locos-results/downstream_results \
        --task longbench_v2_short \
        --out figures/longbench_v2_radar.svg
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.table import Table

from locos.plotting._paths import default_downstream_results_root
from locos_eval.utils.plotting import MODEL_PRETTY_NAMES, save_figure, setup_plot_style

console = Console()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Variants to render (in legend order). Each maps a *family* prefix to a
# pretty label and a base color from the tab10 palette.
_TAB10 = sns.color_palette("tab10")
VARIANT_FAMILIES: list[tuple[str, str, tuple]] = [
    ("greedy", "Greedy", (0.45, 0.45, 0.45)),  # baseline grey
    ("ablation_logitcontrib_nolima", "Ablate LOCOS (NoLiMa)", _TAB10[0]),  # blue
    ("ablation_wu_niah", "Ablate Wu (NIAH)", _TAB10[2]),  # green
    ("ablation_random", "Ablate random", _TAB10[3]),  # red
]

# Model display order across the 2×3 grid (left-right, top-bottom).
MODEL_ORDER = [
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "Olmo-3.1-32B-Instruct",
]

METRIC = "accuracy_compensated"


# ---------------------------------------------------------------------------
# Discovery & loading
# ---------------------------------------------------------------------------

# Variant directory format: <family>_s<seed> (greedy_s1, ablation_wu_niah_s2,
# ablation_random_s42_n50_s1, ablation_logitcontrib_nolima_s3).
_VARIANT_RE = re.compile(r"^(?P<family>.+)_s(?P<seed>\d+)$")


def parse_variant_dir(name: str) -> tuple[str, int] | None:
    """Return (family, seed) for a variant directory, or None if it doesn't parse."""
    m = _VARIANT_RE.match(name)
    if not m:
        return None
    return m.group("family"), int(m.group("seed"))


def family_to_canonical(family: str) -> str | None:
    """Map a parsed family string to one of the canonical VARIANT_FAMILIES keys."""
    for key, _, _ in VARIANT_FAMILIES:
        if family == key or family.startswith(key + "_"):
            return key
    return None


def load_results_jsonl(variant_dir: Path) -> list[dict] | None:
    """Load the latest results_*.jsonl in a variant directory.

    Returns None if no scored results are present.
    """
    candidates = sorted(variant_dir.glob("results_*.jsonl"))
    candidates = [p for p in candidates if not p.name.endswith("_config.json")]
    if not candidates:
        return None
    # Use the most recent (lexicographic timestamp == chronological).
    chosen = candidates[-1]
    rows: list[dict] = []
    with open(chosen) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def per_domain_accuracy(rows: list[dict], metric: str) -> dict[str, float]:
    """Compute mean ``metric`` per domain for a single seed run."""
    by_domain: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        m = r.get("metadata", {})
        domain = m.get("domain")
        if domain is None:
            continue
        score = r.get("scores", {}).get(metric)
        if score is None:
            continue
        by_domain[domain].append(float(score))
    return {d: float(np.mean(v)) for d, v in by_domain.items() if v}


def domain_sample_counts(rows: list[dict]) -> dict[str, int]:
    """Count samples per domain (for axis annotations)."""
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        d = r.get("metadata", {}).get("domain")
        if d is not None:
            counts[d] += 1
    return dict(counts)


def discover(results_root: Path, task: str) -> dict[str, dict[str, list[tuple[int, dict[str, float]]]]]:
    """Walk the results tree and collect per-seed per-domain accuracies.

    Returns ``{model: {family: [(seed, {domain: acc}), ...]}}``.
    Models are stored under their *short* HuggingFace name (e.g. "Qwen3-8B"),
    derived from "<org>_<model>" directory names.
    """
    task_dir = results_root / task
    assert task_dir.is_dir(), f"Task dir not found: {task_dir}"

    out: dict[str, dict[str, list[tuple[int, dict[str, float]]]]] = defaultdict(lambda: defaultdict(list))

    for model_dir in sorted(task_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        # "google_gemma-3-12b-it" → "gemma-3-12b-it" (strip org prefix at first "_")
        short = model_dir.name.split("_", 1)[-1]
        for variant_dir in sorted(model_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            parsed = parse_variant_dir(variant_dir.name)
            if parsed is None:
                console.print(f"[yellow]Skip unparseable variant dir: {variant_dir}[/yellow]")
                continue
            family, seed = parsed
            canonical = family_to_canonical(family)
            if canonical is None:
                console.print(f"[yellow]Skip unknown variant family {family!r} in {variant_dir}[/yellow]")
                continue
            rows = load_results_jsonl(variant_dir)
            if not rows:
                console.print(f"[yellow]No results in {variant_dir}[/yellow]")
                continue
            scores = per_domain_accuracy(rows, METRIC)
            out[short][canonical].append((seed, scores))
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_long_form(
    discovered: dict[str, dict[str, list[tuple[int, dict[str, float]]]]],
) -> pd.DataFrame:
    """Flatten discovery into a long-form DataFrame with one row per seed."""
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
    return pd.DataFrame.from_records(records)


def aggregate_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / min / max / n_seeds per (model, variant, domain)."""
    agg = (
        long_df.groupby(["model", "variant", "domain"])["accuracy_compensated"]
        .agg(["mean", "std", "min", "max", "count"])
        .rename(columns={"count": "n_seeds"})
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    return agg


# ---------------------------------------------------------------------------
# Radar plotting
# ---------------------------------------------------------------------------


def _radar_axes(num_vars: int) -> np.ndarray:
    """Return angles (radians) for ``num_vars`` axes, closing the polygon."""
    angles = np.linspace(0.0, 2 * np.pi, num_vars, endpoint=False)
    return np.concatenate([angles, angles[:1]])


def _short_domain_label(domain: str, n: int) -> str:
    """Compact two-line label for a radar axis."""
    short_map = {
        "Single-Document QA": "Single-Doc QA",
        "Multi-Document QA": "Multi-Doc QA",
        "Long-dialogue History Understanding": "Long-Dialogue",
        "Long Structured Data Understanding": "Long Structured",
        "Code Repository Understanding": "Code Repo",
        "Long In-context Learning": "Long ICL",
    }
    pretty = short_map.get(domain, domain)
    return f"{pretty}\n(n={n})"


def make_radar_grid(
    summary: pd.DataFrame,
    domain_order: list[str],
    domain_counts: dict[str, int],
    out_path: Path,
) -> None:
    """Render the 2×3 radar grid and save it."""
    setup_plot_style()
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"

    angles = _radar_axes(len(domain_order))
    axis_labels = [_short_domain_label(d, domain_counts.get(d, 0)) for d in domain_order]

    n_models = len(MODEL_ORDER)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 4.2 * n_rows),
        subplot_kw={"projection": "polar"},
    )
    axes = np.atleast_2d(axes)

    # Pre-compute global y-max so all subplots share scale.
    y_max = float(np.nanmax(summary["mean"].values)) if not summary.empty else 1.0
    y_top = min(1.0, max(0.55, y_max + 0.05))
    y_ticks = [0.2, 0.4, 0.6, 0.8] if y_top > 0.7 else [0.1, 0.2, 0.3, 0.4, 0.5]
    y_ticks = [t for t in y_ticks if t < y_top]

    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx // n_cols, idx % n_cols]
        ax.set_theta_offset(np.pi / 2)  # start at the top
        ax.set_theta_direction(-1)  # clockwise

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axis_labels, fontsize=8)
        ax.set_ylim(0.0, y_top)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{t:.1f}" for t in y_ticks], fontsize=7)
        ax.tick_params(axis="x", pad=4)
        ax.grid(linewidth=0.5, alpha=0.4)
        ax.spines["polar"].set_linewidth(1.0)

        pretty_model = MODEL_PRETTY_NAMES.get(model, model)
        ax.set_title(pretty_model, fontsize=11, pad=18)

        model_data = summary[summary["model"] == model]
        for family_key, label, color in VARIANT_FAMILIES:
            sub = model_data[model_data["variant"] == family_key]
            if sub.empty:
                continue
            sub_idx = sub.set_index("domain")["mean"]
            values = [sub_idx.get(d, np.nan) for d in domain_order]
            # Skip variant entirely if no usable points.
            if all(np.isnan(v) for v in values):
                continue
            # Replace NaN with 0 so the polygon still closes; this only happens
            # when a (model, variant, domain) cell has zero seeds — log it.
            for d, v in zip(domain_order, values):
                if np.isnan(v):
                    console.print(f"[yellow]Missing: {model} / {family_key} / {d}[/yellow]")
            values = [0.0 if np.isnan(v) else v for v in values]
            closed = [*values, values[0]]
            ax.plot(angles, closed, color=color, linewidth=2.0, label=label)
            ax.fill(angles, closed, color=color, alpha=0.12)

    # Hide unused subplot cells.
    for idx in range(n_models, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    # Legend: deduplicate handles across subplots, attach to the first subplot
    # so ``save_figure`` (which iterates axes-level legends) picks it up and
    # splits it out into a sibling _legend.svg file.
    handles_seen: dict[str, tuple] = {}
    for ax_row in axes:
        for ax in ax_row:
            for handle, lab in zip(*ax.get_legend_handles_labels()):
                if lab not in handles_seen:
                    handles_seen[lab] = handle
    if handles_seen:
        axes[0, 0].legend(
            handles_seen.values(),
            handles_seen.keys(),
            loc="upper right",
            ncol=len(handles_seen),
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(1.0, 1.15),
        )

    fig.tight_layout()
    save_figure(fig, out_path, keep_title=True)
    console.print(f"[green]Saved:[/green] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Radar plot of LongBench-v2 per-domain accuracy across decoding variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_downstream_results_root(),
        help="Root directory holding downstream eval results.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="longbench_v2_short",
        help="Task subdirectory under results-root.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/longbench_v2_radar.svg"),
        help="Output SVG path. A sibling .csv with per-(model,variant,domain) "
        "mean/std/min/max/n_seeds will be written next to it.",
    )
    args = parser.parse_args()

    discovered = discover(args.results_root, args.task)
    if not discovered:
        console.print("[red]No results discovered. Aborting.[/red]")
        return

    long_df = aggregate_long_form(discovered)
    summary = aggregate_summary(long_df)

    # Domain order: by total sample count (across one greedy run, since all
    # variants score the same examples). Use the first available greedy run.
    counts: dict[str, int] = {}
    for model_dir in sorted((args.results_root / args.task).iterdir()):
        if not model_dir.is_dir():
            continue
        for variant_dir in sorted(model_dir.iterdir()):
            if variant_dir.name.startswith("greedy_s"):
                rows = load_results_jsonl(variant_dir)
                if rows:
                    counts = domain_sample_counts(rows)
                    break
        if counts:
            break
    assert counts, "Failed to read sample counts from any greedy run"
    domain_order = sorted(counts.keys(), key=lambda d: -counts[d])

    # Print a short summary table.
    console.rule("[bold]Per-(model, variant, domain) summary[/bold]")
    pivot = summary.pivot_table(
        index=["model", "variant"],
        columns="domain",
        values="mean",
        aggfunc="first",
    )
    pivot = pivot[domain_order]
    table = Table()
    table.add_column("model")
    table.add_column("variant")
    for d in domain_order:
        table.add_column(f"{d}\nn={counts[d]}", justify="right")
    for (model, variant), row in pivot.iterrows():
        table.add_row(
            model,
            variant,
            *[f"{row[d]:.3f}" if pd.notna(row[d]) else "—" for d in domain_order],
        )
    console.print(table)

    # Save CSV alongside the SVG.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    summary_out = summary[summary["domain"].isin(domain_order)].copy()
    summary_out["domain"] = pd.Categorical(summary_out["domain"], categories=domain_order, ordered=True)
    summary_out = summary_out.sort_values(["model", "variant", "domain"])
    summary_out.to_csv(csv_path, index=False)
    console.print(f"[green]Saved CSV:[/green] {csv_path}")

    make_radar_grid(summary, domain_order, counts, args.out)


if __name__ == "__main__":
    main()
