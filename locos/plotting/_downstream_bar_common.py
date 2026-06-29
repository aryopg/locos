"""Shared discovery, aggregation, and bar-plot helpers for per-model
downstream-eval bar charts (babilong / musique / medrag).

Each task family has its own driver script that supplies:
- ``task_dirs``: which subdirectories under ``<results-root>/`` to scan
- ``domain_fn``: maps a result row to its grouping label (the bar group)
- ``metrics``: which score keys to plot (one panel per metric)

The shared code mirrors ``longbench_v2_bar.py``: per-seed discovery via
``parse_variant_dir``/``family_to_canonical``/``load_results_jsonl``, then
mean/std aggregation and bar plotting hued by decoding variant.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console

from locos_eval.utils.plotting import (
    LINE_WIDTH,
    MODEL_PRETTY_NAMES,
    facecolor_alpha,
    save_figure,
    setup_plot_style,
)
from locos.plotting.longbench_v2_radar import (
    VARIANT_FAMILIES,
    family_to_canonical,
    load_results_jsonl,
    parse_variant_dir,
)

console = Console()

DomainFn = Callable[[dict, str], str | None]
DomainOrderFn = Callable[[dict[str, int]], list[str]]


# ---------------------------------------------------------------------------
# Discovery & aggregation
# ---------------------------------------------------------------------------


def discover_long_form(
    results_root: Path,
    task_dirs: Iterable[str],
    domain_fn: DomainFn,
    metrics: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Walk results tree across ``task_dirs`` and return a long-form DataFrame.

    Columns: ``model``, ``variant``, ``seed``, ``domain``, ``metric``,
    ``value`` — one row per (model, variant, seed, domain, metric) computed
    as the mean of ``score[metric]`` over rows in that group.

    Also returns aggregate sample counts per domain (across all task dirs,
    one variant — the first scored greedy variant per model).
    """
    records: list[dict] = []
    counts: dict[str, int] = defaultdict(int)

    for task_dir_name in task_dirs:
        counts_locked = False  # only count one (model, variant) per task dir
        task_dir = results_root / task_dir_name
        if not task_dir.is_dir():
            console.print(f"[yellow]Task dir not found: {task_dir}[/yellow]")
            continue
        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            short = model_dir.name.split("_", 1)[-1]
            for variant_dir in sorted(model_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue
                parsed = parse_variant_dir(variant_dir.name)
                if parsed is None:
                    continue
                family, seed = parsed
                canonical = family_to_canonical(family)
                if canonical is None:
                    continue
                rows = load_results_jsonl(variant_dir)
                if not rows:
                    continue

                # Group rows by (domain, metric) and average.
                buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
                for r in rows:
                    domain = domain_fn(r, task_dir_name)
                    if domain is None:
                        continue
                    scores = r.get("scores", {}) or {}
                    for m in metrics:
                        if m in scores and scores[m] is not None:
                            buckets[(domain, m)].append(float(scores[m]))
                for (domain, metric), vals in buckets.items():
                    if not vals:
                        continue
                    records.append(
                        {
                            "model": short,
                            "variant": canonical,
                            "seed": seed,
                            "domain": domain,
                            "metric": metric,
                            "value": float(np.mean(vals)),
                        }
                    )

                # Sample counts: lock to the first greedy run per task dir.
                if not counts_locked and canonical == "greedy":
                    for r in rows:
                        d = domain_fn(r, task_dir_name)
                        if d is not None:
                            counts[d] += 1

    return pd.DataFrame.from_records(records), dict(counts)


def overall_per_seed(
    results_root: Path,
    task_dirs: Iterable[str],
    domain_fn: DomainFn,
    metrics: list[str],
    excluded_domains: set[str] | None = None,
) -> pd.DataFrame:
    """Per-seed *macro*-averaged overall scores.

    For each (model, variant, seed): take the per-domain mean of each
    metric, then average those domain-means. This weights every domain
    equally regardless of sample count — the intent is that ``Overall``
    should reflect a balanced view across sub-tasks (e.g. babilong qa2
    vs qa3, musique 2-hop vs 4-hop), not be dominated by whichever
    sub-task happens to have more examples. The synthetic ``"Overall"``
    domain is plotted alongside the real per-domain bars.
    """
    excluded = excluded_domains or set()
    # (model, variant, seed) -> {metric: {domain: [scores]}}
    bucket: dict[tuple[str, str, int], dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for task_dir_name in task_dirs:
        task_dir = results_root / task_dir_name
        if not task_dir.is_dir():
            continue
        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            short = model_dir.name.split("_", 1)[-1]
            for variant_dir in sorted(model_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue
                parsed = parse_variant_dir(variant_dir.name)
                if parsed is None:
                    continue
                family, seed = parsed
                canonical = family_to_canonical(family)
                if canonical is None:
                    continue
                rows = load_results_jsonl(variant_dir)
                if not rows:
                    continue
                key = (short, canonical, seed)
                for r in rows:
                    d = domain_fn(r, task_dir_name)
                    if d is None or d in excluded:
                        continue
                    scores = r.get("scores", {}) or {}
                    for m in metrics:
                        if m in scores and scores[m] is not None:
                            bucket[key][m][d].append(float(scores[m]))

    records = []
    for (model, variant, seed), by_metric in bucket.items():
        for metric, by_domain in by_metric.items():
            domain_means = [float(np.mean(vals)) for vals in by_domain.values() if vals]
            if not domain_means:
                continue
            records.append(
                {
                    "model": model,
                    "variant": variant,
                    "seed": seed,
                    "domain": "Overall",
                    "metric": metric,
                    "value": float(np.mean(domain_means)),
                }
            )
    return pd.DataFrame.from_records(records)


def aggregate(long_df: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / n_seeds per (model, variant, domain, metric)."""
    agg = (
        long_df.groupby(["model", "variant", "domain", "metric"])["value"]
        .agg(["mean", "std", "min", "max", "count"])
        .rename(columns={"count": "n_seeds"})
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _compute_ytop(values: pd.Series) -> float:
    """Round mean+std up to the next 0.05 step, capped at 1.0."""
    if values.empty:
        return 1.0
    upper = float(values.max())
    raw = upper + 0.05
    return min(1.0, max(0.3, np.ceil(raw / 0.05) * 0.05))


def _yticks(y_top: float) -> np.ndarray:
    spacing = 0.1 if y_top <= 0.6 else 0.2
    return np.arange(0.0, y_top + 1e-9, spacing)


def _plot_bars(
    ax,
    summary_slice: pd.DataFrame,
    group_keys: list[str],
    group_labels: list[str],
    *,
    group_field: str,
    y_label: str,
    chance_line: float | None = None,
) -> None:
    """Render a grouped bar chart on ``ax``.

    ``summary_slice`` has columns (``group_field``, ``variant``, ``mean``,
    ``std``, ``n_seeds``); bars are grouped on ``group_keys`` (x-axis) and
    hued by variant (in the order defined by ``VARIANT_FAMILIES``).
    """
    n_groups = len(group_keys)
    variants = list(VARIANT_FAMILIES)
    n_variants = len(variants)
    bar_width = 0.8 / n_variants
    x_centers = np.arange(n_groups)

    for v_idx, (key, label, color) in enumerate(variants):
        sub = summary_slice[summary_slice["variant"] == key].set_index(group_field)
        means = np.array(
            [sub.loc[g, "mean"] if g in sub.index else np.nan for g in group_keys],
            dtype=float,
        )
        stds = np.array(
            [sub.loc[g, "std"] if g in sub.index else 0.0 for g in group_keys],
            dtype=float,
        )
        n_seeds = np.array(
            [int(sub.loc[g, "n_seeds"]) if g in sub.index else 0 for g in group_keys],
            dtype=int,
        )
        means_plot = np.where(n_seeds > 0, means, 0.0)
        stds_plot = np.where(n_seeds > 1, stds, 0.0)

        offsets = (v_idx - (n_variants - 1) / 2) * bar_width
        bars = ax.bar(
            x_centers + offsets,
            means_plot,
            width=bar_width,
            color=facecolor_alpha(color, 0.85),
            edgecolor="black",
            linewidth=LINE_WIDTH * 0.5,
            label=label,
            zorder=2,
        )
        for bar, has_data in zip(bars, n_seeds > 0):
            if not has_data:
                bar.set_visible(False)
        if np.any(stds_plot > 0):
            ax.errorbar(
                x_centers + offsets,
                means_plot,
                yerr=stds_plot,
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=2.5,
                zorder=3,
            )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel(y_label)
    if not summary_slice.empty:
        y_top = _compute_ytop(summary_slice["mean"].fillna(0.0) + summary_slice["std"].fillna(0.0))
    else:
        y_top = 1.0
    ax.set_ylim(0.0, y_top)
    ax.set_yticks(_yticks(y_top))
    if chance_line is not None and chance_line < y_top:
        ax.axhline(chance_line, color="black", linestyle=":", linewidth=1.0, alpha=0.4, zorder=1)
    ax.grid(False)
    ax.set_axisbelow(True)


# Fixed figure sizes so Overall.svg matches across babilong/musique/medrag and
# per-model figures match across models within a task.
OVERALL_FIGSIZE = (10.0, 2.0)
PER_MODEL_FIGSIZE = (8.0, 3.0)

# Reduced model set: largest checkpoint of each family. Used for the
# ``_emodels`` Overall variants — narrower figure, no x-tick rotation.
THREE_MODEL_ORDER = ["Qwen3-32B", "gemma-3-27b-it", "Olmo-3.1-32B-Instruct"]
THREE_MODEL_FIGSIZE = (6.0, 2.5)


def make_per_model_figure(
    model: str,
    summary: pd.DataFrame,
    domain_order: list[str],
    domain_counts: dict[str, int],
    metric_key: str,
    metric_label: str,
    out_path: Path,
    *,
    domain_short: dict[str, str] | None = None,
    chance_line: float | None = None,
) -> None:
    """Per-model figure for a single metric: x = domains (incl. Overall),
    hue = decoding variant. Error bars = ±1 std across seeds."""
    setup_plot_style()
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"

    model_data = summary[(summary["model"] == model) & (summary["metric"] == metric_key)]
    if model_data.empty:
        console.print(f"[yellow]No {metric_key} data for {model}, skipping[/yellow]")
        return

    domain_short = domain_short or {}
    overall_n = sum(domain_counts.values())
    xtick_labels = []
    for d in domain_order:
        n = overall_n if d == "Overall" else domain_counts.get(d, 0)
        short = "Overall" if d == "Overall" else domain_short.get(d, d)
        xtick_labels.append(f"{short}\n(n={n})")

    fig, ax = plt.subplots(figsize=PER_MODEL_FIGSIZE)
    _plot_bars(
        ax,
        model_data,
        group_keys=domain_order,
        group_labels=xtick_labels,
        group_field="domain",
        y_label=metric_label,
        chance_line=chance_line,
    )

    pretty = MODEL_PRETTY_NAMES.get(model, model)
    fig.suptitle(pretty, fontsize=12)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", ncol=len(handles), fontsize=9, frameon=False)

    fig.tight_layout()
    save_figure(fig, out_path, keep_title=True)
    console.print(f"[green]Saved:[/green] {out_path}")


def make_overall_figure(
    summary: pd.DataFrame,
    model_order: list[str],
    metric_key: str,
    metric_label: str,
    total_samples: int,
    out_path: Path,
    *,
    chance_line: float | None = None,
    figsize: tuple[float, float] = OVERALL_FIGSIZE,
    rotate_xticks: bool = False,
    task_name: str | None = None,
) -> None:
    """Cross-model figure for a single metric: x = models, hue = variant.

    Uses a fixed figure size (default ``OVERALL_FIGSIZE``) so Overall.svg
    renders at the same dimensions across tasks — letting them be placed
    side-by-side in LaTeX without per-figure scaling. The ``_3models``
    variants pass a smaller ``figsize`` and ``rotate_xticks=False``.

    If ``task_name`` is provided, it is rendered as a suptitle (preserved
    via ``keep_title=True`` so ``save_figure`` doesn't strip it).
    """
    setup_plot_style()
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"

    overall = summary[(summary["domain"] == "Overall") & (summary["metric"] == metric_key)]
    present_models = [m for m in model_order if m in overall["model"].unique()]
    if not present_models:
        console.print(f"[yellow]No overall {metric_key} data — skipping[/yellow]")
        return

    pretty_labels = [MODEL_PRETTY_NAMES.get(m, m) for m in present_models]
    fig, ax = plt.subplots(figsize=figsize)
    _plot_bars(
        ax,
        overall,
        group_keys=present_models,
        group_labels=pretty_labels,
        group_field="model",
        y_label=metric_label,
        chance_line=chance_line,
    )
    if rotate_xticks:
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")
    # ax.set_xlabel(f"Model (n={total_samples})")

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", ncol=len(handles), fontsize=9, frameon=False)

    if task_name:
        # Title on the axes (not a suptitle) so it centers over the plot
        # area, not the whole figure — and ``pad=2`` keeps margin minimal.
        ax.set_title(task_name, fontsize=12, pad=2)

    fig.tight_layout(pad=0.2)
    save_figure(fig, out_path, keep_title=bool(task_name))
    console.print(f"[green]Saved:[/green] {out_path}")


def render_multi_metric_domain_bars(
    *,
    results_root: Path,
    out_dir: Path,
    task_dirs: Iterable[str],
    domain_fn: DomainFn,
    metrics: list[tuple[str, str]],
    task_name: str,
    model_order: list[str],
    domain_short: dict[str, str] | None = None,
    domain_order_fn: DomainOrderFn | None = None,
) -> None:
    """Render the standard downstream bar-chart family.

    Produces:
    - ``summary.csv``
    - one per-model SVG for each metric
    - ``Overall.svg`` and ``Overall_3models.svg`` for each metric

    Task-specific scripts supply only task directories, metric labels, domain
    extraction, and optional domain ordering/labels.
    """
    metric_keys = [m for m, _ in metrics]
    long_df, counts = discover_long_form(results_root, task_dirs, domain_fn, metric_keys)
    overall_df = overall_per_seed(results_root, task_dirs, domain_fn, metric_keys)
    if long_df.empty:
        console.print("[red]No data discovered. Aborting.[/red]")
        return

    full_df = long_df if overall_df.empty else pd.concat([long_df, overall_df], ignore_index=True)
    summary = aggregate(full_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    console.print(f"[green]Saved CSV:[/green] {out_dir / 'summary.csv'}")

    if domain_order_fn is None:
        domain_order = sorted(counts, key=lambda d: -counts[d])
    else:
        domain_order = domain_order_fn(counts)

    primary_key = metrics[0][0]
    for model in model_order:
        if model not in summary["model"].unique():
            console.print(f"[yellow]No data for {model} — skipping[/yellow]")
            continue
        pretty = MODEL_PRETTY_NAMES.get(model, model)
        for metric_key, metric_label in metrics:
            suffix = "" if metric_key == primary_key else f"_{metric_key}"
            make_per_model_figure(
                model,
                summary,
                domain_order,
                counts,
                metric_key,
                metric_label,
                out_dir / f"{pretty}{suffix}.svg",
                domain_short=domain_short,
            )

    total_samples = sum(counts.values())
    for metric_key, metric_label in metrics:
        suffix = "" if metric_key == primary_key else f"_{metric_key}"
        make_overall_figure(
            summary,
            model_order,
            metric_key,
            metric_label,
            total_samples,
            out_dir / f"Overall{suffix}.svg",
            task_name=task_name,
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
            task_name=task_name,
        )
