#!/usr/bin/env python3
"""E3 — Off-needle contribution provenance.

Hypothesis: Off-needle answer-aligned contribution (Φ⁻) originates from
identifiable position classes; spatial contrast doesn't subtract legitimate
needle-derived signal.

REQUIRES V1: per-position φ arrays stored per (model, trial, answer-step,
head, position). Exits with a clear error if these are absent.

Outputs:
    analysis/outputs/e3/e3_provenance.svg
    analysis/outputs/e3/e3_provenance_legend.svg
    analysis/outputs/e3/e3_provenance.csv

Usage:
    python locos/analysis/e3_provenance.py
    python locos/analysis/e3_provenance.py --no-download
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console

from locos_eval.utils.plotting import save_figure, setup_plot_style
from locos.analysis._utils import (
    ALL_MODELS,
    MODEL_LABELS,
    get_output_dir,
    load_score_file,
    mean_scores,
    top_k_heads,
)

console = Console()

# Position taxonomy priority (highest wins)
CATEGORIES = ["question", "answer_mention", "sink", "needle_adjacent", "bulk"]
CATEGORY_COLORS = {
    "question": "#4C72B0",
    "answer_mention": "#DD8452",
    "sink": "#55A868",
    "needle_adjacent": "#C44E52",
    "bulk": "#8172B2",
}

K = 50  # head set size


def _load_arrays(model: str, download: bool) -> Path | None:
    """Return path to per-position arrays Parquet for a model, or None."""
    short = model.split("/")[-1]
    local = _REPO_ROOT / "retrieval_heads" / f"{short}_logit_contrib_nolima.arrays.parquet"
    if local.exists():
        return local
    if download:
        try:
            from huggingface_hub import hf_hub_download

            from locos.analysis._utils import HF_RESULTS_REPO

            hf_p = f"retrieval_heads/{short}_logit_contrib_nolima.arrays.parquet"
            cached = hf_hub_download(repo_id=HF_RESULTS_REPO, filename=hf_p, repo_type="dataset")
            import shutil

            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, local)
            return local
        except Exception:
            pass
    return None


def _classify_position(
    pos: int,
    needle_start: int,
    needle_end: int,
    question_start: int,
    question_end: int,
    answer_mention_positions: set[int],
    sink_positions: set[int],
) -> str:
    if question_start <= pos < question_end:
        return "question"
    if pos in answer_mention_positions:
        return "answer_mention"
    if pos in sink_positions:
        return "sink"
    if needle_start - 25 <= pos < needle_start or needle_end <= pos < needle_end + 25:
        return "needle_adjacent"
    return "bulk"


def _aggregate_provenance_from_parquet(
    arr_path: Path,
    locos_top50: set[tuple[int, int]],
    wu_top50: set[tuple[int, int]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Read per-position arrays and aggregate Φ⁻ by category for two head groups.

    Returns (locos_provenance, wu_provenance) where each is {category: total_phi_minus}.
    The arrays Parquet schema is expected to contain columns:
        trial_id, step, layer, head, position, phi, alpha, token_id,
        in_needle, in_question, is_sink, needle_start, needle_end,
        question_start, question_end, answer_token_ids (list).

    If schema differs, this raises AssertionError with a clear message.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("pyarrow required for E3. Install with: pip install pyarrow") from e

    table = pq.read_table(arr_path)
    required_cols = {
        "trial_id",
        "layer",
        "head",
        "position",
        "phi",
        "in_needle",
        "needle_start",
        "needle_end",
    }
    actual_cols = set(table.schema.names)
    missing = required_cols - actual_cols
    assert not missing, (
        f"Per-position array file is missing columns: {missing}. " f"Re-run detection with updated --save-arrays flag."
    )

    df = table.to_pydict()
    n_rows = len(df["phi"])

    locos_prov: dict[str, float] = {c: 0.0 for c in CATEGORIES}
    wu_prov: dict[str, float] = {c: 0.0 for c in CATEGORIES}

    for i in range(n_rows):
        layer = int(df["layer"][i])
        head = int(df["head"][i])
        phi = float(df["phi"][i])
        in_needle = bool(df["in_needle"][i])

        if in_needle:
            continue  # only off-needle (Φ⁻) positions

        pos = int(df["position"][i])
        needle_start = int(df["needle_start"][i])
        needle_end = int(df["needle_end"][i])
        q_start = int(df.get("question_start", [0] * n_rows)[i])
        q_end = int(df.get("question_end", [0] * n_rows)[i])
        answer_mentions = set(df.get("answer_mention_positions", [[]] * n_rows)[i] or [])
        sink_set = set(df.get("sink_positions", [[]] * n_rows)[i] or [])

        cat = _classify_position(pos, needle_start, needle_end, q_start, q_end, answer_mentions, sink_set)

        head_tuple = (layer, head)
        if head_tuple in locos_top50:
            locos_prov[cat] += abs(phi)
        if head_tuple in wu_top50:
            wu_prov[cat] += abs(phi)

    # Normalize to fractions
    def _normalize(d: dict[str, float]) -> dict[str, float]:
        total = sum(d.values())
        return {c: v / total if total > 0 else 0.0 for c, v in d.items()}

    return _normalize(locos_prov), _normalize(wu_prov)


def run(download: bool = True) -> None:
    setup_plot_style()
    out_dir = get_output_dir("e3")

    # Check V1
    any_arrays = any(_load_arrays(m, download=False) is not None for m in ALL_MODELS)
    if not any_arrays and not download:
        console.print(
            "[bold red]V1 FAIL:[/bold red] No per-position array files found locally.\n"
            "Run with HF download enabled or re-run logit_contrib.py with --save-arrays."
        )
        sys.exit(1)

    models_ok = []
    model_results: list[tuple[str, dict, dict]] = []

    for model in ALL_MODELS:
        arr_path = _load_arrays(model, download=download)
        if arr_path is None:
            console.print(f"[yellow]SKIP {MODEL_LABELS[model]}: arrays not found[/yellow]")
            continue

        locos_sf = load_score_file(model, "locos", "nolima", download=download)
        wu_sf = load_score_file(model, "wu", "nolima", download=download)
        if locos_sf is None or wu_sf is None:
            console.print(f"[yellow]SKIP {MODEL_LABELS[model]}: score files missing[/yellow]")
            continue

        locos_top50 = top_k_heads(mean_scores(locos_sf), K)
        wu_top50 = top_k_heads(mean_scores(wu_sf), K)

        try:
            locos_prov, wu_prov = _aggregate_provenance_from_parquet(arr_path, locos_top50, wu_top50)
        except (AssertionError, Exception) as e:
            console.print(f"[red]ERROR {MODEL_LABELS[model]}: {e}[/red]")
            sys.exit(1)

        models_ok.append(model)
        model_results.append((model, locos_prov, wu_prov))

    assert models_ok, (
        "[bold red]V1 FAIL:[/bold red] No models with per-position arrays. "
        "Re-run logit_contrib.py with --save-arrays and retry."
    )

    import matplotlib.pyplot as plt

    n_models = len(models_ok)
    ncols = n_models
    fig, axes = plt.subplots(1, ncols * 2, figsize=(3.0 * ncols * 2, 4.0))
    if ncols * 2 == 1:
        axes = [axes]

    csv_rows: list[dict] = []
    col = 0
    for model, locos_prov, wu_prov in model_results:
        label = MODEL_LABELS[model]
        for group_label, prov in [("LOCOS", locos_prov), ("Wu/NoLiMa", wu_prov)]:
            ax = axes[col]
            col += 1
            bottom = 0.0
            for cat in CATEGORIES:
                val = prov.get(cat, 0.0)
                ax.bar(
                    group_label,
                    val,
                    bottom=bottom,
                    color=CATEGORY_COLORS[cat],
                    label=cat,
                    edgecolor="black",
                    linewidth=0.8,
                )
                bottom += val
                csv_rows.append(
                    {
                        "model": label,
                        "head_group": group_label,
                        "category": cat,
                        "fraction": f"{val:.4f}",
                    }
                )
            ax.set_ylim(0, 1)
            ax.set_ylabel("Fraction of |Φ⁻|" if col <= 2 else "")
            title_str = f"{label}\n{group_label}"
            ax.set_title(title_str, fontsize=8)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=CATEGORY_COLORS[c]) for c in CATEGORIES]
    axes[0].legend(handles, CATEGORIES, fontsize=7, loc="upper right")

    fig.tight_layout()
    save_figure(fig, out_dir / "e3_provenance.svg")

    csv_path = out_dir / "e3_provenance.csv"
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)

    # Decision rule check
    for model, locos_prov, _ in model_results:
        label = MODEL_LABELS[model]
        high_frac = locos_prov.get("question", 0) + locos_prov.get("answer_mention", 0)
        if high_frac > 0.5:
            console.print(
                f"[yellow]{label}: question+answer_mention = {high_frac:.1%} > 50%.[/yellow]\n"
                f"  → Add robustness variant excluding question-span positions and report Kendall τ."
            )
        else:
            console.print(f"[green]{label}: question+answer_mention = {high_frac:.1%} (< 50%). OK.[/green]")

    console.print(f"\n[dim]Saved:[/dim] {out_dir}/e3_provenance.svg, e3_provenance.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E3 — Off-needle contribution provenance (requires V1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    run(download=not args.no_download)


if __name__ == "__main__":
    main()
