#!/usr/bin/env python3
"""KV-group-granularity reporting for GQA-model retrieval-head rankings.

For GQA models, Q-heads sharing a KV group have identical value vectors, so
their per-head scores (e.g. logit-contribution) are correlated by construction.
A top-k Q-head list can therefore look diverse while covering only a handful
of unique KV groups. This script re-aggregates an existing detection JSON at
KV-group granularity alongside Q-head granularity so the paper can report
both.

Outputs:
- ``kv_group_stats.csv`` — tidy long-form (model, method, k, unique_groups,
  concentration, coverage) suitable for plotting.
- ``{stem}_kvgroup.json`` — envelope JSON keyed by ``"layer-kvgroup"`` with
  group-averaged scores, consumable by the same downstream tooling that
  reads head-level detection JSONs.

Usage:
    python -m locos.analysis.kv_group_analysis \\
        --scores path/to/logit_contrib_nolima.json \\
        --model Qwen/Qwen3-8B \\
        --method logit_contrib \\
        --output-dir analysis_out/kv_group/Qwen3-8B
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

console = Console()


def load_scores(path: str | Path) -> dict[str, list[float]]:
    """Load detector JSON, envelope-aware (mirrors compare_heads.load_scores)."""
    with open(path) as f:
        data = json.load(f)
    if "scores" in data and isinstance(data["scores"], dict):
        return data["scores"]
    return data


def read_gqa_config(model_name: str) -> tuple[int, int]:
    """Return ``(num_attention_heads, num_key_value_heads)`` from HF config.

    CPU-only; does not load weights. Uses the same fallback chain as
    ``logit_contrib.extract_head_config`` so GQA ratio detection stays
    consistent across the codebase.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)

    num_heads = getattr(text_config, "num_attention_heads", None) or getattr(config, "num_attention_heads", None)
    if num_heads is None:
        raise RuntimeError(f"Could not find num_attention_heads for {model_name}")

    num_kv_heads = (
        getattr(text_config, "num_key_value_heads", None) or getattr(config, "num_key_value_heads", None) or num_heads
    )
    return int(num_heads), int(num_kv_heads)


def rank_heads_by_mean(scores: dict[str, list[float]]) -> list[tuple[int, int, float]]:
    """Return ``(layer, head, mean_score)`` sorted by mean descending."""
    ranked: list[tuple[int, int, float]] = []
    for key, vals in scores.items():
        layer, head = map(int, key.split("-"))
        ranked.append((layer, head, float(np.mean(vals)) if vals else 0.0))
    ranked.sort(key=lambda t: t[2], reverse=True)
    return ranked


def head_to_kv_group(head_idx: int, gqa_ratio: int) -> int:
    """Map a Q-head index to its KV-group index under HF's GQA layout.

    HF convention (used by Llama/Qwen/Gemma/Olmo): Q-heads are laid out
    contiguously per KV group, so Q-head ``h`` belongs to KV group
    ``h // gqa_ratio``.
    """
    return head_idx // gqa_ratio


def count_unique_kv_groups(top_k_heads: list[tuple[int, int]], gqa_ratio: int) -> int:
    """Count distinct (layer, kv_group) pairs covered by the top-k Q-heads."""
    return len({(layer, head_to_kv_group(h, gqa_ratio)) for layer, h in top_k_heads})


def aggregate_scores_by_kv_group(
    scores: dict[str, list[float]],
    gqa_ratio: int,
) -> dict[str, list[float]]:
    """Average per-trial Q-head scores within each KV group.

    The output is an envelope-compatible ``{"layer-kvgroup": [per-trial mean]}``
    dict. Trials are aligned by position (all Q-heads in a group are assumed
    to share the trial ordering used during detection).
    """
    # Group Q-head scores by (layer, kv_group)
    groups: dict[tuple[int, int], list[list[float]]] = {}
    for key, vals in scores.items():
        layer, head = map(int, key.split("-"))
        g = head_to_kv_group(head, gqa_ratio)
        groups.setdefault((layer, g), []).append(list(vals))

    out: dict[str, list[float]] = {}
    for (layer, g), member_vals in groups.items():
        # NOTE: averaging per-trial across Q-heads in a KV group is the
        # simplest aggregation, but an alternative is max-over-heads (interpret
        # a group as "retrieval-relevant if any member head is"). The reviewer
        # may prefer max. Chose mean because it matches how per-head means are
        # used downstream in ranking.
        if not member_vals:
            out[f"{layer}-{g}"] = []
            continue
        # Q-heads within a KV group share the same per-trial upstream evaluation,
        # so their score-list lengths must match. If they ever diverge that's
        # a real upstream bug worth surfacing, not silently averaging away.
        trial_lens = {len(v) for v in member_vals}
        assert len(trial_lens) == 1, f"Inconsistent trial counts in layer {layer} group {g}: {sorted(trial_lens)}"
        arr = np.asarray(member_vals, dtype=float)  # (num_members, num_trials)
        out[f"{layer}-{g}"] = arr.mean(axis=0).tolist()
    return out


def sweep_k_values(num_heads_total: int) -> list[int]:
    """Log-ish sweep of k from 1 up to roughly num_heads_total."""
    points = [1, 2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
    return [k for k in points if k <= num_heads_total] or [num_heads_total]


def compute_kv_group_stats(
    scores: dict[str, list[float]],
    *,
    num_heads: int,
    num_kv_heads: int,
    model: str,
    method: str,
    k_values: list[int] | None = None,
) -> list[dict]:
    """Return per-k rows: unique_groups, concentration, coverage."""
    assert (
        num_heads % num_kv_heads == 0
    ), f"GQA layout requires num_heads % num_kv_heads == 0, got {num_heads}/{num_kv_heads}"
    gqa_ratio = num_heads // num_kv_heads
    ranked = rank_heads_by_mean(scores)
    num_heads_total = len(ranked)

    # Total unique (layer, kv_group) cells possible across the model.
    # NOTE: we treat each (layer, kv_group) as a separate cell. That
    # matches how heads are ranked (per-layer). If the reviewer wants a global
    # "how many distinct KV groups across all layers" number, that's a second
    # column we could add.
    layers_seen = {layer for layer, _, _ in ranked}
    max_possible_groups = len(layers_seen) * num_kv_heads

    if k_values is None:
        k_values = sweep_k_values(num_heads_total)

    rows: list[dict] = []
    for k in k_values:
        top_k = [(layer, head) for layer, head, _ in ranked[:k]]
        unique_groups = count_unique_kv_groups(top_k, gqa_ratio)
        # Fraction of the maximum *achievable* unique groups given k and
        # the total group pool. At k ≤ max_possible_groups the ceiling is k;
        # above that ceiling saturates at max_possible_groups.
        ceiling = min(k, max_possible_groups)
        coverage = unique_groups / ceiling if ceiling else 0.0
        concentration = k / unique_groups if unique_groups else float("nan")
        rows.append(
            {
                "model": model,
                "method": method,
                "k": k,
                "unique_groups": unique_groups,
                "concentration": concentration,
                "coverage": coverage,
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "gqa_ratio": gqa_ratio,
                "num_layers": len(layers_seen),
            }
        )
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_envelope_json(grouped_scores: dict[str, list[float]], meta: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "scores": grouped_scores}, f, indent=2)


def render_summary_table(rows: list[dict]) -> Table:
    table = Table(title="KV-group coverage")
    table.add_column("k", justify="right")
    table.add_column("unique groups", justify="right")
    table.add_column("coverage", justify="right")
    table.add_column("concentration (k / groups)", justify="right")
    for r in rows:
        table.add_row(
            str(r["k"]),
            str(r["unique_groups"]),
            f"{r['coverage']:.3f}",
            f"{r['concentration']:.2f}",
        )
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="Path to detector JSON (flat or envelope)")
    parser.add_argument("--model", required=True, help="HF model name (e.g. Qwen/Qwen3-8B)")
    parser.add_argument("--method", required=True, help="Tag identifying the detection method for the CSV")
    parser.add_argument("--output-dir", required=True, help="Directory to write CSV + envelope JSON into")
    parser.add_argument(
        "--k",
        type=int,
        nargs="*",
        default=None,
        help="Explicit k values to report. Defaults to a log-spaced sweep up to the total head count.",
    )
    args = parser.parse_args()

    console.print(Panel.fit(f"[bold]KV-group analysis[/bold]\nmodel: {args.model}\nmethod: {args.method}"))

    scores = load_scores(args.scores)
    num_heads, num_kv_heads = read_gqa_config(args.model)
    console.print(f"GQA config: {num_heads} Q-heads / {num_kv_heads} KV-heads (ratio {num_heads // num_kv_heads}:1)")

    rows = compute_kv_group_stats(
        scores,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        model=args.model,
        method=args.method,
        k_values=args.k,
    )
    console.print(render_summary_table(rows))

    out_dir = Path(args.output_dir)
    stem = Path(args.scores).stem
    write_csv(rows, out_dir / f"{stem}__kv_group_stats.csv")

    grouped_scores = aggregate_scores_by_kv_group(scores, gqa_ratio=num_heads // num_kv_heads)
    meta = {
        "source_scores": str(args.scores),
        "model": args.model,
        "method": args.method,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "gqa_ratio": num_heads // num_kv_heads,
        "aggregation": "mean_per_trial_over_group_members",
    }
    write_envelope_json(grouped_scores, meta, out_dir / f"{stem}__kvgroup.json")

    console.print(f"[green]Wrote[/green] {out_dir}/{stem}__kv_group_stats.csv and {stem}__kvgroup.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
