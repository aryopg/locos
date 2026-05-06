#!/usr/bin/env python3
"""Rank-agreement analysis between CRI and logit-contribution rankings (H3).

The H3 rebuttal hinges on comparing a per-head causal activation-patching
metric (CRI) against the linear-probe logit-contribution metric. If the two
rankings agree across layers, direct-path bias is closed; if mid-layer heads
cluster in the "high CRI, low LC" quadrant, the concern is real.

This script:
- Loads one CRI JSON and one logit-contribution JSON (envelope format).
- Computes Spearman ρ, Kendall τ, and top-k overlap for k ∈ {10, 20, 50}.
- Tags each head with a layer-depth bucket (early/mid/late).
- Writes a tidy CSV ``direct_path_bias.csv`` with per-head
  (layer, head, cri_score, lc_score, cri_rank, lc_rank, bucket) rows plus
  a summary CSV of agreement metrics.

Usage:
    python -m locos.analysis.direct_path_bias \\
        --cri-json retrieval_heads/Qwen3-8B_cri_first_token_logit_diff.json \\
        --lc-json  retrieval_heads/Qwen3-8B_logit_contrib_nolima.json \\
        --model Qwen/Qwen3-8B \\
        --output-dir analysis_out/direct_path_bias/Qwen3-8B
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

console = Console()

TOP_K_VALUES = (10, 20, 50)


def load_scores(path: str | Path) -> dict[str, list[float]]:
    with open(path) as f:
        data = json.load(f)
    if "scores" in data and isinstance(data["scores"], dict):
        return data["scores"]
    return data


def mean_per_head(scores: dict[str, list[float]]) -> dict[str, float]:
    return {k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()}


def layer_bucket(layer: int, num_layers: int) -> str:
    """Assign ``early``/``mid``/``late`` to a layer index.

    Three equal-sized contiguous bins. ``mid`` is the bucket where the
    direct-path-bias concern is most salient.
    """
    edges = np.linspace(0, num_layers, num=4)
    if layer < edges[1]:
        return "early"
    if layer < edges[2]:
        return "mid"
    return "late"


def infer_num_layers(keys) -> int:
    return max(int(k.split("-")[0]) for k in keys) + 1


def top_k_overlap(cri_ranked: list[tuple[str, float]], lc_ranked: list[tuple[str, float]], k: int) -> int:
    return len({kk for kk, _ in cri_ranked[:k]} & {kk for kk, _ in lc_ranked[:k]})


def compute_rows(
    cri_means: dict[str, float],
    lc_means: dict[str, float],
    num_layers: int,
) -> list[dict]:
    common = sorted(set(cri_means) & set(lc_means))
    if not common:
        raise ValueError("CRI and LC JSONs share no (layer, head) keys")

    cri_ranked = sorted(common, key=lambda k: cri_means[k], reverse=True)
    lc_ranked = sorted(common, key=lambda k: lc_means[k], reverse=True)
    cri_rank = {k: i for i, k in enumerate(cri_ranked)}
    lc_rank = {k: i for i, k in enumerate(lc_ranked)}

    rows = []
    for key in common:
        layer, head = map(int, key.split("-"))
        rows.append(
            {
                "layer": layer,
                "head": head,
                "cri_score": cri_means[key],
                "lc_score": lc_means[key],
                "cri_rank": cri_rank[key],
                "lc_rank": lc_rank[key],
                "bucket": layer_bucket(layer, num_layers),
                "layer_depth": layer / max(num_layers - 1, 1),
            }
        )
    return rows


def compute_agreement(cri_means: dict[str, float], lc_means: dict[str, float], model: str) -> dict:
    common = sorted(set(cri_means) & set(lc_means))
    cri_vals = np.array([cri_means[k] for k in common])
    lc_vals = np.array([lc_means[k] for k in common])
    rho, _ = spearmanr(cri_vals, lc_vals)
    tau, _ = kendalltau(cri_vals, lc_vals)
    cri_ranked = sorted([(k, cri_means[k]) for k in common], key=lambda t: t[1], reverse=True)
    lc_ranked = sorted([(k, lc_means[k]) for k in common], key=lambda t: t[1], reverse=True)
    overlap = {f"overlap@{k}": top_k_overlap(cri_ranked, lc_ranked, k) for k in TOP_K_VALUES}
    return {
        "model": model,
        "spearman": float(rho),
        "kendall": float(tau),
        **overlap,
        "n_heads": len(common),
    }


def identify_disagreement(rows: list[dict]) -> list[dict]:
    """Return heads with high CRI rank (top 10%) and low LC rank (bottom 50%)."""
    n = len(rows)
    if n == 0:
        return []
    cri_top_cutoff = max(1, int(0.10 * n))  # cri_rank < cri_top_cutoff  → top 10%
    lc_bottom_cutoff = n - max(1, int(0.50 * n))  # lc_rank >= lc_bottom_cutoff → bottom 50%
    disagreement = [r for r in rows if r["cri_rank"] < cri_top_cutoff and r["lc_rank"] >= lc_bottom_cutoff]
    # FIXME(aryo): the "top-10% vs bottom-50%" thresholds are arbitrary. An
    # alternative is absolute-rank-difference > X (e.g. > num_heads/4). The
    # chosen percentile thresholds emphasise the clearest disagreements and
    # keep the table readable; a reviewer may prefer a different cut.
    return disagreement


def write_csvs(
    rows: list[dict],
    agreement: dict,
    disagreement: list[dict],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "direct_path_bias.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "agreement_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agreement.keys()))
        writer.writeheader()
        writer.writerow(agreement)

    if disagreement:
        with open(out_dir / "disagreement_quadrant.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(disagreement[0].keys()))
            writer.writeheader()
            writer.writerows(disagreement)


def render_summary_table(agreement: dict) -> Table:
    table = Table(title="Rank-agreement (CRI vs logit-contribution)")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in agreement.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cri-json", required=True)
    parser.add_argument("--lc-json", required=True)
    parser.add_argument("--model", required=True, help="HF model name (tag for the CSV)")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cri_scores = load_scores(args.cri_json)
    lc_scores = load_scores(args.lc_json)
    cri_means = mean_per_head(cri_scores)
    lc_means = mean_per_head(lc_scores)

    num_layers = infer_num_layers(set(cri_means) & set(lc_means))

    rows = compute_rows(cri_means, lc_means, num_layers)
    # Tag every row with model for cross-model concat downstream.
    for r in rows:
        r["model"] = args.model

    agreement = compute_agreement(cri_means, lc_means, model=args.model)
    console.print(render_summary_table(agreement))

    disagreement = identify_disagreement(rows)
    console.print(f"Disagreement quadrant (top-10% CRI, bottom-50% LC): {len(disagreement)} heads")

    write_csvs(rows, agreement, disagreement, Path(args.output_dir))
    console.print(f"[green]Wrote[/green] {args.output_dir}/direct_path_bias.csv and friends")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
