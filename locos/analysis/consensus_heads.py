#!/usr/bin/env python3
"""Extract consensus / control head sets from two detector JSONs.

The paper's consensus is defined as the *cross-method, cross-dataset*
intersection::

    consensus_k  =  top-k(Wu-behavioral, NIAH)  ∩  top-k(LogitContrib, NoLiMa)

Given two detector JSONs, this script emits, for each requested k:

- ``{prefix}_consensus_k{k}.json`` — the intersection.
- ``{prefix}_wu_only_k{k}.json``   — heads in method A's top-k but not B's.
- ``{prefix}_lc_only_k{k}.json``   — heads in method B's top-k but not A's.

Each output file is in the flat ``{"layer-head": [score]}`` format that
``locos/analysis/nolima_ablation.py`` consumes (with
``--mode top-k --values <|set|>``). The score value is constant (1.0); only
the keys matter for selection.

Usage:
    python -m locos.analysis.consensus_heads \\
        --method-a-json wu_niah.json \\
        --method-b-json lc_nolima.json \\
        --method-a-label wu \\
        --method-b-label lc \\
        --top-k 10 20 50 \\
        --output-dir consensus_heads/Qwen3-8B \\
        --prefix Qwen3-8B
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

console = Console()


def load_scores(path: str | Path) -> dict[str, list[float]]:
    with open(path) as f:
        data = json.load(f)
    if "scores" in data and isinstance(data["scores"], dict):
        return data["scores"]
    return data


def top_k_keys(scores: dict[str, list[float]], k: int) -> set[str]:
    """Return the keys of the top-k heads by mean score."""
    ranked = sorted(
        scores.items(),
        key=lambda kv: float(np.mean(kv[1])) if kv[1] else 0.0,
        reverse=True,
    )
    return {key for key, _ in ranked[:k]}


def write_head_set(keys: set[str], out_path: Path, meta: dict) -> None:
    """Write a flat head JSON consumable by nolima_ablation.py."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # FIXME(aryo): we emit constant score 1.0 because nolima_ablation.py only
    # uses the keys and a top-k cut. If a downstream consumer starts reading
    # the score values, we should switch to writing the *actual* mean score
    # from one of the methods (e.g. method A) so the ordering is preserved.
    payload = {
        "meta": meta,
        "scores": {k: [1.0] for k in sorted(keys)},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def render_summary(rows: list[dict]) -> Table:
    table = Table(title="Consensus set sizes")
    table.add_column("k", justify="right")
    table.add_column("|A top-k|", justify="right")
    table.add_column("|B top-k|", justify="right")
    table.add_column("|consensus|", justify="right")
    table.add_column("|A only|", justify="right")
    table.add_column("|B only|", justify="right")
    for r in rows:
        table.add_row(
            str(r["k"]),
            str(r["a_size"]),
            str(r["b_size"]),
            str(r["consensus"]),
            str(r["a_only"]),
            str(r["b_only"]),
        )
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-a-json", required=True, help="Path to method A detection JSON (e.g. Wu NIAH)")
    parser.add_argument("--method-b-json", required=True, help="Path to method B detection JSON (e.g. LC NoLiMa)")
    parser.add_argument("--method-a-label", default="a", help="Short label for method A, used in filenames")
    parser.add_argument("--method-b-label", default="b", help="Short label for method B, used in filenames")
    parser.add_argument("--method-a-dataset", default="", help="Dataset tag for method A (for meta only)")
    parser.add_argument("--method-b-dataset", default="", help="Dataset tag for method B (for meta only)")
    parser.add_argument("--top-k", type=int, nargs="+", required=True, help="k values to sweep")
    parser.add_argument("--output-dir", required=True, help="Directory to write the head JSONs into")
    parser.add_argument("--prefix", required=True, help="Filename prefix (typically the model slug)")
    args = parser.parse_args()

    scores_a = load_scores(args.method_a_json)
    scores_b = load_scores(args.method_b_json)
    out_dir = Path(args.output_dir)

    rows: list[dict] = []
    for k in args.top_k:
        a_set = top_k_keys(scores_a, k)
        b_set = top_k_keys(scores_b, k)
        consensus = a_set & b_set
        a_only = a_set - b_set
        b_only = b_set - a_set

        meta_common = {
            "method_a": args.method_a_label,
            "method_b": args.method_b_label,
            "method_a_dataset": args.method_a_dataset,
            "method_b_dataset": args.method_b_dataset,
            "top_k_per_method": k,
            "method_a_json": args.method_a_json,
            "method_b_json": args.method_b_json,
        }

        write_head_set(
            consensus,
            out_dir / f"{args.prefix}_consensus_k{k}.json",
            {**meta_common, "set_kind": "consensus_intersection"},
        )
        write_head_set(
            a_only,
            out_dir / f"{args.prefix}_{args.method_a_label}_only_k{k}.json",
            {**meta_common, "set_kind": f"{args.method_a_label}_topk_minus_{args.method_b_label}"},
        )
        write_head_set(
            b_only,
            out_dir / f"{args.prefix}_{args.method_b_label}_only_k{k}.json",
            {**meta_common, "set_kind": f"{args.method_b_label}_topk_minus_{args.method_a_label}"},
        )

        rows.append(
            {
                "k": k,
                "a_size": len(a_set),
                "b_size": len(b_set),
                "consensus": len(consensus),
                "a_only": len(a_only),
                "b_only": len(b_only),
            }
        )

    console.print(render_summary(rows))

    # Manifest CSV for the plotting stage
    manifest_path = out_dir / f"{args.prefix}_consensus_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "a_size", "b_size", "consensus", "a_only", "b_only"])
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"[green]Wrote manifest:[/green] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
