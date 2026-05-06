#!/usr/bin/env python3
"""Build parametric recall and arithmetic evaluation dataset.

Creates a 300-sample dataset for control experiments in retrieval head
ablation: parametric knowledge (City-Country, PopQA) should not degrade
when retrieval heads are ablated, while arithmetic should also be stable.
This tests that identified retrieval heads are retrieval-specific, not
generically output-critical.

Sources:
  - City-Country: WorkWithData/cities (top 100 by population)
  - PopQA: akariasai/PopQA (top 100 by max(s_pop, o_pop))
  - Arithmetic: EleutherAI/arithmetic, subset arithmetic_1dc (100 random)

Usage:
    python scripts/eval/build_parametric_and_arithmetic_dataset.py \\
        --hf-repo aryopg/parametric-arithmetic-eval

    # Local only (no upload)
    python scripts/eval/build_parametric_and_arithmetic_dataset.py \\
        --hf-repo aryopg/parametric-arithmetic-eval \\
        --skip-upload --output-dir data/parametric_arithmetic
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from datasets import Dataset, load_dataset
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------


def _load_city_country(n: int = 100) -> list[dict]:
    """Load top-N cities by population, format as country-identification QA."""
    ds = load_dataset("WorkWithData/cities", split="train")
    records = list(ds)
    assert len(records) > 0, "WorkWithData/cities returned empty dataset"

    # Validate expected columns
    required = {"city", "country", "population"}
    actual = set(records[0].keys())
    assert required.issubset(actual), f"Missing columns: {required - actual} (have {actual})"

    # Sort by population descending, take top N
    records.sort(key=lambda r: r["population"], reverse=True)
    records = records[:n]

    # Normalise country names to common English forms
    _COUNTRY_NORMALISE = {
        "Korea": "South Korea",
        "United States": "United States of America",
        "Dem. Rep. Congo": "Democratic Republic of the Congo",
        "Syrian Arab Republic": "Syria",
    }

    return [
        {
            "question": f"Which country does {r['city']} belong to?",
            "answer": _COUNTRY_NORMALISE.get(r["country"], r["country"]),
            "source": "city_country",
        }
        for r in records
    ]


def _load_popqa(n: int = 100) -> list[dict]:
    """Load top-N PopQA entries by entity popularity, format as short-form QA."""
    ds = load_dataset("akariasai/PopQA", split="test")
    records = list(ds)
    assert len(records) > 0, "akariasai/PopQA returned empty dataset"

    # Validate expected columns
    required = {"question", "obj", "subj", "s_pop", "o_pop"}
    actual = set(records[0].keys())
    assert required.issubset(actual), f"Missing columns: {required - actual} (have {actual})"

    # Sort by geometric mean of popularity — penalises lopsided (s_pop, o_pop)
    # pairs so we don't end up with 95 % "United States of America" answers
    records.sort(key=lambda r: math.sqrt(max(r["s_pop"], 1) * max(r["o_pop"], 1)), reverse=True)

    # Deduplicate by {subj, obj} pair — keep first (highest-ranked) occurrence
    seen: set[frozenset] = set()
    deduped = []
    for r in records:
        pair = frozenset({r["subj"], r["obj"]})
        if pair not in seen:
            seen.add(pair)
            deduped.append(r)
    records = deduped[:n]

    return [
        {
            "question": r["question"],
            "answer": r["obj"],
            "source": "popqa",
        }
        for r in records
    ]


def _load_arithmetic(n: int = 100, seed: int = 42) -> list[dict]:
    """Load N random arithmetic_1dc samples, reformat to clean QA."""
    ds = load_dataset("EleutherAI/arithmetic", "arithmetic_1dc", split="validation")
    records = list(ds)
    assert len(records) > 0, "EleutherAI/arithmetic (arithmetic_1dc) returned empty dataset"

    # Validate expected columns
    required = {"context", "completion"}
    actual = set(records[0].keys())
    assert required.issubset(actual), f"Missing columns: {required - actual} (have {actual})"

    # Random sample
    rng = random.Random(seed)
    records = rng.sample(records, min(n, len(records)))

    results = []
    for r in records:
        question = r["context"]
        # Strip "Question: " prefix and " Answer:" suffix
        question = question.removeprefix("Question:").removesuffix("Answer:").strip()
        answer = r["completion"].strip()
        results.append(
            {
                "question": question,
                "answer": answer,
                "source": "arithmetic",
            }
        )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build parametric recall and arithmetic evaluation dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hf-repo", required=True, help="HuggingFace dataset repo to push to")
    parser.add_argument("--skip-upload", action="store_true", help="Skip HF upload, save locally only")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/parametric_arithmetic"),
        help="Local output directory for JSONL fallback",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for arithmetic sampling")
    parser.add_argument("--n-per-source", type=int, default=200, help="Number of samples per source")
    args = parser.parse_args()

    console.print(
        Panel(
            f"[bold]HF repo:[/bold] {args.hf_repo}\n"
            f"[bold]Upload:[/bold] {'yes' if not args.skip_upload else 'no (local only)'}\n"
            f"[bold]Samples per source:[/bold] {args.n_per_source}\n"
            f"[bold]Seed:[/bold] {args.seed}",
            title="[green]Parametric & Arithmetic Dataset Builder[/green]",
        )
    )

    # Load all sources
    console.rule("[bold]Loading sources[/bold]")

    console.print("Loading City-Country (WorkWithData/cities)...")
    city_country = _load_city_country(n=args.n_per_source)
    console.print(f"  {len(city_country)} samples")

    console.print("Loading PopQA (akariasai/PopQA)...")
    popqa = _load_popqa(n=args.n_per_source)
    console.print(f"  {len(popqa)} samples")

    console.print("Loading Arithmetic (EleutherAI/arithmetic, arithmetic_1dc)...")
    arithmetic = _load_arithmetic(n=args.n_per_source, seed=args.seed)
    console.print(f"  {len(arithmetic)} samples")

    # Combine and index
    all_records = city_country + popqa + arithmetic
    for i, record in enumerate(all_records):
        record["index"] = i

    console.print(f"\nTotal: {len(all_records)} samples")

    # Summary table
    summary = Table(title="Dataset Summary")
    summary.add_column("Source", style="bold")
    summary.add_column("Count", justify="right")
    summary.add_column("Example Question")
    summary.add_column("Example Answer")

    for source, records in [("city_country", city_country), ("popqa", popqa), ("arithmetic", arithmetic)]:
        summary.add_row(
            source,
            str(len(records)),
            records[0]["question"][:60] + ("..." if len(records[0]["question"]) > 60 else ""),
            records[0]["answer"],
        )
    console.print(summary)

    # Save locally
    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_path = args.output_dir / "parametric_arithmetic.jsonl"
    with open(local_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    console.print(f"\nSaved locally: {local_path}")

    # Upload to HuggingFace
    if not args.skip_upload:
        console.rule("[bold]Uploading to HuggingFace[/bold]")
        console.print(f"Uploading to {args.hf_repo}...")
        ds = Dataset.from_list(all_records)
        ds.push_to_hub(args.hf_repo)
        console.print("[green]Upload complete.[/green]")
    else:
        console.print("[dim]Skipping HF upload (--skip-upload)[/dim]")

    console.print("\n[green]Done![/green]")


if __name__ == "__main__":
    main()
