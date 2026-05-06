#!/usr/bin/env python3
"""Pre-compute BM25 retrieval for MedRAG benchmark datasets.

Retrieves top-k PubMed passages per question using Lucene BM25 (via Pyserini)
and uploads the combined dataset to HuggingFace Hub.

Usage:
    python scripts/eval/build_medrag_dataset.py \
        --hf-repo aryopg/medrag-bm25-pubmed \
        --top-k 32

    # Resume from checkpoint after interruption
    python scripts/eval/build_medrag_dataset.py \
        --hf-repo aryopg/medrag-bm25-pubmed \
        --checkpoint-dir checkpoints/medrag

    # Build a single sub-dataset
    python scripts/eval/build_medrag_dataset.py \
        --hf-repo aryopg/medrag-bm25-pubmed \
        --datasets medqa

    # Skip indexing if index already built
    python scripts/eval/build_medrag_dataset.py \
        --hf-repo aryopg/medrag-bm25-pubmed \
        --index-dir indexes/pubmed

Requires: pip install pyserini faiss-cpu datasets
"""

import argparse
import json
import string
import subprocess
import sys
from pathlib import Path

from datasets import Dataset, load_dataset
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Sub-dataset loaders
# ---------------------------------------------------------------------------

MMLU_MED_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "professional_medicine",
    "college_medicine",
    "medical_genetics",
    "college_biology",
]

SUPERGPQA_MED_FIELDS = [
    "Clinical Medicine",
    "Basic Medicine",
    "Pharmacy",
    "Stomatology",
    "Public Health and Preventive Medicine",
]


def _index_to_letter(idx: int) -> str:
    """Convert 0-based index to letter (0->A, 1->B, ...)."""
    return string.ascii_uppercase[idx]


def _load_mmlu_med() -> list[dict]:
    """Load MMLU medical subjects (test split)."""
    records = []
    for subject in track(MMLU_MED_SUBJECTS, description="MMLU-Med subjects"):
        ds = load_dataset("cais/mmlu", subject, split="test")
        for row in ds:
            options = {_index_to_letter(i): choice for i, choice in enumerate(row["choices"])}
            records.append(
                {
                    "question": row["question"],
                    "options": options,
                    "answer": _index_to_letter(row["answer"]),
                    "dataset": "mmlu_med",
                }
            )
    console.print(f"  MMLU-Med: {len(records)} questions")
    return records


def _load_medqa() -> list[dict]:
    """Load MedQA-USMLE 4-option (test split)."""
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    records = []
    for row in ds:
        records.append(
            {
                "question": row["question"],
                "options": row["options"],
                "answer": row["answer_idx"],
                "dataset": "medqa",
            }
        )
    console.print(f"  MedQA: {len(records)} questions")
    return records


def _load_medmcqa() -> list[dict]:
    """Load MedMCQA (validation split — test split has no labels)."""
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    records = []
    for row in ds:
        options = {
            "A": row["opa"],
            "B": row["opb"],
            "C": row["opc"],
            "D": row["opd"],
        }
        records.append(
            {
                "question": row["question"],
                "options": options,
                "answer": _index_to_letter(row["cop"]),
                "dataset": "medmcqa",
            }
        )
    console.print(f"  MedMCQA: {len(records)} questions")
    return records


def _load_pubmedqa() -> list[dict]:
    """Load PubMedQA (expert-labeled subset)."""
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    records = []
    for row in ds:
        records.append(
            {
                "question": row["question"],
                "options": {"A": "yes", "B": "no", "C": "maybe"},
                "answer": {"yes": "A", "no": "B", "maybe": "C"}[row["final_decision"]],
                "dataset": "pubmedqa",
            }
        )
    console.print(f"  PubMedQA: {len(records)} questions")
    return records


def _load_supergpqa_med() -> list[dict]:
    """Load SuperGPQA Medicine questions (filtered by discipline and field)."""
    ds = load_dataset("m-a-p/SuperGPQA", split="train")
    records = []
    for row in ds:
        if row.get("discipline") != "Medicine":
            continue
        if row.get("field") not in SUPERGPQA_MED_FIELDS:
            continue
        options_list = row.get("options", [])
        if not options_list:
            continue
        options = {_index_to_letter(i): opt for i, opt in enumerate(options_list)}
        records.append(
            {
                "question": row["question"],
                "options": options,
                "answer": row["answer_letter"],
                "dataset": "supergpqa_med",
            }
        )
    console.print(f"  SuperGPQA-Med: {len(records)} questions")
    return records


DATASET_LOADERS = {
    "mmlu_med": _load_mmlu_med,
    "medqa": _load_medqa,
    "medmcqa": _load_medmcqa,
    "pubmedqa": _load_pubmedqa,
    "supergpqa_med": _load_supergpqa_med,
}


# ---------------------------------------------------------------------------
# Lucene BM25 indexing & retrieval (via Pyserini)
# ---------------------------------------------------------------------------

JSONL_BATCH_SIZE = 50_000  # docs per JSONL shard written to disk


def build_lucene_index(
    corpus_name: str,
    index_dir: Path,
    threads: int = 8,
) -> None:
    """Stream corpus from HuggingFace and build a Lucene BM25 index on disk.

    Writes JSONL doc files to a temp directory, then invokes Pyserini's indexer.
    """
    assert threads >= 1, f"threads must be >= 1, got {threads}"

    if (index_dir / "segments.gen").exists() or list(index_dir.glob("segments_*")):
        console.print(f"  Index already exists at {index_dir}, skipping build.")
        return

    index_dir.mkdir(parents=True, exist_ok=True)

    # -- Step A: stream corpus → JSONL shards on disk --
    jsonl_dir = index_dir / "_jsonl_staging"
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    # Check for a completion marker — if staging finished previously, skip it
    staging_done_marker = jsonl_dir / "_STAGING_COMPLETE"
    existing_shards = sorted(jsonl_dir.glob("docs_*.jsonl"))

    if staging_done_marker.exists():
        console.print(f"  JSONL staging already complete ({len(existing_shards)} shards), " "skipping to indexing.")
    else:
        # Determine resume point from existing shards
        if existing_shards:
            # Delete last shard (may be incomplete from interruption)
            last_shard = existing_shards[-1]
            last_shard.unlink()
            existing_shards = existing_shards[:-1]

            # Count docs in all remaining (complete) shards
            n_written = len(existing_shards) * JSONL_BATCH_SIZE
            next_shard_id = len(existing_shards)
            console.print(
                f"  Resuming: {len(existing_shards)} complete shards "
                f"({n_written:,} docs), deleted last partial shard."
            )
        else:
            n_written = 0
            next_shard_id = 0

        console.print(f"  Streaming corpus from {corpus_name}...")
        corpus_ds = load_dataset(corpus_name, split="train", streaming=True)

        # Use HF datasets skip() to efficiently jump past already-staged docs
        if n_written > 0:
            console.print(f"  Skipping first {n_written:,} docs...")
            corpus_ds = corpus_ds.skip(n_written)

        batch = []
        doc_id = n_written
        shard_id = next_shard_id

        for row in track(corpus_ds, description="  Staging JSONL"):
            # Pyserini expects {"id": str, "contents": str}
            title = row.get("title", "")
            content = row.get("content", "")
            contents = row.get("contents", "") or f"{title}\n{content}".strip()

            batch.append(
                {
                    "id": str(doc_id),
                    "contents": contents,
                }
            )
            doc_id += 1

            if len(batch) >= JSONL_BATCH_SIZE:
                shard_path = jsonl_dir / f"docs_{shard_id:06d}.jsonl"
                with open(shard_path, "w") as f:
                    for doc in batch:
                        f.write(json.dumps(doc) + "\n")
                batch = []
                shard_id += 1

        # Write remaining docs
        if batch:
            shard_path = jsonl_dir / f"docs_{shard_id:06d}.jsonl"
            with open(shard_path, "w") as f:
                for doc in batch:
                    f.write(json.dumps(doc) + "\n")

        # Mark staging as complete
        staging_done_marker.touch()
        console.print(f"  {doc_id:,} docs staged in {jsonl_dir}")

    # -- Step B: run Pyserini indexer --
    console.print(f"  Building Lucene index at {index_dir} (threads={threads})...")
    cmd = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(jsonl_dir),
        "--index",
        str(index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    console.print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    console.print("  Lucene index built.")


def retrieve_passages(
    searcher,
    question: str,
    top_k: int,
) -> list[dict]:
    """Retrieve top-k passages for a question using the Lucene BM25 index."""
    hits = searcher.search(question, k=top_k)

    passages = []
    for hit in hits:
        raw = json.loads(hit.lucene_document.get("raw"))
        passages.append(
            {
                "content": raw.get("contents", ""),
                "score": float(hit.score),
            }
        )
    return passages


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(records: list[dict], checkpoint_dir: Path, dataset_name: str):
    """Save intermediate results for a sub-dataset."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{dataset_name}.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    console.print(f"  Checkpoint saved: {path} ({len(records)} records)")


def load_checkpoint(checkpoint_dir: Path, dataset_name: str) -> list[dict] | None:
    """Load checkpoint if it exists."""
    path = checkpoint_dir / f"{dataset_name}.jsonl"
    if not path.exists():
        return None
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    console.print(f"  Loaded checkpoint: {path} ({len(records)} records)")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Build MedRAG pre-retrieved dataset")
    parser.add_argument("--hf-repo", required=True, help="HuggingFace repo to upload to (e.g., aryopg/medrag-bm25)")
    parser.add_argument(
        "--top-k", type=int, default=32, help="Number of passages to retrieve per question (default: 32)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=list(DATASET_LOADERS.keys()),
        help="Specific datasets to process (default: all)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/medrag", help="Directory for intermediate checkpoints"
    )
    parser.add_argument("--skip-upload", action="store_true", help="Skip HuggingFace upload (save locally only)")
    parser.add_argument(
        "--corpus", default="MedRAG/pubmed", help="HuggingFace corpus for retrieval (default: MedRAG/pubmed)"
    )
    parser.add_argument(
        "--index-dir", type=str, default="indexes/pubmed", help="Directory for Lucene index (default: indexes/pubmed)"
    )
    parser.add_argument("--threads", type=int, default=8, help="Threads for Lucene indexing (default: 8)")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    index_dir = Path(args.index_dir)
    datasets_to_process = args.datasets or list(DATASET_LOADERS.keys())

    console.print(
        Panel(
            f"HF repo:     {args.hf_repo}\n"
            f"Top-k:       {args.top_k}\n"
            f"Datasets:    {', '.join(datasets_to_process)}\n"
            f"Corpus:      {args.corpus}\n"
            f"Index dir:   {index_dir}\n"
            f"Checkpoints: {checkpoint_dir}",
            title="[bold]MedRAG Dataset Builder[/bold]",
        )
    )
    console.print()

    # Step 1: Load all QA datasets
    console.rule("Step 1: Loading QA datasets")
    all_qa_records: dict[str, list[dict]] = {}
    for name in datasets_to_process:
        all_qa_records[name] = DATASET_LOADERS[name]()
    console.print()

    # Step 2: Build Lucene index (skipped if already exists)
    console.rule("Step 2: Building Lucene BM25 index")
    build_lucene_index(args.corpus, index_dir, threads=args.threads)
    console.print()

    # Step 3: Retrieve passages for each dataset
    console.rule("Step 3: Retrieving passages")
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(index_dir))

    all_records = []

    for dataset_name, qa_records in all_qa_records.items():
        # Check for checkpoint
        cached = load_checkpoint(checkpoint_dir, dataset_name)
        if cached is not None:
            all_records.extend(cached)
            continue

        console.print(f"  Retrieving for {dataset_name} ({len(qa_records)} questions)...")
        enriched = []
        for record in track(qa_records, description=f"  {dataset_name}"):
            passages = retrieve_passages(searcher, record["question"], args.top_k)
            enriched.append({**record, "retrieved_passages": passages})

        save_checkpoint(enriched, checkpoint_dir, dataset_name)
        all_records.extend(enriched)

    console.print(f"\n  Total records: {len(all_records)}")

    # Step 4: Upload to HuggingFace
    if not args.skip_upload:
        console.rule("Step 4: Uploading to HuggingFace")
        console.print(f"  Uploading to {args.hf_repo}...")
        ds = Dataset.from_list(all_records)
        ds.push_to_hub(args.hf_repo)
        console.print("  Upload complete.")
    else:
        # Save locally
        out_path = checkpoint_dir / "combined.jsonl"
        with open(out_path, "w") as f:
            for r in all_records:
                f.write(json.dumps(r) + "\n")
        console.print(f"\nSaved to {out_path}")

    # Final summary table
    summary_table = Table(title="Summary")
    summary_table.add_column("Dataset", style="bold")
    summary_table.add_column("Questions")
    for name, records in all_qa_records.items():
        summary_table.add_row(name, str(len(records)))
    summary_table.add_row("[bold]Total[/bold]", f"[bold]{len(all_records)}[/bold]")
    console.print()
    console.print(summary_table)
    console.print(
        Panel(
            f"Done! {len(all_records)} questions with top-{args.top_k} passages each.",
            title="[bold]Complete[/bold]",
        )
    )


if __name__ == "__main__":
    main()
