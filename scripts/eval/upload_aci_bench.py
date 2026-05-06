#!/usr/bin/env python3
"""Upload ACI-Bench D2N data to HuggingFace Hub.

Usage:
    python scripts/eval/upload_aci_bench.py --hf-repo aryopg/aci-bench-d2n
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "aci_bench_d2n"


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Upload ACI-Bench D2N to HF Hub")
    parser.add_argument("--hf-repo", default="aryopg/aci-bench-d2n")
    args = parser.parse_args()

    train = read_jsonl(DATA_DIR / "train.jsonl")
    test = read_jsonl(DATA_DIR / "test.jsonl")

    ds_dict = DatasetDict(
        {
            "train": Dataset.from_list(train),
            "test": Dataset.from_list(test),
        }
    )

    print(f"Train: {len(train)}, Test: {len(test)}")
    print(f"Uploading to {args.hf_repo}...")
    ds_dict.push_to_hub(args.hf_repo)
    print("Done!")


if __name__ == "__main__":
    main()
