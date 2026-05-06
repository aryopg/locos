#!/usr/bin/env python3
"""Download haystack data for retrieval head detection.

Fetches needle definitions and per-needle haystack corpora from:
  - NIAH: Retrieval_Head repo (github.com/nightdessert/Retrieval_Head)
  - NoLiMa: Adobe Research NoLiMa repo (github.com/adobe-research/NoLiMa)

Usage:
    # NIAH only (default, backward compatible)
    python locos/download_haystack_data.py

    # NoLiMa only
    python locos/download_haystack_data.py --dataset nolima

    # Both datasets
    python locos/download_haystack_data.py --dataset all
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Ensure repo root is on sys.path so `locos.*` imports work
# regardless of the working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console

console = Console()

REPO_RAW = "https://raw.githubusercontent.com/nightdessert/Retrieval_Head/main"
NEEDLES_URL = f"{REPO_RAW}/haystack_for_detect/needles.jsonl"

# Per-needle haystack files (HF dataset JSON format)
# part1: James Fenimore Cooper (Last of the Mohicans)
# part2: Cooper continued (different chapters)
# part3: Bram Stoker (Dracula)
HAYSTACK_PARTS = {
    "part1/p1.txt": f"{REPO_RAW}/haystack_for_detect/part1/p1.txt",
    "part2/p1.txt": f"{REPO_RAW}/haystack_for_detect/part2/p1.txt",
    "part3/p1.txt": f"{REPO_RAW}/haystack_for_detect/part3/p1.txt",
}


def fetch_url(url: str, max_retries: int = 5) -> str:
    """Fetch text content from a URL with retry on rate-limiting."""
    import time as _time

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = 2**attempt
                console.print(
                    f"[yellow]Rate-limited (429), retrying in {wait}s... ({attempt + 1}/{max_retries})[/yellow]"
                )
                _time.sleep(wait)
            else:
                raise


def extract_text_from_hf_json(raw: str) -> str:
    """Extract plain text from HuggingFace dataset JSON format.

    The haystack files use the format:
        {"features":[...], "rows":[{"row_idx":0, "row":{"text":"...", ...}}, ...]}

    We concatenate all row["text"] fields into a single string.
    """
    try:
        data = json.loads(raw)
        texts = []
        for entry in data["rows"]:
            text = entry["row"].get("text", "")
            if text:
                texts.append(text)
        if texts:
            return "\n\n".join(texts)
    except json.JSONDecodeError:
        # Some upstream files are served as truncated JSON blobs.
        # Fall through to a best-effort recovery path.
        pass

    # Recovery path: extract the first JSON string after `"text":"...`.
    marker = '"text":"'
    start = raw.find(marker)
    if start != -1:
        fragment = raw[start + len(marker) :]
        chars: list[str] = []
        i = 0
        while i < len(fragment):
            ch = fragment[i]
            if ch == '"':
                break
            if ch != "\\":
                chars.append(ch)
                i += 1
                continue

            # JSON escape sequence.
            if i + 1 >= len(fragment):
                break
            esc = fragment[i + 1]
            if esc == "n":
                chars.append("\n")
                i += 2
            elif esc == "t":
                chars.append("\t")
                i += 2
            elif esc == "r":
                chars.append("\r")
                i += 2
            elif esc == "b":
                chars.append("\b")
                i += 2
            elif esc == "f":
                chars.append("\f")
                i += 2
            elif esc in {'"', "\\", "/"}:
                chars.append(esc)
                i += 2
            elif esc == "u":
                # Unicode escape: \uXXXX
                if i + 6 <= len(fragment):
                    hex_digits = fragment[i + 2 : i + 6]
                    try:
                        chars.append(chr(int(hex_digits, 16)))
                        i += 6
                    except ValueError:
                        # Keep parsing if malformed escape appears.
                        i += 2
                else:
                    break
            else:
                # Unknown escape; preserve escaped char best-effort.
                chars.append(esc)
                i += 2

        if chars:
            return "".join(chars)

    # Last resort: return raw content to avoid hard failure.
    return raw


def download_haystack_data(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download needles
    needles_path = output_dir / "needles.jsonl"
    if needles_path.exists():
        console.print(f"[dim]Skipping {needles_path} (already exists)[/dim]")
    else:
        console.print("Downloading needles.jsonl ...")
        content = fetch_url(NEEDLES_URL)
        needles_path.write_text(content)
        lines = [json.loads(line) for line in content.strip().splitlines()]
        assert len(lines) >= 1, "needles.jsonl is empty"
        for line in lines:
            assert "needle" in line and "question" in line and "real_needle" in line
        console.print(f"  [green]Saved {len(lines)} needles[/green]")

    # Download per-needle haystack parts
    for rel_path, url in HAYSTACK_PARTS.items():
        part_dir = output_dir / Path(rel_path).parent
        part_dir.mkdir(parents=True, exist_ok=True)

        # Save both the raw JSON and extracted plain text
        raw_path = output_dir / rel_path
        txt_path = raw_path.with_suffix(".plain.txt")

        if txt_path.exists():
            console.print(f"[dim]Skipping {rel_path} (already exists)[/dim]")
            continue

        console.print(f"Downloading {rel_path} ...")
        raw_content = fetch_url(url)
        raw_path.write_text(raw_content)

        # Extract plain text from HF dataset JSON
        plain_text = extract_text_from_hf_json(raw_content)
        txt_path.write_text(plain_text)
        console.print(f"  [green]Saved {rel_path} ({len(plain_text):,} chars plain text)[/green]")

    console.print(f"\n[bold green]Haystack data ready in {output_dir}[/bold green]")
    console.print("  Structure:")
    console.print("    needles.jsonl       — 3 needle/question/answer triples")
    console.print("    part1/p1.plain.txt  — haystack for needle 0 (Cooper)")
    console.print("    part2/p1.plain.txt  — haystack for needle 1 (Cooper ch.2)")
    console.print("    part3/p1.plain.txt  — haystack for needle 2 (Stoker)")


def download_nolima_data_cli(output_dir: Path) -> None:
    """Download NoLiMa data using the datasets module."""
    from locos.utils.datasets import download_nolima_data

    console.print("[bold]Downloading NoLiMa data...[/bold]")
    download_nolima_data(output_dir)

    console.print(f"\n[bold green]NoLiMa data ready in {output_dir}[/bold green]")
    console.print("  Structure:")
    console.print("    needle_set.json          — 9 needle entries (standard)")
    console.print("    needle_set_hard.json     — 4 needle entries (hard subset)")
    console.print("    haystack/rand_shuffle/   — 5 shuffled book texts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=("Output directory. Defaults: data/haystack_for_detect (niah), " "data/nolima (nolima)"),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="niah",
        choices=["niah", "nolima", "all"],
        help="Which dataset to download (default: niah)",
    )
    args = parser.parse_args()

    if args.dataset in ("niah", "all"):
        niah_dir = args.output_dir or Path("data/haystack_for_detect")
        download_haystack_data(niah_dir)

    if args.dataset in ("nolima", "all"):
        nolima_dir = args.output_dir or Path("data/nolima")
        download_nolima_data_cli(nolima_dir)


if __name__ == "__main__":
    main()
