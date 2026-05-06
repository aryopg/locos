#!/usr/bin/env python3
"""Download retrieval heads JSON from HuggingFace Hub.

Examples::

    # Download default heads for a model
    python scripts/download_heads.py --repo-id aryopg/locos-results \
        --heads retrieval_heads/Meta-Llama-3-8B-Instruct.json

    # Download a specific variant
    python scripts/download_heads.py --repo-id aryopg/locos-results \
        --heads retrieval_heads/Meta-Llama-3-8B-Instruct_nolima.json

    # Custom output path
    python scripts/download_heads.py --repo-id aryopg/locos-results \
        --heads retrieval_heads/Meta-Llama-3-8B-Instruct.json \
        --output /tmp/heads.json
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from rich.console import Console

console = Console()


def download_heads(
    repo_id: str,
    heads_path: str,
    output: str | None = None,
    repo_type: str = "dataset",
    token: str | None = None,
) -> Path:
    """Download a retrieval heads file from HuggingFace Hub.

    The HF path mirrors the local path (e.g. ``retrieval_heads/Model.json``).

    Args:
        repo_id: HF Hub repo (e.g. ``"aryopg/locos-results"``).
        heads_path: Path to the heads file, used as both the HF filename and
            the default local output path.
        output: Override local output path. Defaults to *heads_path*.
        repo_type: HF repo type.
        token: HF API token. Falls back to ``HF_TOKEN`` env var / cached login.

    Returns:
        Path to the downloaded file.
    """
    output_path = Path(output or heads_path)

    if output_path.exists():
        console.print(f"[dim]Heads already exist locally: {output_path}[/dim]")
        return output_path

    console.print(
        f"[bold]Downloading [cyan]{heads_path}[/cyan] " f"from [cyan]{repo_id}[/cyan]...",
    )

    try:
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=heads_path,
            repo_type=repo_type,
            token=token,
        )
    except EntryNotFoundError:
        console.print(f"[red]Error:[/red] {heads_path} not found in {repo_id}")
        raise SystemExit(1) from None
    except RepositoryNotFoundError:
        console.print(f"[red]Error:[/red] repository {repo_id} not found")
        raise SystemExit(1) from None

    # Copy from HF cache to the expected local path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, output_path)
    console.print(f"[green]Downloaded:[/green] {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download retrieval heads JSON from HuggingFace Hub.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HF repo id (e.g. aryopg/locos-results).",
    )
    parser.add_argument(
        "--heads",
        required=True,
        help="Heads file path (e.g. retrieval_heads/Model.json). "
        "Used as both the HF filename and the default local output path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override local output path (default: same as --heads).",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF API token (default: HF_TOKEN env var).",
    )
    args = parser.parse_args()

    download_heads(
        repo_id=args.repo_id,
        heads_path=args.heads,
        output=args.output,
        repo_type=args.repo_type,
        token=args.token,
    )


if __name__ == "__main__":
    main()
