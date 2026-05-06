#!/usr/bin/env python3
"""Reorganize HuggingFace repo: flatten model-slug directories into top-level result dirs.

Moves files from:
    <model_slug>/ablation_results/...       → ablation_results/...
    <model_slug>/ablation_parametric_results/... → ablation_parametric_results/...
    <model_slug>/retrieval_heads/...        → retrieval_heads/...

Deletes:
    <model_slug>/logs/...

This is a one-time migration script. After running, model-slug directories at the
repo root should be empty and only the flat top-level directories remain.

Usage::

    python scripts/reorganize_hf_repo.py --repo-id aryopg/locos-results --dry-run
    python scripts/reorganize_hf_repo.py --repo-id aryopg/locos-results
"""

from __future__ import annotations

import argparse
import re
import tempfile

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from rich.console import Console
from rich.table import Table

console = Console()

# Known top-level directories that are already correctly structured
KNOWN_GOOD_PREFIXES = {"ablation_results", "ablation_parametric_results", "eval_results", "retrieval_heads"}

# Result subdirectories inside model-slug dirs that should be flattened
RESULT_SUBDIRS = {"ablation_results", "ablation_parametric_results", "retrieval_heads"}

# Subdirectories to delete entirely
DELETE_SUBDIRS = {"logs"}

# Pattern matching model-slug directories (provider_ModelName format)
MODEL_SLUG_RE = re.compile(r"^[a-zA-Z][\w.-]+_[\w.-]+$")


def find_model_slug_dirs(api: HfApi, repo_id: str, repo_type: str) -> list[str]:
    """Find top-level directories that look like model slugs."""
    entries = api.list_repo_tree(repo_id, repo_type=repo_type, recursive=False)
    model_dirs = []
    for entry in entries:
        if type(entry).__name__ != "RepoFolder":
            continue
        name = entry.path
        if name in KNOWN_GOOD_PREFIXES:
            continue
        if MODEL_SLUG_RE.match(name):
            model_dirs.append(name)
    return sorted(model_dirs)


def plan_operations(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    model_dirs: list[str],
) -> tuple[list[tuple[str, str]], list[str]]:
    """Plan move and delete operations.

    Returns:
        (moves, deletes) where moves is [(old_path, new_path), ...] and
        deletes is [path, ...] for files to delete outright (logs).
    """
    moves: list[tuple[str, str]] = []
    deletes: list[str] = []

    all_entries = api.list_repo_tree(repo_id, repo_type=repo_type, recursive=True)
    files = [e.path for e in all_entries if type(e).__name__ == "RepoFile"]

    for file_path in files:
        parts = file_path.split("/")
        if len(parts) < 3:
            continue

        model_slug = parts[0]
        if model_slug not in model_dirs:
            continue

        subdir = parts[1]

        if subdir in DELETE_SUBDIRS:
            deletes.append(file_path)
        elif subdir in RESULT_SUBDIRS:
            # Flatten: strip the model-slug prefix
            new_path = "/".join(parts[1:])
            moves.append((file_path, new_path))

    return moves, deletes


def execute(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    moves: list[tuple[str, str]],
    deletes: list[str],
    dry_run: bool,
    token: str | None,
) -> None:
    """Execute the reorganization as a single HF commit."""
    if not moves and not deletes:
        console.print("[green]Nothing to do — repo is already clean.[/green]")
        return

    # Summary table
    table = Table(title="Planned operations")
    table.add_column("Operation", style="bold")
    table.add_column("From", style="cyan")
    table.add_column("To", style="green")

    for old, new in moves:
        table.add_row("MOVE", old, new)
    for path in deletes:
        table.add_row("DELETE", path, "")

    console.print(table)
    console.print(f"\nTotal: {len(moves)} moves, {len(deletes)} deletes")

    if dry_run:
        console.print("[yellow]Dry run — no changes made.[/yellow]")
        return

    # Build commit operations:
    # - For moves: download file content, create Add at new path + Delete at old path
    # - For deletes: just Delete
    operations = []

    # Download files that need to be moved
    console.print("\n[bold]Downloading files to move...[/bold]")
    with tempfile.TemporaryDirectory() as tmpdir:
        for old_path, new_path in moves:
            console.print(f"  Downloading {old_path}")
            local_path = api.hf_hub_download(
                repo_id=repo_id,
                filename=old_path,
                repo_type=repo_type,
                local_dir=tmpdir,
            )
            operations.append(
                CommitOperationAdd(
                    path_in_repo=new_path,
                    path_or_fileobj=local_path,
                )
            )

        # Delete old paths (both moved files and logs)
        for old_path, _ in moves:
            operations.append(CommitOperationDelete(path_in_repo=old_path))
        for path in deletes:
            operations.append(CommitOperationDelete(path_in_repo=path))

        console.print(f"\n[bold]Committing {len(operations)} operations...[/bold]")
        api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=operations,
            commit_message=(
                f"reorganize: flatten model-slug dirs into top-level result dirs "
                f"({len(moves)} moved, {len(deletes)} deleted)"
            ),
        )

    console.print("[bold green]Done![/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorganize HF repo: flatten model-slug directories.",
    )
    parser.add_argument("--repo-id", required=True, help="HF dataset repo (e.g. aryopg/locos-results)")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    parser.add_argument("--token", default=None, help="HF API token")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    console.print(f"[bold]Scanning {args.repo_id}...[/bold]")
    model_dirs = find_model_slug_dirs(api, args.repo_id, args.repo_type)

    if not model_dirs:
        console.print("[green]No model-slug directories found at repo root.[/green]")
        return

    console.print(f"Found {len(model_dirs)} model-slug directories to flatten:")
    for d in model_dirs:
        console.print(f"  {d}")

    moves, deletes = plan_operations(api, args.repo_id, args.repo_type, model_dirs)
    execute(api, args.repo_id, args.repo_type, moves, deletes, args.dry_run, args.token)


if __name__ == "__main__":
    main()
