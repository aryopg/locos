"""Upload experiment results to HuggingFace Hub."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi
from rich.console import Console

console = Console()


def upload_results(
    local_dir: str | Path,
    repo_id: str,
    path_in_repo: str | None = None,
    repo_type: str = "dataset",
    commit_message: str | None = None,
    token: str | None = None,
) -> str:
    """Upload a local directory to a HuggingFace Hub repository.

    Args:
        local_dir: Path to the directory to upload.
        repo_id: HF Hub repo (e.g. ``"aryopg/locos-results"``).
        path_in_repo: Subfolder inside the repo. Defaults to the local
            directory name.
        repo_type: One of ``"dataset"``, ``"model"``, ``"space"``.
        commit_message: Commit message. Auto-generated with a timestamp
            if not provided.
        token: HF API token. Falls back to ``HF_TOKEN`` env var / cached
            login.

    Returns:
        The URL of the repository.
    """
    local_dir = Path(local_dir).resolve()
    assert local_dir.is_dir(), f"Not a directory: {local_dir}"

    if path_in_repo is None:
        path_in_repo = local_dir.name

    if commit_message is None:
        ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        commit_message = f"Upload {path_in_repo} ({ts})"

    api = HfApi(token=token)

    console.print(f"[bold]Creating repo [cyan]{repo_id}[/cyan] if needed...")
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)

    console.print(f"[bold]Uploading [cyan]{local_dir}[/cyan] " f"-> [cyan]{repo_id}/{path_in_repo}[/cyan]...")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        repo_type=repo_type,
        commit_message=commit_message,
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    console.print(f"[bold green]Done![/] {url}")
    return url
