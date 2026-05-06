#!/usr/bin/env python3
"""Scan local eval_results/ and sync new runs to HuggingFace Hub.

Pure scanning logic (scan_local_experiments, config_hash, compute_metrics_summary)
has no HF dependency and is fully unit-tested. The sync_experiment function uses
huggingface_hub for uploads.

Usage::

    python scripts/sync_results.py --repo-id aryopg/locos-results
    python scripts/sync_results.py --repo-id aryopg/locos-results --dry-run
    python scripts/sync_results.py --repo-id aryopg/locos-results --local-dir ./eval_results
    python scripts/sync_results.py --repo-id aryopg/locos-results --rebuild
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Default prefix for eval results in HF repo. Override with --hf-prefix on
# the CLI, or infer from the local-dir basename when it differs from the default.
DEFAULT_HF_EVAL_PREFIX = "eval_results"

# Regex for results files: results_YYYYMMDD_HHMMSS.jsonl
_RESULTS_RE = re.compile(r"^results_(\d{8}_\d{6})\.jsonl$")
# Regex for config sidecars: results_YYYYMMDD_HHMMSS_config.json
_CONFIG_RE = re.compile(r"^results_(\d{8}_\d{6})_config\.json$")
# Regex for generation files: *_generations.jsonl or generations.jsonl
_GENERATIONS_RE = re.compile(r"^.*generations\.jsonl$")


# ---------------------------------------------------------------------------
# Pure helpers (no HF dependency)
# ---------------------------------------------------------------------------


def config_hash(config_path: Path) -> str:
    """SHA-256 of config JSON with timestamp removed, sorted keys.

    The timestamp field is excluded so that re-runs of the same config
    produce the same hash. Returns empty string if file is missing or malformed.
    """
    if not config_path.exists():
        console.print(f"  [yellow]WARNING: config file missing: {config_path}[/yellow]")
        return ""
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except json.JSONDecodeError as exc:
        console.print(f"  [yellow]WARNING: malformed config JSON {config_path}: {exc}[/yellow]")
        return ""
    # Remove timestamp — it varies between runs of the same config
    cfg.pop("timestamp", None)
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def compute_metrics_summary(results_path: Path) -> dict[str, float]:
    """Mean of each numeric score across all results in a JSONL file.

    Each line must have a ``"scores"`` dict with string keys and numeric values.
    Returns an empty dict for empty files or on parse errors.
    """
    if not results_path.exists():
        return {}

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    with open(results_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                console.print(f"  [yellow]WARNING: malformed JSONL line {line_num} in {results_path.name}[/yellow]")
                continue
            scores = record.get("scores", {})
            if not isinstance(scores, dict):
                continue
            for key, value in scores.items():
                if isinstance(value, int | float):
                    sums[key] = sums.get(key, 0.0) + value
                    counts[key] = counts.get(key, 0) + 1

    return {k: sums[k] / counts[k] for k in sums}


def scan_local_experiments(local_dir: Path | str) -> list[dict[str, Any]]:
    """Scan a local directory for experiment results.

    Walks 3 levels deep: ``{task}/{model_slug}/{variant}/``.

    For each variant directory containing at least one ``results_YYYYMMDD_HHMMSS.jsonl``
    file, produces an experiment descriptor with runs, metrics, and generation files.

    Returns a list of experiment descriptors sorted by experiment_key.
    """
    local_dir = Path(local_dir).resolve()
    assert local_dir.is_dir(), f"Not a directory: {local_dir}"

    experiments: list[dict[str, Any]] = []

    # Walk exactly 3 levels: task / model_slug / variant
    for task_dir in sorted(local_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for variant_dir in sorted(model_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue

                exp = _scan_variant_dir(
                    variant_dir,
                    task=task_dir.name,
                    model_slug=model_dir.name,
                    variant=variant_dir.name,
                )
                if exp is not None:
                    experiments.append(exp)

    return experiments


def _scan_variant_dir(
    variant_dir: Path,
    task: str,
    model_slug: str,
    variant: str,
) -> dict[str, Any] | None:
    """Scan a single variant directory for results, configs, and generations.

    Returns None if no timestamped results files are found.
    """
    files = sorted(variant_dir.iterdir())
    filenames = {f.name for f in files if f.is_file()}

    # Find all timestamped results files
    results_timestamps: list[str] = []
    for name in sorted(filenames):
        m = _RESULTS_RE.match(name)
        if m:
            results_timestamps.append(m.group(1))

    if not results_timestamps:
        return None

    # Build run descriptors
    runs: list[dict[str, Any]] = []
    for ts in results_timestamps:
        results_name = f"results_{ts}.jsonl"
        config_name = f"results_{ts}_config.json"

        run_files = [results_name]
        if config_name in filenames:
            run_files.append(config_name)

        # Line count for n_samples
        results_path = variant_dir / results_name
        n_samples = _count_jsonl_lines(results_path)

        # Config hash
        config_path = variant_dir / config_name
        cfg_hash: str | None = None
        if config_path.exists():
            cfg_hash = config_hash(config_path)

        # Metrics
        metrics = compute_metrics_summary(results_path)

        runs.append(
            {
                "timestamp": ts,
                "files": run_files,
                "n_samples": n_samples,
                "config_hash": cfg_hash,
                "metrics": metrics,
            }
        )

    # Find generation files
    generation_files: list[str] = []
    for name in sorted(filenames):
        if _GENERATIONS_RE.match(name):
            generation_files.append(name)

    return {
        "experiment_key": f"{task}/{model_slug}/{variant}",
        "task": task,
        "model_slug": model_slug,
        "variant": variant,
        "local_path": variant_dir,
        "runs": runs,
        "generation_files": generation_files,
    }


def _count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ---------------------------------------------------------------------------
# Sync logic (uses HuggingFace Hub)
# ---------------------------------------------------------------------------


def sync_experiment(
    experiment: dict[str, Any],
    repo_id: str,
    *,
    hf_prefix: str = DEFAULT_HF_EVAL_PREFIX,
    dry_run: bool = False,
    rebuild: bool = False,
    token: str | None = None,
) -> bool:
    """Sync a single experiment to HuggingFace Hub.

    1. Download existing manifest (if any) from HF.
    2. Determine which runs are new (timestamp not in manifest).
    3. Build CommitOperationAdd list for new files + generations + updated manifest.
    4. Create a single HF commit per experiment.

    Returns True if anything was uploaded, False if already up to date.
    """
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.utils import EntryNotFoundError

    from locos_eval.evals.manifest import ExperimentManifest

    api = HfApi(token=token)
    exp_key = experiment["experiment_key"]
    local_path = Path(experiment["local_path"])
    exp_hf_prefix = f"{hf_prefix}/{exp_key}"
    manifest_hf_path = f"{exp_hf_prefix}/manifest.json"

    # 1. Try to download existing manifest
    manifest: ExperimentManifest | None = None
    if not rebuild:
        try:
            local_manifest = api.hf_hub_download(
                repo_id=repo_id,
                filename=manifest_hf_path,
                repo_type="dataset",
            )
            manifest = ExperimentManifest.load(Path(local_manifest))
        except (EntryNotFoundError, Exception):
            manifest = None

    if manifest is None:
        # Parse variant to infer decoding mode
        variant = experiment["variant"]
        decoding = variant.split("_")[0] if "_" in variant else variant
        # Try to get the real model name from the first config file
        model_name = experiment["model_slug"]  # fallback: slug
        if experiment["runs"]:
            first_config = experiment["local_path"] / f"results_{experiment['runs'][0]['timestamp']}_config.json"
            if first_config.exists():
                with open(first_config) as f:
                    model_name = json.load(f).get("model", model_name)
        manifest = ExperimentManifest(
            experiment_key=exp_key,
            task=experiment["task"],
            model=model_name,
            decoding=decoding,
            variant=variant,
        )

    # 2. Determine new runs
    new_runs = [r for r in experiment["runs"] if not manifest.has_timestamp(r["timestamp"])]

    if not new_runs and not rebuild:
        return False

    # 3. Build commit operations
    operations: list[CommitOperationAdd] = []

    for run in new_runs:
        manifest.add_run(
            timestamp=run["timestamp"],
            n_samples=run["n_samples"],
            limit=None,  # Not tracked in scan
            metrics=run["metrics"],
            config_hash=run["config_hash"] or "",
            files=run["files"],
        )

        for filename in run["files"]:
            file_path = local_path / filename
            if file_path.exists():
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=f"{exp_hf_prefix}/{filename}",
                        path_or_fileobj=str(file_path),
                    )
                )

    # Upload generation files too
    for gen_file in experiment["generation_files"]:
        gen_path = local_path / gen_file
        if gen_path.exists():
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"{exp_hf_prefix}/{gen_file}",
                    path_or_fileobj=str(gen_path),
                )
            )

    # Write manifest to a temp file for the commit (NOT to the local experiment
    # dir yet — only write there after HF commit succeeds, to prevent manifest
    # corruption if the upload fails).
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        dir=local_path.parent,
    ) as tmp_manifest:
        json.dump(manifest.to_dict(), tmp_manifest, indent=2)
    operations.append(
        CommitOperationAdd(
            path_in_repo=manifest_hf_path,
            path_or_fileobj=tmp_manifest.name,
        )
    )

    if dry_run:
        console.print(f"  [yellow]DRY RUN[/yellow] would upload {len(operations)} files")
        for op in operations:
            console.print(f"    {op.path_in_repo}")
        Path(tmp_manifest.name).unlink(missing_ok=True)
        return True

    # 4. Commit to HF, then save manifest locally only on success
    n_new = len(new_runs)
    commit_msg = f"sync {exp_key}: {n_new} new run(s)"
    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=commit_msg,
        )
    except Exception:
        # Upload failed — do NOT save manifest locally (would mark runs as
        # uploaded when they weren't, causing them to be silently skipped
        # on the next sync).
        Path(tmp_manifest.name).unlink(missing_ok=True)
        raise
    # HF commit succeeded — now safe to save manifest locally
    local_manifest_path = local_path / "manifest.json"
    manifest.save(local_manifest_path)
    Path(tmp_manifest.name).unlink(missing_ok=True)
    console.print(f"  [green]Uploaded[/green] {len(operations)} files ({n_new} new runs)")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Scan local eval_results/ and sync to HuggingFace Hub.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace dataset repo (e.g. aryopg/locos-results)",
    )
    parser.add_argument(
        "--local-dir",
        default="eval_results",
        help="Local results directory (default: eval_results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore existing manifests and rebuild from scratch",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace API token (default: HF_TOKEN env var)",
    )
    parser.add_argument(
        "--hf-prefix",
        default=None,
        help=(
            "Prefix for results in the HF repo. Defaults to the HF_EVAL_PREFIX env var "
            f"if set, else the basename of --local-dir, else '{DEFAULT_HF_EVAL_PREFIX}'."
        ),
    )
    args = parser.parse_args(argv)

    import os

    local_dir = Path(args.local_dir)
    assert local_dir.is_dir(), f"Local directory does not exist: {local_dir}"

    hf_prefix = (
        args.hf_prefix
        or os.environ.get("HF_EVAL_PREFIX")
        or (local_dir.name if local_dir.name else DEFAULT_HF_EVAL_PREFIX)
    )

    # Scan
    console.print(Panel(f"Scanning [bold]{local_dir}[/bold] → HF prefix [bold]{hf_prefix}/[/bold]"))
    experiments = scan_local_experiments(local_dir)

    if not experiments:
        console.print("[yellow]No experiments found.[/yellow]")
        return

    # Summary table
    table = Table(title="Experiments found")
    table.add_column("Experiment key", style="cyan")
    table.add_column("Runs", justify="right")
    table.add_column("Generation files", justify="right")
    for exp in experiments:
        table.add_row(
            exp["experiment_key"],
            str(len(exp["runs"])),
            str(len(exp["generation_files"])),
        )
    console.print(table)

    # Ensure repo exists (private by default)
    if not args.dry_run:
        from huggingface_hub import HfApi

        api = HfApi(token=args.token)
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=True, exist_ok=True)

    # Sync each experiment
    console.print()
    uploaded = 0
    for exp in experiments:
        console.print(f"[bold]{exp['experiment_key']}[/bold]")
        try:
            did_upload = sync_experiment(
                exp,
                repo_id=args.repo_id,
                hf_prefix=hf_prefix,
                dry_run=args.dry_run,
                rebuild=args.rebuild,
                token=args.token,
            )
            if did_upload:
                uploaded += 1
            else:
                console.print("  [dim]Already up to date[/dim]")
        except Exception as e:
            console.print(f"  [red]Error:[/red] {e}")

    console.print()
    if args.dry_run:
        console.print(f"[yellow]Dry run complete.[/yellow] {uploaded}/{len(experiments)} experiments have new data.")
    else:
        console.print(f"[green]Done.[/green] Uploaded {uploaded}/{len(experiments)} experiments.")


if __name__ == "__main__":
    main()
