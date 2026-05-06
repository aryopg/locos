#!/usr/bin/env python3
"""Check whether an experiment is already complete on HuggingFace.

Exit codes:
  0 = complete (skip)
  1 = needs running (not found, incomplete, or --force)
  2 = check itself failed (HF unreachable, bad config, etc.)

IMPORTANT: Job scripts MUST distinguish exit 1 (run) from exit 2 (error).
Exit 2 means the check could not be performed — proceeding blindly risks
duplicate GPU work or masking configuration bugs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Default prefix for eval results in HF repo (must match sync_results.py).
# Override with --hf-prefix or the HF_EVAL_PREFIX env var.
DEFAULT_HF_EVAL_PREFIX = "eval_results"


def _download_manifest(repo_id: str, experiment_key: str, hf_prefix: str, repo_type: str = "dataset") -> Path | None:
    """Download per-experiment manifest.json from HF.

    Returns local path on success, None if the file doesn't exist on HF.
    Raises on network/auth errors (callers must handle).
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{hf_prefix}/{experiment_key}/manifest.json",
            repo_type=repo_type,
        )
        return Path(path)
    except EntryNotFoundError:
        # Manifest file doesn't exist — experiment genuinely not done
        return None
    except RepositoryNotFoundError as e:
        # Repo not found usually means auth failure on a private repo.
        # Raise so callers treat this as a check error (exit 2), not "run it".
        raise RuntimeError(f"Repository {repo_id!r} not found — is HF_TOKEN set for this private repo?") from e
    # All other exceptions (network, auth, etc.) propagate to caller


def is_experiment_complete(
    repo_id: str,
    experiment_key: str,
    hf_prefix: str = DEFAULT_HF_EVAL_PREFIX,
    min_samples: int | None = None,
    repo_type: str = "dataset",
) -> bool:
    """Check if an experiment has a complete manifest on HF.

    Raises on network/auth/JSON errors — callers must handle.
    """
    manifest_path = _download_manifest(repo_id, experiment_key, hf_prefix, repo_type)
    if manifest_path is None:
        return False
    with open(manifest_path) as f:
        data = json.load(f)
    if data.get("status") != "complete":
        return False
    if min_samples is not None:
        max_samples = max((r.get("n_samples", 0) for r in data.get("runs", [])), default=0)
        if max_samples < min_samples:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Check if experiment is complete on HF.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--decoding", required=True)
    parser.add_argument("--heads", default=None)
    parser.add_argument("--heads-label", default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--sampling-seed", default=None)
    parser.add_argument(
        "--ablation-mode",
        default="zero",
        choices=["zero", "mean"],
        help="Ablation replacement mode (only meaningful with --decoding ablation)",
    )
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Always exit 1 (run)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument(
        "--hf-prefix",
        default=None,
        help=f"Prefix for eval results in HF repo (default: ${{HF_EVAL_PREFIX:-{DEFAULT_HF_EVAL_PREFIX}}})",
    )
    args = parser.parse_args()

    import os

    hf_prefix = args.hf_prefix or os.environ.get("HF_EVAL_PREFIX") or DEFAULT_HF_EVAL_PREFIX

    if args.force:
        if not args.quiet:
            print("FORCE: skipping check")
        sys.exit(1)

    try:
        from locos_eval.evals.experiment_key import ExperimentKey

        # Normalize empty strings to None (shell scripts pass "" for unset vars)
        heads = args.heads if args.heads else None
        heads_label = args.heads_label if args.heads_label else None
        # sampling_seed comes as string from shell; convert to int or None
        sampling_seed_raw = args.sampling_seed
        sampling_seed = int(sampling_seed_raw) if sampling_seed_raw else None
        ek = ExperimentKey(
            task=args.task,
            model=args.model,
            decoding=args.decoding,
            heads_path=heads,
            heads_label=heads_label,
            num_heads=args.num_heads,
            random_seed=args.random_seed,
            sampling_seed=sampling_seed,
            ablation_mode=args.ablation_mode,
        )

        complete = is_experiment_complete(
            repo_id=args.repo_id,
            experiment_key=ek.key,
            hf_prefix=hf_prefix,
            min_samples=args.min_samples,
            repo_type=args.repo_type,
        )
    except Exception as exc:
        # Exit 2 = check failed (do NOT treat as "needs running")
        print(f"CHECK_ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    if complete:
        if not args.quiet:
            print(f"SKIP: {ek.key} already complete")
        sys.exit(0)
    else:
        if not args.quiet:
            print(f"RUN: {ek.key} not found or incomplete")
        sys.exit(1)


if __name__ == "__main__":
    main()
