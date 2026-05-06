#!/usr/bin/env python3
"""Upload experiment results to HuggingFace Hub.

Examples::

    # Upload Inspect AI eval logs
    python scripts/upload_results.py ./logs/2024-01-15T10_30_00_medrag \\
        --repo-id aryopg/decore-results

    # Upload retrieval heads with a custom subfolder
    python scripts/upload_results.py ./retrieval_heads \\
        --repo-id aryopg/decore-results \\
        --path-in-repo retrieval_heads/llama3-8b
"""

from __future__ import annotations

import argparse

from locos_eval.utils.hf_upload import upload_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a results directory to HuggingFace Hub.",
    )
    parser.add_argument("local_dir", help="Local directory to upload.")
    parser.add_argument("--repo-id", required=True, help="HF repo id (e.g. aryopg/decore-results).")
    parser.add_argument("--path-in-repo", default=None, help="Subfolder in the repo (default: directory name).")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    parser.add_argument("--commit-message", default=None, help="Custom commit message.")
    parser.add_argument("--token", default=None, help="HF API token (default: HF_TOKEN env var).")
    args = parser.parse_args()

    upload_results(
        local_dir=args.local_dir,
        repo_id=args.repo_id,
        path_in_repo=args.path_in_repo,
        repo_type=args.repo_type,
        commit_message=args.commit_message,
        token=args.token,
    )


if __name__ == "__main__":
    main()
