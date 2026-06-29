#!/usr/bin/env python3
"""Data inventory check for E1–E9 experiments.

Verifies V1–V6 prerequisites and reports what's available before running
any analysis. Must be run first — E3/E4 are gated on V1.

Usage:
    python locos/analysis/inventory.py
    python locos/analysis/inventory.py --no-download
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from locos.analysis._utils import (
    _DOWNSTREAM_DIRS,
    ALL_MODELS,
    E6_AFFECTED_MODELS,
    HF_DOWNSTREAM_PREFIX,
    HF_RESULTS_REPO,
    MODEL_SHORT,
    get_output_dir,
    load_score_file,
)

console = Console()


# ---------------------------------------------------------------------------
# V1 — per-position φ/α arrays
# ---------------------------------------------------------------------------


def _check_v1(download: bool) -> tuple[bool, str]:
    """V1: per-position φ, α arrays exist (not only pooled scores)."""
    # logit_contrib.py saves only pooled per-trial scores, never per-position.
    # A re-run with --save-arrays (to be added) would produce .arrays.parquet files.
    arr_files = list((_REPO_ROOT / "retrieval_heads").glob("*_logit_contrib*.arrays.parquet"))

    if not arr_files and download:
        # Try HF
        try:
            from huggingface_hub import list_repo_files

            hf_files = list(list_repo_files(HF_RESULTS_REPO, repo_type="dataset"))
            arr_files_hf = [f for f in hf_files if f.endswith(".arrays.parquet")]
        except Exception:
            arr_files_hf = []
        if arr_files_hf:
            return True, f"{len(arr_files_hf)} array files on HF (not downloaded)"

    if arr_files:
        return True, f"{len(arr_files)} .arrays.parquet files found locally"

    return (
        False,
        "No per-position array files found. " "Re-run logit_contrib.py with --save-arrays (to be added) then retry.",
    )


# ---------------------------------------------------------------------------
# V2 — φ/α logged at non-answer steps
# ---------------------------------------------------------------------------


def _check_v2(download: bool) -> tuple[bool, str]:
    """V2: φ/α logged at non-answer decode steps (needed for E7 cell C)."""
    na_files = list((_REPO_ROOT / "retrieval_heads").glob("*_logit_contrib*.non_answer.parquet"))
    if na_files:
        return True, f"{len(na_files)} non-answer array files found"
    return False, "Non-answer step arrays not found (E7 cell C needs these; E1-E6 do not)."


# ---------------------------------------------------------------------------
# V3 — tokenized context metadata (needle/question spans + gold token IDs)
# ---------------------------------------------------------------------------


def _check_v3(download: bool) -> tuple[bool, str]:
    """V3: tokenized contexts with needle/question spans and gold answer token IDs."""
    meta_files = list((_REPO_ROOT / "retrieval_heads").glob("*_logit_contrib*.trial_meta.parquet"))
    if meta_files:
        return True, f"{len(meta_files)} trial_meta files found"
    return False, "Trial metadata files not found (needed for E3/E4 position taxonomy)."


# ---------------------------------------------------------------------------
# V4 — BABILong generations (4 conditions × 3 models)
# ---------------------------------------------------------------------------


def _check_v4(download: bool) -> tuple[bool, str]:
    """V4: BABILong generations for {baseline, wu_ablated, locos_ablated, random_ablated} × 3 models."""
    required_variants = [
        "greedy",
        "ablation_wu_niah",
        "ablation_wu_nolima",
        "ablation_logitcontrib_nolima",
        "ablation_random_s42_n50",
    ]
    task_subsets = ["babilong_qa2_0k", "babilong_qa3_0k"]

    found: list[str] = []
    missing: list[str] = []

    if download:
        try:
            from huggingface_hub import list_repo_files

            hf_files = set(list_repo_files(HF_RESULTS_REPO, repo_type="dataset"))
        except Exception:
            hf_files = set()
    else:
        hf_files = set()

    for model in E6_AFFECTED_MODELS:
        slug = model.replace("/", "_")
        for subset in task_subsets:
            for variant in required_variants:
                # Check all seeded variants (s1, s2, s3)
                seeded = [f"{variant}_s{s}" for s in (1, 2, 3)] + [variant]
                path_found = False
                for v in seeded:
                    for pfx in (
                        f"{HF_DOWNSTREAM_PREFIX}/{subset}/{slug}/{v}/",
                        f"{subset}/{slug}/{v}/",
                    ):
                        if any(f.startswith(pfx) and f.endswith(".jsonl") for f in hf_files):
                            path_found = True
                            break
                    if path_found:
                        break
                    for base in _DOWNSTREAM_DIRS:
                        variant_dir = base / subset / slug / v
                        if variant_dir.is_dir() and (
                            any(variant_dir.glob("results_*.jsonl")) or any(variant_dir.glob("results*.jsonl"))
                        ):
                            path_found = True
                            break
                    if path_found:
                        break
                if path_found:
                    found.append(f"{MODEL_SHORT[model]}/{subset}/{variant}")
                else:
                    missing.append(f"{MODEL_SHORT[model]}/{subset}/{variant}")

    if not missing:
        return True, f"All {len(found)} BABILong generation files present"
    if found:
        return (
            False,
            f"{len(found)}/{len(found)+len(missing)} present. "
            f"Missing: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}",
        )
    return False, f"No BABILong generation files found. Missing {len(missing)} files."


# ---------------------------------------------------------------------------
# V5 — head score files for all 6 models (4 methods)
# ---------------------------------------------------------------------------


def _check_v5(download: bool) -> tuple[bool, str]:
    """V5: head score files for LOCOS, α-spatial, Wu/NIAH, Wu/NoLiMa for all 6 models."""
    checks = [
        ("locos", "nolima"),
        ("alpha_spatial", "nolima"),
        ("wu", "niah"),
        ("wu", "nolima"),
    ]
    found = []
    missing = []
    for model in ALL_MODELS:
        for method, dataset in checks:
            sf = load_score_file(model, method, dataset, download=download)
            key = f"{MODEL_SHORT[model]}/{method}/{dataset}"
            if sf is not None:
                n_heads = len(sf.scores)
                n_trials = len(next(iter(sf.scores.values()), []))
                found.append(f"{key} ({n_heads}h, {n_trials}t)")
            else:
                missing.append(key)

    if not missing:
        return True, f"All {len(found)} score files present"
    return (
        len(missing) == 0,
        f"{len(found)}/{len(found)+len(missing)} score files present. "
        f"Missing: {', '.join(missing[:6])}{'...' if len(missing) > 6 else ''}",
    )


# ---------------------------------------------------------------------------
# V6 — consensus head set sizes (Wu ∩ LOCOS at k=50)
# ---------------------------------------------------------------------------


def _check_v6(download: bool) -> tuple[bool, str]:
    """V6: per-model consensus (Wu ∩ LOCOS) set sizes at k=50 for size-matching in E8."""
    from locos.analysis._utils import mean_scores, top_k_heads

    sizes = {}
    for model in ALL_MODELS:
        locos = load_score_file(model, "locos", "nolima", download=download)
        wu = load_score_file(model, "wu", "nolima", download=download)
        if locos is None or wu is None:
            sizes[MODEL_SHORT[model]] = None
        else:
            locos_top50 = top_k_heads(mean_scores(locos), 50)
            wu_top50 = top_k_heads(mean_scores(wu), 50)
            sizes[MODEL_SHORT[model]] = len(locos_top50 & wu_top50)

    missing = [k for k, v in sizes.items() if v is None]
    if not missing:
        size_str = ", ".join(f"{k}={v}" for k, v in sizes.items())
        return True, f"Consensus sizes: {size_str}"
    return False, f"Cannot compute consensus for: {', '.join(missing)} (score files missing)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CHECKS = [
    ("V1", "Per-position φ/α arrays (E3/E4)", _check_v1),
    ("V2", "Non-answer step arrays (E7 cell C)", _check_v2),
    ("V3", "Trial metadata (needle/question spans)", _check_v3),
    ("V4", "BABILong generations (E6)", _check_v4),
    ("V5", "Head score files (E1/E2/E5)", _check_v5),
    ("V6", "Consensus sizes (E8)", _check_v6),
]

E1_E6_BLOCKERS = {"V1": "E3, E4", "V3": "E3, E4", "V4": "E6"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data inventory check for E1–E9 experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip HuggingFace downloads; check local files only.",
    )
    args = parser.parse_args()
    download = not args.no_download

    console.print(Panel("[bold]E1–E9 Data Inventory Check[/bold]", expand=False))
    if download:
        console.print("[dim]Will attempt HF downloads for missing files.[/dim]")
    else:
        console.print("[dim]Local files only (--no-download).[/dim]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("ID", width=4)
    table.add_column("Description", width=38)
    table.add_column("Status", width=6)
    table.add_column("Detail")

    results: dict[str, bool] = {}
    for vid, desc, fn in CHECKS:
        ok, detail = fn(download)
        results[vid] = ok
        status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        table.add_row(vid, desc, status, detail)

    console.print(table)

    # Blockers for E1–E6
    blockers_hit = []
    for vid, experiments in E1_E6_BLOCKERS.items():
        if not results.get(vid, False):
            blockers_hit.append((vid, experiments))

    out_dir = get_output_dir(".")
    inventory = {v: results[v] for v, _, _ in CHECKS}
    (out_dir.parent / "inventory.json").write_text(json.dumps(inventory, indent=2))
    console.print("\n[dim]Written: analysis/outputs/inventory.json[/dim]")

    if blockers_hit:
        console.print("\n[bold red]BLOCKERS for E1–E6:[/bold red]")
        for vid, exps in blockers_hit:
            console.print(f"  [red]{vid} FAIL[/red] → blocks {exps}")
        console.print("\n[yellow]E1, E2, E5, E6 can still run (V1/V3 only block E3/E4).[/yellow]")
        if not results["V1"] or not results["V3"]:
            console.print(
                "[yellow]E3/E4 need detection re-run with --save-arrays flag "
                "(to be added to logit_contrib.py).[/yellow]"
            )
    else:
        console.print("\n[bold green]All checks passed — all E1–E6 scripts can run.[/bold green]")

    passes = sum(results.values())
    console.print(f"\n{passes}/{len(results)} checks passed.")
    sys.exit(0 if passes == len(results) else 1)


if __name__ == "__main__":
    main()
