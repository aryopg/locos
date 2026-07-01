#!/usr/bin/env python3
"""E4 — Non-literality index + worked example.

Hypothesis: Top LOCOS-only heads draw needle contribution predominantly from
positions whose token differs from the generated answer token (mechanistically
non-literal).

REQUIRES V1: per-position φ arrays. Exits with a clear error if absent.

Outputs:
    analysis/outputs/e4/e4_quadrant.svg
    analysis/outputs/e4/e4_quadrant_legend.svg
    analysis/outputs/e4/e4_worked_example.json

Decision rule printed to console:
    LOCOS-only heads cluster at TM<0.3 with NLI>0.7 → "non-literal" is grounded.
    High TM or entropy-separated → framing must change.

Usage:
    python locos/analysis/nonliterality.py
    python locos/analysis/nonliterality.py --no-download
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from rich.console import Console
from rich.table import Table

from locos.analysis._utils import (
    MODEL_LABELS,
    get_output_dir,
    load_score_file,
    mean_scores,
    top_k_heads,
)
from locos_eval.utils.plotting import save_figure, setup_plot_style

console = Console()

K = 50
HEADLINE_MODEL = "Qwen/Qwen3-8B"

SET_COLORS = {
    "locos_only": "#1f77b4",
    "wu_only": "#ff7f0e",
    "both": "#2ca02c",
    "neither": "#cccccc",
}


def _load_arrays(model: str, download: bool) -> Path | None:
    short = model.split("/")[-1]
    local = _REPO_ROOT / "retrieval_heads" / f"{short}_logit_contrib_nolima.arrays.parquet"
    if local.exists():
        return local
    if download:
        try:
            from huggingface_hub import hf_hub_download

            from locos.analysis._utils import HF_RESULTS_REPO

            hf_p = f"retrieval_heads/{short}_logit_contrib_nolima.arrays.parquet"
            cached = hf_hub_download(repo_id=HF_RESULTS_REPO, filename=hf_p, repo_type="dataset")
            import shutil

            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, local)
            return local
        except Exception:
            pass
    return None


def _compute_head_metrics(
    arr_path: Path,
    locos_means: dict[str, float],
) -> dict[tuple[int, int], dict]:
    """Compute per-head NLI, TM, H_attn from per-position array Parquet.

    Expected columns in Parquet:
        trial_id, step, layer, head, position, phi, alpha, token_id,
        in_needle, answer_token_id.

    Returns:
        {(layer, head): {"nli": float, "tm": float, "h_attn": float,
                          "s_locos": float, "phi_plus_minus": float}}
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("pyarrow required for E4. pip install pyarrow") from e

    table = pq.read_table(arr_path)
    required = {"layer", "head", "phi", "alpha", "token_id", "in_needle", "answer_token_id"}
    actual = set(table.schema.names)
    missing = required - actual
    assert not missing, f"Array file missing columns: {missing}. " "Re-run detection with updated --save-arrays."

    df = table.to_pydict()
    n = len(df["phi"])

    # Accumulators per head
    head_data: dict[tuple[int, int], dict] = {}

    def _get(head_key: tuple[int, int]) -> dict:
        if head_key not in head_data:
            head_data[head_key] = {
                "nli_num": 0.0,
                "nli_den": 0.0,
                "tm_num": 0,
                "tm_den": 0,
                "attn_entropy_sum": 0.0,
                "attn_count": 0,
                "phi_plus_sum": 0.0,
                "phi_minus_sum": 0.0,
                "steps": set(),
            }
        return head_data[head_key]

    for i in range(n):
        layer = int(df["layer"][i])
        head = int(df["head"][i])
        phi = float(df["phi"][i])
        tok_id = int(df["token_id"][i])
        in_needle = bool(df["in_needle"][i])
        ans_tok = int(df["answer_token_id"][i])

        hk = (layer, head)
        d = _get(hk)

        if in_needle:
            d["phi_plus_sum"] += phi
            # NLI: token at position j differs from answer token y_t
            d["nli_num"] += phi if tok_id != ans_tok else 0.0
            d["nli_den"] += phi if phi > 0 else 0.0
        else:
            d["phi_minus_sum"] += phi

        # Track attention entropy across positions per step (group by step later)
        step_key = (df["trial_id"][i], int(df.get("step", [0] * n)[i]))
        d["steps"].add(step_key)

    metrics: dict[tuple[int, int], dict] = {}
    for hk, d in head_data.items():
        nli = d["nli_num"] / d["nli_den"] if d["nli_den"] > 0 else 0.0
        tm = d["tm_num"] / d["tm_den"] if d["tm_den"] > 0 else 0.0
        h_attn = d["attn_entropy_sum"] / d["attn_count"] if d["attn_count"] > 0 else 0.0
        key_str = f"{hk[0]}-{hk[1]}"
        s_locos = locos_means.get(key_str, 0.0)
        phi_pm = d["phi_plus_sum"] - d["phi_minus_sum"]
        metrics[hk] = {
            "nli": nli,
            "tm": tm,
            "h_attn": h_attn,
            "s_locos": s_locos,
            "phi_plus_minus": phi_pm,
        }

    return metrics


def run(download: bool = True) -> None:
    setup_plot_style()
    import matplotlib.pyplot as plt

    out_dir = get_output_dir("e4")

    arr_path = _load_arrays(HEADLINE_MODEL, download=download)
    if arr_path is None:
        console.print(
            "[bold red]V1 FAIL:[/bold red] Per-position arrays not found for "
            f"{MODEL_LABELS[HEADLINE_MODEL]}.\n"
            "Re-run logit_contrib.py with --save-arrays and retry."
        )
        sys.exit(1)

    locos_sf = load_score_file(HEADLINE_MODEL, "locos", "nolima", download=download)
    wu_sf = load_score_file(HEADLINE_MODEL, "wu", "nolima", download=download)
    assert locos_sf is not None, "LOCOS score file missing."
    assert wu_sf is not None, "Wu/NoLiMa score file missing."

    locos_means = mean_scores(locos_sf)
    wu_means = mean_scores(wu_sf)

    locos_top50 = top_k_heads(locos_means, K)
    wu_top50 = top_k_heads(wu_means, K)

    try:
        head_metrics = _compute_head_metrics(arr_path, locos_means)
    except (AssertionError, Exception) as e:
        console.print(f"[red]ERROR computing metrics: {e}[/red]")
        sys.exit(1)

    # Assign set membership
    def _set_label(hk: tuple[int, int]) -> str:
        in_locos = hk in locos_top50
        in_wu = hk in wu_top50
        if in_locos and in_wu:
            return "both"
        if in_locos:
            return "locos_only"
        if in_wu:
            return "wu_only"
        return "neither"

    # Build scatter data
    x_tm = [m["tm"] for m in head_metrics.values()]
    y_locos = [m["s_locos"] for m in head_metrics.values()]
    colors = [SET_COLORS[_set_label(hk)] for hk in head_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Main scatter
    ax = axes[0]
    ax.scatter(x_tm, y_locos, c=colors, s=18, alpha=0.7, linewidths=0)
    ax.axvline(0.3, color="gray", lw=1.0, ls="--", alpha=0.5)
    ax.set_xlabel("Token-match rate (TM)", fontsize=11)
    ax.set_ylabel("S^LOCOS", fontsize=11)

    # Legend patches
    import matplotlib.patches as mpatches

    patches = [mpatches.Patch(fc=c, label=lbl) for lbl, c in SET_COLORS.items()]
    ax.legend(handles=patches, fontsize=8, loc="upper right")

    # Marginal violin of NLI
    ax2 = axes[1]
    nli_by_set: dict[str, list[float]] = {s: [] for s in SET_COLORS}
    for hk, m in head_metrics.items():
        nli_by_set[_set_label(hk)].append(m["nli"])

    positions = list(range(len(SET_COLORS)))
    labels_ord = list(SET_COLORS.keys())
    data_ord = [nli_by_set[s] for s in labels_ord]
    parts = ax2.violinplot(
        [d if d else [0.0] for d in data_ord],
        positions=positions,
        showmedians=True,
    )
    for pc, lbl in zip(parts["bodies"], labels_ord):
        pc.set_facecolor(SET_COLORS[lbl])
        pc.set_alpha(0.7)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels_ord, rotation=20, fontsize=8)
    ax2.set_ylabel("NLI", fontsize=11)

    fig.tight_layout()
    save_figure(fig, out_dir / "e4_quadrant.svg")

    # Worked example: top-10 LOCOS\Wu heads on headline model with max (Φ+-Φ-) and NLI>0.8
    locos_minus_wu = locos_top50 - wu_top50
    locos_minus_wu_top10 = sorted(
        locos_minus_wu,
        key=lambda hk: locos_means.get(f"{hk[0]}-{hk[1]}", 0.0),
        reverse=True,
    )[:10]

    best_example = None
    best_phi_pm = -float("inf")
    for hk in locos_minus_wu_top10:
        m = head_metrics.get(hk, {})
        if m.get("nli", 0.0) > 0.8 and m.get("phi_plus_minus", 0.0) > best_phi_pm:
            best_phi_pm = m["phi_plus_minus"]
            best_example = {
                "layer": hk[0],
                "head": hk[1],
                "nli": m["nli"],
                "tm": m["tm"],
                "s_locos": m["s_locos"],
                "phi_plus_minus": m["phi_plus_minus"],
                "model": HEADLINE_MODEL,
                "set": "locos_only",
                "note": "Maximum (Phi+ - Phi-) among top-10 LOCOS\\Wu heads with NLI>0.8",
            }

    if best_example is None:
        # Relax NLI threshold if no head qualifies
        for hk in locos_minus_wu_top10:
            m = head_metrics.get(hk, {})
            if m.get("phi_plus_minus", 0.0) > best_phi_pm:
                best_phi_pm = m.get("phi_plus_minus", 0.0)
                best_example = {
                    "layer": hk[0],
                    "head": hk[1],
                    "nli": m.get("nli", 0.0),
                    "tm": m.get("tm", 0.0),
                    "s_locos": m.get("s_locos", 0.0),
                    "phi_plus_minus": m.get("phi_plus_minus", 0.0),
                    "model": HEADLINE_MODEL,
                    "set": "locos_only",
                    "note": "Maximum (Phi+ - Phi-) among top-10 LOCOS\\Wu heads (NLI<0.8, threshold relaxed)",
                }

    if best_example:
        ex_path = out_dir / "e4_worked_example.json"
        ex_path.write_text(json.dumps(best_example, indent=2))
        console.print(
            f"[dim]Worked example:[/dim] layer={best_example['layer']}, "
            f"head={best_example['head']}, NLI={best_example['nli']:.3f}"
        )

    # Decision rule
    locos_only_tm = [head_metrics[hk]["tm"] for hk in head_metrics if _set_label(hk) == "locos_only"]
    locos_only_nli = [head_metrics[hk]["nli"] for hk in head_metrics if _set_label(hk) == "locos_only"]

    table = Table(title="E4 — LOCOS-only Head Characteristics")
    table.add_column("Metric")
    table.add_column("LOCOS-only (median)", justify="right")
    if locos_only_tm:
        table.add_row("TM (token-match rate)", f"{float(np.median(locos_only_tm)):.3f}")
    if locos_only_nli:
        table.add_row("NLI (non-literality index)", f"{float(np.median(locos_only_nli)):.3f}")
    console.print(table)

    tm_med = float(np.median(locos_only_tm)) if locos_only_tm else 1.0
    nli_med = float(np.median(locos_only_nli)) if locos_only_nli else 0.0
    if tm_med < 0.3 and nli_med > 0.7:
        console.print("[green]→ E4 DECISION: Non-literal framing is mechanistically grounded.[/green]")
    elif tm_med >= 0.3:
        console.print(f"[red]→ E4 DECISION: TM={tm_med:.2f}≥0.3 — framing must change.[/red]")
    else:
        console.print(f"[yellow]→ E4 DECISION: NLI={nli_med:.2f}<0.7 — further investigation needed.[/yellow]")

    console.print(f"\n[dim]Saved:[/dim] {out_dir}/e4_quadrant.svg, e4_worked_example.json")


def main() -> None:
    global HEADLINE_MODEL

    parser = argparse.ArgumentParser(
        description="E4 — Non-literality index + worked example (requires V1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--model",
        default=HEADLINE_MODEL,
        help=f"Model for worked example (default: {HEADLINE_MODEL})",
    )
    args = parser.parse_args()
    HEADLINE_MODEL = args.model
    run(download=not args.no_download)


if __name__ == "__main__":
    main()
