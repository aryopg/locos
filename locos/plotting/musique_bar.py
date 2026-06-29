#!/usr/bin/env python3
"""MuSiQue accuracy bar charts across decoding variants.

Two kinds of figures:
- Per-model: one SVG per model under ``<out_dir>/<ModelPretty>.svg``, with
  one subplot per metric (``accuracy``, ``tag_present``). x-axis groups
  questions by ``n_hops`` (2/3/4); hue is decoding variant.
- Overall: one cross-model SVG (``Overall.svg``) with x = model, hue = variant,
  one subplot per metric.

``tag_present`` is the rate at which the model emitted the required answer
tag — a low value means the accuracy score is computed over malformed
outputs, so it's worth surfacing alongside accuracy.

Usage:
    python locos/plotting/musique_bar.py \\
        --results-root ../locos-results/downstream_results \\
        --out-dir figures/musique_bar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console

from locos.plotting._downstream_bar_common import (
    render_multi_metric_domain_bars,
)
from locos.plotting._paths import default_downstream_results_root
from locos.plotting.longbench_v2_radar import MODEL_ORDER

console = Console()


TASK_DIRS = ["musique_answerable"]

METRICS = [
    ("accuracy", "Accuracy"),
    ("tag_present", "Tag-present rate"),
]

TASK_NAME = "MuSiQue"


def domain_for_row(row: dict, task_dir_name: str) -> str | None:
    """Group by ``n_hops`` (string-keyed for stable ordering)."""
    n_hops = (row.get("metadata") or {}).get("n_hops")
    if n_hops is None:
        return None
    return f"{int(n_hops)}-hop"


DOMAIN_SHORT = {
    "2-hop": "2-hop",
    "3-hop": "3-hop",
    "4-hop": "4-hop",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-model bar charts of MuSiQue accuracy & tag-present rate by hop count",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_downstream_results_root(),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/musique_bar"),
    )
    args = parser.parse_args()

    render_multi_metric_domain_bars(
        results_root=args.results_root,
        out_dir=args.out_dir,
        task_dirs=TASK_DIRS,
        domain_fn=domain_for_row,
        metrics=METRICS,
        task_name=TASK_NAME,
        model_order=MODEL_ORDER,
        domain_short=DOMAIN_SHORT,
        domain_order_fn=lambda counts: sorted(counts, key=lambda d: int(d.split("-", 1)[0])),
    )


if __name__ == "__main__":
    main()
