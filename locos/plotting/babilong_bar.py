#!/usr/bin/env python3
"""BabiLong accuracy bar charts across decoding variants.

Two kinds of figures:
- Per-model: one SVG per model under ``<out_dir>/<ModelPretty>.svg``, with
  one subplot per metric (``accuracy``, ``tag_present``). x-axis groups by
  babi subtask (qa2 / qa3); hue is decoding variant.
- Overall: one cross-model SVG (``Overall.svg``) with x = model, hue = variant,
  one subplot per metric.

``tag_present`` is the rate at which the model emitted the required answer
tag — a low value means the accuracy score is computed over malformed
outputs, so it's worth surfacing alongside accuracy.

Usage:
    python locos/plotting/babilong_bar.py \\
        --results-root ../locos-results/downstream_results \\
        --out-dir figures/babilong_bar
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
from locos.plotting._downstream_common import MODEL_ORDER
from locos.plotting._paths import default_downstream_results_root

console = Console()


# Task subdirectories: one per (subset, length).
TASK_DIRS = ["babilong_qa2_0k", "babilong_qa3_0k"]

# Metrics: (score_key, axis_label). First entry is treated as primary.
METRICS = [
    ("accuracy", "Accuracy"),
    ("tag_present", "Tag-present rate"),
]

TASK_NAME = "BABILong (qa2, qa3)"


def domain_for_row(row: dict, task_dir_name: str) -> str | None:
    """Group by subset+length so qa2 vs qa3 are different bars."""
    meta = row.get("metadata", {}) or {}
    subset = meta.get("subset")
    split = meta.get("split")
    if subset is None or split is None:
        # Fall back to the task dir name (already encodes both).
        return task_dir_name.removeprefix("babilong_") or None
    return f"{subset}_{split}"


# Pretty labels for the x-axis.
DOMAIN_SHORT = {
    "qa2_0k": "qa2 (0k)",
    "qa3_0k": "qa3 (0k)",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-model bar charts of BabiLong accuracy & tag-present rate by subtask",
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
        default=Path("figures/babilong_bar"),
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
    )


if __name__ == "__main__":
    main()
