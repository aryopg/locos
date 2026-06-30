#!/usr/bin/env python3
"""Shared discovery/registry helpers for downstream-eval bar charts.

Provides the decoding-variant registry, model ordering, and per-seed results
loading shared by ``babilong_bar.py``, ``musique_bar.py``, and
``_downstream_bar_common.py``. Variant directories follow the
``<family>_s<seed>`` convention (e.g. ``greedy_s1``,
``ablation_logitcontrib_nolima_s3``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Variants to render (in legend order). Each maps a *family* prefix to a
# pretty label and a base color from the tab10 palette.
_TAB10 = sns.color_palette("tab10")
VARIANT_FAMILIES: list[tuple[str, str, tuple]] = [
    ("greedy", "Greedy", (0.45, 0.45, 0.45)),  # baseline grey
    ("ablation_logitcontrib_nolima", "Ablate LOCOS (NoLiMa)", _TAB10[0]),  # blue
    ("ablation_wu_niah", "Ablate Wu (NIAH)", _TAB10[2]),  # green
    ("ablation_random", "Ablate random", _TAB10[3]),  # red
]

# Model display order across the bar/grid layouts (left-right, top-bottom).
MODEL_ORDER = [
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "Olmo-3.1-32B-Instruct",
]


# ---------------------------------------------------------------------------
# Discovery & loading
# ---------------------------------------------------------------------------

# Variant directory format: <family>_s<seed> (greedy_s1, ablation_wu_niah_s2,
# ablation_random_s42_n50_s1, ablation_logitcontrib_nolima_s3).
_VARIANT_RE = re.compile(r"^(?P<family>.+)_s(?P<seed>\d+)$")


def parse_variant_dir(name: str) -> tuple[str, int] | None:
    """Return (family, seed) for a variant directory, or None if it doesn't parse."""
    m = _VARIANT_RE.match(name)
    if not m:
        return None
    return m.group("family"), int(m.group("seed"))


def family_to_canonical(family: str) -> str | None:
    """Map a parsed family string to one of the canonical VARIANT_FAMILIES keys."""
    for key, _, _ in VARIANT_FAMILIES:
        if family == key or family.startswith(key + "_"):
            return key
    return None


def load_results_jsonl(variant_dir: Path) -> list[dict] | None:
    """Load the latest results_*.jsonl in a variant directory.

    Returns None if no scored results are present.
    """
    candidates = sorted(variant_dir.glob("results_*.jsonl"))
    candidates = [p for p in candidates if not p.name.endswith("_config.json")]
    if not candidates:
        return None
    # Use the most recent (lexicographic timestamp == chronological).
    chosen = candidates[-1]
    rows: list[dict] = []
    with open(chosen) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
