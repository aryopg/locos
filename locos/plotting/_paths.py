"""Shared filesystem defaults for plotting entrypoints."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def default_downstream_results_root() -> Path:
    """Return the default downstream-results directory.

    Override with ``LOCOS_DOWNSTREAM_DIR`` when results live elsewhere.
    """
    return Path(os.environ.get("LOCOS_DOWNSTREAM_DIR", REPO_ROOT.parent / "locos-results" / "downstream_results"))
