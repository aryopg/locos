"""Per-model YAML configuration for sampling and hardware defaults.

Lookup order (first match wins):
    1. ``configs/{ModelName}.yaml``   (e.g. ``Meta-Llama-3-8B-Instruct.yaml``)
    2. ``configs/_default.yaml``

CLI arguments always take priority over YAML values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

# Hardcoded fallback defaults — used when neither YAML nor CLI provides a value.
DEFAULTS: dict[str, Any] = {
    "max_tokens": 8192,
    "temperature": 0.0,
    "sampling_top_p": 1.0,
    "sampling_top_k": -1,
    "max_model_len": None,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.5,
}

# Keys that the YAML config files are allowed to set.
CONFIGURABLE_KEYS = frozenset(DEFAULTS.keys())


def _model_short_name(model: str) -> str:
    """Extract the model name after the provider slash.

    ``"meta-llama/Meta-Llama-3-8B-Instruct"`` → ``"Meta-Llama-3-8B-Instruct"``
    """
    return model.split("/")[-1]


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_model_config(model: str, extra_config_path: str | None = None) -> dict[str, Any]:
    """Load per-model YAML config, falling back to ``_default.yaml``.

    Args:
        model: Full HuggingFace model name (e.g. ``"meta-llama/Meta-Llama-3-8B-Instruct"``).
        extra_config_path: Optional explicit path to a YAML config file.
            If given, this takes priority over automatic lookup.

    Returns:
        Dict of resolved defaults (hardcoded < _default.yaml < model.yaml < extra_config_path).
        Only keys in :data:`CONFIGURABLE_KEYS` are included.
    """
    merged: dict[str, Any] = dict(DEFAULTS)

    def _apply(source: dict[str, Any], *, allow_null: bool = False) -> None:
        """Merge values from *source* into *merged*.

        When *allow_null* is False (used for ``_default.yaml``), ``null``
        values are skipped.  When True (model-specific configs), ``null``
        explicitly resets the key to its hardcoded disabled default — e.g.
        ``sampling_top_p: null`` → ``1.0`` (disabled).
        """
        for k, v in source.items():
            if k not in CONFIGURABLE_KEYS:
                continue
            if v is None:
                if allow_null:
                    # Reset to hardcoded default (the "disabled" sentinel)
                    merged[k] = DEFAULTS[k]
            else:
                merged[k] = v

    # Layer 1: _default.yaml — null is ignored (no meaningful reset target)
    default_path = _CONFIGS_DIR / "_default.yaml"
    if default_path.exists():
        _apply(_load_yaml(default_path), allow_null=False)

    # Layer 2: model-specific YAML — null means "disable this parameter"
    model_path = _CONFIGS_DIR / f"{_model_short_name(model)}.yaml"
    if model_path.exists():
        _apply(_load_yaml(model_path), allow_null=True)

    # Layer 3: explicit --model-config path — same semantics as model YAML
    if extra_config_path is not None:
        p = Path(extra_config_path)
        assert p.exists(), f"Model config not found: {p}"
        _apply(_load_yaml(p), allow_null=True)

    return merged
