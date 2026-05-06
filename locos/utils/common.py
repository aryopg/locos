"""Shared utilities for the retrieval head detection package.

Contains generic helpers used across multiple detection scripts:
formatting, checkpointing, model loading, config extraction, and
score file I/O.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS (or MM:SS for short durations)."""
    if seconds < 0 or not np.isfinite(seconds):
        return "n/a"
    total = round(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Checkpoint / resume (generic, atomic writes)
# ---------------------------------------------------------------------------


def save_checkpoint(data: dict, path: Path) -> None:
    """Save a JSON checkpoint atomically (write to tmp, then rename).

    Callers are responsible for structuring *data* — this function
    only handles serialisation and the atomic file swap.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.rename(path)


def load_checkpoint(path: Path) -> dict | None:
    """Load a JSON checkpoint, returning ``None`` if the file doesn't exist."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
}


def load_model_and_tokenizer(
    model_name: str,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
):
    """Load a HuggingFace causal LM with eager attention (for output_attentions).

    Returns:
        (model, tokenizer) tuple.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="eager",
        trust_remote_code=trust_remote_code,
    ).eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Model config extraction
# ---------------------------------------------------------------------------


def extract_model_config(model) -> tuple[int, int]:
    """Extract (num_layers, num_heads) from a HuggingFace model config.

    Handles different model families (Llama, Qwen, GPT-2, Gemma, VLMs)
    which use different attribute names and may nest config under text_config.

    Returns:
        (num_layers, num_heads) tuple.

    Raises:
        AssertionError: If num_layers or num_heads cannot be determined.
    """
    config = model.config
    text_config = getattr(config, "text_config", config)

    num_layers = (
        getattr(text_config, "num_hidden_layers", None)
        or getattr(text_config, "num_layers", None)
        or getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
        or getattr(config, "n_layer", None)
    )
    num_heads = (
        getattr(text_config, "num_attention_heads", None)
        or getattr(text_config, "num_heads", None)
        or getattr(config, "num_attention_heads", None)
        or getattr(config, "num_heads", None)
        or getattr(config, "n_head", None)
    )

    assert num_layers is not None, f"Could not determine num_layers from config: {list(config.to_dict().keys())}"
    assert num_heads is not None, f"Could not determine num_heads from config: {list(config.to_dict().keys())}"

    return num_layers, num_heads


# ---------------------------------------------------------------------------
# Context / depth grid
# ---------------------------------------------------------------------------


def build_context_depth_ranges(
    min_length: int,
    max_length: int,
    num_lengths: int,
    num_depths: int,
) -> tuple[list[int], list[int]]:
    """Build evenly-spaced context length and depth percent grids.

    Returns:
        (context_lengths, depth_percents) — both as lists of ints.
    """
    context_lengths = np.round(np.linspace(min_length, max_length, num=num_lengths, endpoint=True)).astype(int).tolist()
    depth_percents = np.round(np.linspace(0, 100, num=num_depths, endpoint=True)).astype(int).tolist()
    return context_lengths, depth_percents


# ---------------------------------------------------------------------------
# Head score I/O
# ---------------------------------------------------------------------------


def load_head_scores(json_path: str | Path) -> dict[str, float]:
    """Load retrieval head JSON and return {layer-head: mean_score}.

    Supports both flat format ``{"0-1": [...]}`` and envelope format
    ``{"scores": {"0-1": [...]}}`` (used by CRI output).
    Heads with no trials (empty list) are assigned a score of 0.
    """
    with open(json_path) as f:
        data = json.load(f)

    if "scores" in data and isinstance(data["scores"], dict):
        scores_data = data["scores"]
    else:
        scores_data = data

    result: dict[str, float] = {}
    for key, values in scores_data.items():
        if len(values) > 0:
            result[key] = sum(values) / len(values)
        else:
            result[key] = 0.0
    return result
