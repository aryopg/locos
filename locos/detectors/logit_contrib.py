#!/usr/bin/env python3
"""Contrastive logit-contribution scoring for retrieval head detection.

Measures whether a head's output at needle positions pushes the residual
stream toward the correct answer token in the unembedding space. Unlike
the attention-based contrastive method (detect_contrastive.py) which
measures *where a head looks*, this measures *what information it writes*.

For each answer decode step t and head (l, h), the per-position logit
contribution is:

    phi_{t,j}^{(l,h)} = u_{y_t}^T * W_O^{(l,h)} * v_j^{(l,h)} * alpha_{t,j}^{(l,h)}

where u_{y_t} is the unembedding vector for the correct answer token,
W_O is the output projection, v_j is the value vector at position j,
and alpha is the attention weight. The contrast is spatial (needle vs
off-needle) rather than temporal (answer vs non-answer steps).

Per-trial score: S^tau = L+ - L- where L+ sums phi over needle positions
and L- sums over off-needle positions (rescaled by span length ratio).

See docs/contrastive_logit_contribution_scoring.md for the full derivation.

Usage:
    # Quick test with NoLiMa
    python locos/detect_logit_contrib.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --dataset nolima --max-length 4000 --num-lengths 3

    # Full detection
    python locos/detect_logit_contrib.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --dataset nolima --min-length 1000 --max-length 50000

    # Resume from checkpoint
    python locos/detect_logit_contrib.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct --resume

Requires: GPU, transformers, rouge-score (pip install -e ".[eval]")
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gc
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rouge_score import rouge_scorer

from locos.detectors.contrastive import identify_answer_steps
from locos.utils.common import (
    build_context_depth_ranges,
    extract_model_config,
    format_duration,
    load_checkpoint,
    save_checkpoint,
)
from locos.utils.datasets import (
    build_niah_dataset,
    build_nolima_dataset,
    load_nolima_needle_set,
    stratified_sample,
)
from locos.utils.model_utils import (
    detect_thinking_tokens,
    get_decoder_layers,
    get_input_device,
    get_stop_token_ids,
    set_model_attn_impl,
    strip_thinking_content,
    tokenizer_adds_bos,
)
from locos.utils.needle_utils import (
    build_period_token_positions,
    build_tracked_prompt,
    get_period_tokens,
)
from locos.utils.needle_utils import load_needles as load_niah_needles

console = Console()
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


# ---------------------------------------------------------------------------
# Model component extraction
# ---------------------------------------------------------------------------


def extract_head_config(model) -> tuple[int, int, int, int]:
    """Extract (num_layers, num_heads, num_kv_heads, head_dim) from model config.

    num_heads is the number of Q-heads (attention heads).
    num_kv_heads is the number of KV-heads (may differ for GQA models).
    """
    config = model.config
    text_config = getattr(config, "text_config", config)

    num_layers, num_heads = extract_model_config(model)

    num_kv_heads = (
        getattr(text_config, "num_key_value_heads", None)
        or getattr(config, "num_key_value_heads", None)
        or num_heads  # Non-GQA: KV heads == Q heads
    )

    # Prefer explicit head_dim from config (e.g. Qwen3-4B has head_dim=128
    # with hidden_size=2560 and num_heads=32, so hidden_size // num_heads != head_dim).
    head_dim = getattr(text_config, "head_dim", None) or getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(text_config, "hidden_size", None) or getattr(config, "hidden_size", None)
        assert hidden_size is not None, "Could not determine hidden_size or head_dim"
        head_dim = hidden_size // num_heads

    return num_layers, num_heads, num_kv_heads, head_dim


def get_o_proj_weights(model, num_layers: int, num_heads: int, head_dim: int) -> tuple[list[torch.Tensor], int]:
    """Extract per-layer o_proj weights reshaped for per-head access.

    Returns ``(weights, actual_num_heads)`` where *weights* is a list of
    tensors (one per layer, each shape ``(num_heads, head_dim, hidden_dim)``)
    and *actual_num_heads* is derived from the weight shape.  Some VLMs
    (e.g. Gemma 4) report fewer ``num_attention_heads`` in the config than
    the o_proj weight implies, so the caller should use *actual_num_heads*.
    """
    decoder_layers = get_decoder_layers(model)
    assert (
        len(decoder_layers) == num_layers
    ), f"Expected {num_layers} decoder layers from config, found {len(decoder_layers)}"

    # Derive actual num_heads from the first layer's o_proj weight.
    w0 = decoder_layers[0].self_attn.o_proj.weight
    in_features = w0.shape[1]
    actual_num_heads = in_features // head_dim
    assert (
        in_features == actual_num_heads * head_dim
    ), f"o_proj in_features={in_features} not divisible by head_dim={head_dim}"
    if actual_num_heads != num_heads:
        from rich.console import Console

        Console().print(
            f"[yellow]Config num_attention_heads={num_heads} but o_proj implies "
            f"{actual_num_heads} heads (in_features={in_features}, head_dim={head_dim}). "
            f"Using {actual_num_heads}.[/yellow]"
        )

    o_proj_per_layer = []
    for layer_idx in range(num_layers):
        # o_proj.weight shape: (hidden_dim, num_heads * head_dim)
        w = decoder_layers[layer_idx].self_attn.o_proj.weight.detach()
        # Some models store weights as non-contiguous views of fused
        # parameters; .contiguous() materialises a clean copy.
        if not w.is_contiguous():
            w = w.contiguous()
        hidden_dim = w.shape[0]
        assert w.numel() == hidden_dim * actual_num_heads * head_dim, (
            f"Layer {layer_idx} o_proj size mismatch: shape={tuple(w.shape)}, "
            f"numel={w.numel()}, expected {hidden_dim}×{actual_num_heads}×{head_dim}"
            f"={hidden_dim * actual_num_heads * head_dim}"
        )
        # Reshape to (hidden_dim, num_heads, head_dim) then permute to (num_heads, head_dim, hidden_dim)
        w_reshaped = w.reshape(hidden_dim, actual_num_heads, head_dim).permute(1, 2, 0)
        o_proj_per_layer.append(w_reshaped)
    return o_proj_per_layer, actual_num_heads


def get_unembedding_matrix(model) -> torch.Tensor:
    """Get the unembedding (lm_head) weight matrix.

    Returns shape (vocab_size, hidden_dim).
    Handles standard (model.lm_head) and composite/VLM layouts
    (model.language_model.lm_head).
    """
    # Standard: model.lm_head
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight.detach()

    # Composite / VLM (e.g. Gemma3ForConditionalGeneration)
    lang = getattr(model, "language_model", None)
    if lang is None:
        lang = getattr(getattr(model, "model", None), "language_model", None)
    if lang is not None and hasattr(lang, "lm_head") and hasattr(lang.lm_head, "weight"):
        return lang.lm_head.weight.detach()

    # Tied embeddings fallback
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "embed_tokens"):
        return inner.embed_tokens.weight.detach()

    raise RuntimeError("Could not find lm_head or embed_tokens. " "Model architecture may not be supported.")


def _resolve_tuned_lens_path(
    *,
    repo_id: str | None = None,
    url: str | None = None,
    filename: str = "translators.pt",
) -> str:
    """Resolve a local file path for a tuned-lens .pt artifact.

    Supports three inputs:

    - ``repo_id``: HF model repo hosting a ``translators.pt`` (uzaymacar convention).
    - ``url``: Arbitrary URL pointing directly at a ``.pt`` file.
      If the URL is on ``huggingface.co``, parsed into ``hf_hub_download`` args
      (models / datasets / spaces) so we get HF caching. Otherwise fetched via
      ``urllib`` into ``~/.cache/locos_eval/tuned_lens/``.

    Exactly one of ``repo_id`` or ``url`` must be provided.
    """
    if (repo_id is None) == (url is None):
        raise ValueError("Provide exactly one of repo_id or url.")

    from huggingface_hub import hf_hub_download

    if repo_id is not None:
        return hf_hub_download(repo_id=repo_id, filename=filename)

    assert url is not None
    hf_parsed = _parse_hf_url(url)
    if hf_parsed is not None:
        hf_repo_id, hf_filename, repo_type, revision = hf_parsed
        return hf_hub_download(
            repo_id=hf_repo_id,
            filename=hf_filename,
            repo_type=repo_type,
            revision=revision,
        )

    import hashlib
    import urllib.request

    cache_dir = Path.home() / ".cache" / "locos_eval" / "tuned_lens"
    cache_dir.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    suffix = Path(url.split("?")[0]).suffix or ".pt"
    cache_path = cache_dir / f"{url_hash}{suffix}"
    if not cache_path.exists():
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.rename(cache_path)
    return str(cache_path)


def _parse_hf_url(url: str) -> tuple[str, str, str, str] | None:
    """Parse an ``https://huggingface.co/...`` URL into hf_hub_download kwargs.

    Returns ``(repo_id, filename, repo_type, revision)`` if the URL matches
    the resolve/blob pattern; otherwise returns None so callers can fall back
    to plain HTTP.

    Recognised patterns (paths after ``huggingface.co/``):

    - ``<owner>/<repo>/resolve/<rev>/<path>``                       → model
    - ``datasets/<owner>/<repo>/resolve/<rev>/<path>``              → dataset
    - ``spaces/<owner>/<repo>/resolve/<rev>/<path>``                → space
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.netloc != "huggingface.co":
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 5:
        return None

    repo_type = "model"
    idx = 0
    if parts[0] in ("datasets", "spaces"):
        repo_type = {"datasets": "dataset", "spaces": "space"}[parts[0]]
        idx = 1
    if len(parts) < idx + 4:
        return None
    owner, repo, kind, revision, *rest = parts[idx:]
    if kind not in ("resolve", "blob") or not rest:
        return None
    repo_id = f"{owner}/{repo}"
    filename = "/".join(rest)
    return repo_id, filename, repo_type, revision


_AR_KEY_RE = re.compile(r"^(?:layer_translators\.)?(\d+)\.(weight|bias)$")


def _ar_layer_keys(data: dict) -> dict[int, dict[str, str]]:
    """Group AlignmentResearch-style flat state-dict keys by layer index.

    Matches both the bare ``{i}.weight`` / ``{i}.bias`` convention (observed
    in AlignmentResearch's hosted ``params.pt``) and the fully-qualified
    ``layer_translators.{i}.weight`` / ``.bias`` form (older `tuned-lens`
    library checkpoints).
    """
    out: dict[int, dict[str, str]] = {}
    for k in data:
        if not isinstance(k, str):
            continue
        m = _AR_KEY_RE.match(k)
        if m is None:
            continue
        idx = int(m.group(1))
        out.setdefault(idx, {})[m.group(2)] = k
    return out


def _detect_tuned_lens_format(data) -> str:
    """Sniff the on-disk tuned-lens format.

    Returns:
        "uzaymacar"          — dict keyed by int layer, entries {"A", "b"};
                               A stored in *full* form (init identity). We must
                               subtract I to recover residual form.
        "alignmentresearch"  — flat state dict of per-layer affine translators
                               with keys ``{i}.weight`` / ``{i}.bias`` (or the
                               older ``layer_translators.{i}.weight``/``.bias``
                               form). Weights are already in residual form.
    """
    if not isinstance(data, dict) or not data:
        raise ValueError("Tuned-lens file does not contain a dict; unrecognized format.")

    ar_layers = _ar_layer_keys(data)
    if ar_layers and all("weight" in v and "bias" in v for v in ar_layers.values()):
        return "alignmentresearch"

    int_keyed = all(isinstance(k, int) for k in data)
    if int_keyed:
        first = next(iter(data.values()))
        if isinstance(first, dict) and "A" in first and "b" in first:
            return "uzaymacar"

    raise ValueError(
        "Could not detect tuned-lens format from top-level keys "
        f"(sample: {list(data.keys())[:5]}). Expected uzaymacar "
        '({int: {"A", "b"}}) or AlignmentResearch '
        "('{i}.weight/bias' or 'layer_translators.{i}.weight/bias').",
    )


def _collect_alignmentresearch_translators(
    state: dict,
    num_layers: int,
    hidden_dim: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract per-layer (A_res, b) from an AlignmentResearch flat state dict.

    AR's Lens module implements TL(h) = h + W @ h + b, so the stored weight is
    already A_residual — no identity subtraction needed.
    """
    layer_keys = _ar_layer_keys(state)
    if len(layer_keys) < num_layers:
        raise ValueError(
            f"AlignmentResearch tuned-lens has {len(layer_keys)} layer translators but model requires "
            f"{num_layers}. Ensure the tuned-lens was trained on the same model."
        )

    translators: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx in range(num_layers):
        entry = layer_keys.get(layer_idx)
        if entry is None or "weight" not in entry or "bias" not in entry:
            raise ValueError(
                f"AlignmentResearch tuned-lens missing weight/bias for layer {layer_idx}; "
                f"found keys: {sorted((entry or {}).values())}."
            )
        A_res = state[entry["weight"]].detach().float()
        b = state[entry["bias"]].detach().float()
        expected_shape = (hidden_dim, hidden_dim)
        if A_res.shape != expected_shape:
            raise ValueError(
                f"Layer {layer_idx}: weight shape {tuple(A_res.shape)} doesn't match expected hidden dim "
                f"{expected_shape}. Ensure the tuned-lens matches the model."
            )
        if b.shape != (hidden_dim,):
            raise ValueError(f"Layer {layer_idx}: bias shape {tuple(b.shape)} doesn't match ({hidden_dim},).")
        translators.append((A_res, b))

    return translators


def load_tuned_lens_translators(
    path: str,
    num_layers: int,
    hidden_dim: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load per-layer affine translators from a tuned-lens .pt file.

    Each translator is an affine map that projects intermediate residual-stream
    activations into the final-layer space, correcting for the direct-path
    unembedding bias (H3 hypothesis).

    Two on-disk formats are auto-detected:

    - **uzaymacar**: ``{layer_idx: {"A": ..., "b": ...}}`` or
      ``{"translators": {layer_idx: {"A": ..., "b": ...}}}``. ``A`` is the
      *full* transformation matrix with identity initialization. We convert
      to residual form ``A_res = A - I`` so ``apply_tuned_lens_correction``
      receives the near-zero perturbation rather than a near-identity matrix.

    - **AlignmentResearch**: flat state dict from the ``tuned-lens`` library
      (``layer_translators.{i}.weight`` / ``.bias``). The module implements
      ``TL(h) = h + W @ h + b`` internally, so the stored weight is already
      ``A_res`` — no identity subtraction.

    Args:
        path: Path to the .pt file containing the tuned-lens translators.
        num_layers: Expected number of decoder layers in the model.
        hidden_dim: Expected hidden dimension of the model.

    Returns:
        List of (A_residual, b) tuples, one per layer (0..num_layers-1).
        A_residual has shape (hidden_dim, hidden_dim), b has shape (hidden_dim,).
        Both are float32 on CPU.

    Raises:
        ValueError: If the file format is unrecognized, has fewer layers than
            num_layers, or shapes don't match hidden_dim.
    """
    # weights_only=False because uzaymacar's translators.pt contains numpy
    # scalars that the safe unpickler rejects, and AlignmentResearch files may
    # also need it. Only pass paths to trusted artifacts.
    data = torch.load(path, map_location="cpu", weights_only=False)

    # Handle uzaymacar wrapped format: {"translators": {0: {"A": ..., "b": ...}, ...}}
    if isinstance(data, dict) and "translators" in data and isinstance(data["translators"], dict):
        data = data["translators"]

    fmt = _detect_tuned_lens_format(data)

    if fmt == "alignmentresearch":
        return _collect_alignmentresearch_translators(data, num_layers, hidden_dim)

    if len(data) < num_layers:
        raise ValueError(
            f"Tuned-lens file has {len(data)} layers but model requires {num_layers} layers. "
            f"Ensure the tuned-lens was trained on the same model."
        )

    eye = torch.eye(hidden_dim)
    translators: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx in range(num_layers):
        entry = data[layer_idx]
        A_full = entry["A"].detach().float()
        b = entry["b"].detach().float()

        expected_shape = (hidden_dim, hidden_dim)
        if A_full.shape != expected_shape:
            raise ValueError(
                f"Layer {layer_idx}: A shape {tuple(A_full.shape)} doesn't match expected hidden dim "
                f"{expected_shape}. Ensure the tuned-lens matches the model."
            )
        assert b.shape == (hidden_dim,), f"Layer {layer_idx}: b shape {b.shape} != ({hidden_dim},)"

        A_res = A_full - eye
        translators.append((A_res, b))

    return translators


def apply_tuned_lens_correction(
    u_y: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Apply tuned lens affine correction to an unembedding vector.

    The tuned lens at layer l is a residual affine map:
        TL_l(o) = o + A_l @ o + b_l = (I + A_l) @ o + b_l

    For logit contribution phi = u_y^T @ TL_l(o), this decomposes as:
        phi = u_y_corrected^T @ o + bias_scalar
    where:
        u_y_corrected = (I + A_l)^T @ u_y
        bias_scalar   = u_y^T @ b_l

    Args:
        u_y: (hidden_dim,) unembedding vector for the correct answer token.
        A: (hidden_dim, hidden_dim) translator weight matrix for this layer.
        b: (hidden_dim,) translator bias for this layer.

    Returns:
        (u_y_corrected, bias_scalar): corrected unembedding vector and scalar bias.
    """
    device = u_y.device
    dtype = u_y.dtype
    A = A.to(device=device, dtype=dtype)
    b = b.to(device=device, dtype=dtype)
    # u_corrected = (I + A)^T @ u_y = u_y + A^T @ u_y
    u_corrected = u_y + A.T @ u_y
    bias_scalar = float(u_y @ b)
    return u_corrected, bias_scalar


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_logit_contribution_per_step(
    attn_weights: torch.Tensor,
    value_cache: torch.Tensor,
    o_proj_weight: torch.Tensor,
    u_y: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    needle_start: int,
    needle_end: int,
    context_len: int,
    logit_bias: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-head logit contribution from needle vs off-needle positions.

    For one layer, one decode step.

    Args:
        attn_weights: (num_heads, key_len) attention weights for this step.
        value_cache: (num_kv_heads, key_len, head_dim) V cache up to this step.
        o_proj_weight: (num_heads, head_dim, hidden_dim) per-head output projection.
        u_y: (hidden_dim,) unembedding vector for the correct answer token.
        num_heads: Number of Q-heads.
        num_kv_heads: Number of KV-heads (for GQA expansion).
        needle_start, needle_end: Needle span [start, end) in key positions.
        context_len: Total context length (for off-needle rescaling).
        logit_bias: Scalar added to per-position logit contribution before
            attention scaling (for tuned-lens correction). Default 0.0 (no bias).

    Returns:
        (phi_needle, phi_off_rescaled) each shape (num_heads,) as numpy arrays.
        phi_needle: sum of phi over needle positions.
        phi_off_rescaled: sum of phi over off-needle positions, rescaled by
            (needle_len / off_needle_len) so both represent the contribution
            of a span of length needle_len.
    """
    key_len = attn_weights.shape[1]
    needle_len = needle_end - needle_start
    off_needle_len = key_len - needle_len
    assert needle_len > 0
    assert off_needle_len > 0, f"No off-needle positions: key_len={key_len}, needle_len={needle_len}"

    # Expand V cache for GQA: (num_kv_heads, key_len, head_dim) -> (num_heads, key_len, head_dim)
    if num_kv_heads != num_heads:
        gqa_ratio = num_heads // num_kv_heads
        V = value_cache.repeat_interleave(gqa_ratio, dim=0)  # (num_heads, key_len, head_dim)
    else:
        V = value_cache  # (num_heads, key_len, head_dim)

    assert V.shape[0] == num_heads

    # Ensure all tensors are on the same device (multi-GPU device_map splits
    # layers across GPUs, so V cache, o_proj, and u_y may be on different devices)
    compute_device = V.device
    o_proj_weight = o_proj_weight.to(compute_device)
    u_y = u_y.to(compute_device)
    attn_weights = attn_weights.to(compute_device)

    # Precompute per-head projection of u_y through W_O:
    # u_projected[h] = W_O[h]^T @ u_y, shape (head_dim,)
    # o_proj_weight: (num_heads, head_dim, hidden_dim)
    # u_y: (hidden_dim,)
    u_projected = torch.einsum("hde,e->hd", o_proj_weight, u_y)  # (num_heads, head_dim)

    # Per-position logit contribution (before attention scaling):
    # logit_contrib[h, j] = v_j^{(h)} . u_projected[h]
    logit_contrib = torch.einsum("hkd,hd->hk", V, u_projected)  # (num_heads, key_len)

    if logit_bias != 0.0:
        logit_contrib = logit_contrib + logit_bias

    # Scale by attention weights: phi[h, j] = alpha_{t,j}^{(h)} * logit_contrib[h, j]
    phi = attn_weights * logit_contrib  # (num_heads, key_len)

    # Sum over needle and off-needle positions
    phi_needle = phi[:, needle_start:needle_end].sum(dim=-1)  # (num_heads,)
    phi_off = phi.sum(dim=-1) - phi_needle  # (num_heads,)

    # Rescale off-needle by (needle_len / off_needle_len) so both represent
    # the average contribution of a span of length needle_len (Eq. L- in design doc)
    scale = needle_len / off_needle_len
    phi_off_rescaled = phi_off * scale  # (num_heads,)

    return phi_needle.float().cpu().numpy(), phi_off_rescaled.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Trial result
# ---------------------------------------------------------------------------


@dataclass
class LogitContribTrialResult:
    """Result of a single logit-contribution detection trial."""

    S_tau: np.ndarray  # (num_layers, num_heads) per-trial score S = L+ - L-
    L_plus: np.ndarray  # (num_layers, num_heads) needle logit contribution
    L_minus: np.ndarray  # (num_layers, num_heads) off-needle contribution (rescaled)
    generated_text: str
    num_answer_steps: int
    num_total_steps: int


# ---------------------------------------------------------------------------
# Single trial detection
# ---------------------------------------------------------------------------


def detect_single_trial_logit_contrib(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    needle_start: int,
    needle_end: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    answer_text: str,
    prefill_attn_impl: str,
    o_proj_weights: list[torch.Tensor],
    W_U: torch.Tensor,
    max_decode_steps: int = 50,
    newline_token_id: int | None = None,
    stop_token_ids: set[int] | None = None,
    think_start_id: int | None = None,
    think_end_id: int | None = None,
    max_thinking_tokens: int = 0,
    translators: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> LogitContribTrialResult:
    """Run one logit-contribution retrieval head detection trial.

    Args:
        model: HuggingFace model (with output_attentions support).
        tokenizer: Tokenizer instance.
        input_ids: Full prompt token IDs, shape (1, seq_len).
        needle_start, needle_end: Token positions of the needle in input_ids.
        num_layers, num_heads, num_kv_heads, head_dim: Model dims.
        answer_text: Gold answer string for answer step identification.
        prefill_attn_impl: Attention backend for prefill.
        o_proj_weights: Per-layer o_proj weights, each (num_heads, head_dim, hidden_dim).
        W_U: Unembedding matrix, shape (vocab_size, hidden_dim).
        max_decode_steps: Maximum decode steps before stopping.
        newline_token_id: Token ID for newline (stop condition).
        think_start_id, think_end_id: Token IDs for ``<think>``/``</think>``
            markers (None to disable thinking handling).
        max_thinking_tokens: Maximum tokens allowed inside thinking blocks.
        translators: Per-layer tuned lens (A, b) pairs from load_tuned_lens_translators().
            When provided, applies affine correction to the unembedding projection.
            None disables correction (standard direct-path scoring).

    Returns:
        LogitContribTrialResult with S_tau, L_plus, L_minus, and diagnostics.
        *generated_text* in the result excludes thinking content.
    """
    device = get_input_device(model)
    input_ids = input_ids.to(device)
    # --- Prefill ---
    switched = False
    if prefill_attn_impl != "eager":
        switched = set_model_attn_impl(model, prefill_attn_impl)

    with torch.inference_mode():
        try:
            prefill_out = model(
                input_ids=input_ids[:, :-1],
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
        except Exception as err:
            if switched:
                set_model_attn_impl(model, "eager")
                prefill_out = model(
                    input_ids=input_ids[:, :-1],
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True,
                )
                console.print(
                    f"[yellow]Prefill backend '{prefill_attn_impl}' failed; "
                    f"fell back to eager. ({type(err).__name__})[/yellow]"
                )
            else:
                raise
        past_kv = prefill_out.past_key_values

    if prefill_attn_impl != "eager":
        set_model_attn_impl(model, "eager")

    # --- Autoregressive decode ---
    current_token = input_ids[:, -1:].clone()
    generated_ids: list[int] = []  # content tokens only (excludes thinking)

    # Store per-step attention weights for all layers (content steps only)
    # Each entry: list of (num_heads, key_len) tensors, one per layer
    step_attentions: list[list[torch.Tensor]] = []

    if newline_token_id is None:
        newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

    has_thinking = think_start_id is not None and think_end_id is not None
    in_thinking = False
    thinking_token_count = 0
    content_steps = 0

    with torch.inference_mode():
        for _ in range(max_decode_steps + max_thinking_tokens):
            outputs = model(
                input_ids=current_token,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=not in_thinking,
                return_dict=True,
            )
            past_kv = outputs.past_key_values

            next_token_id = outputs.logits[0, -1].argmax().item()

            # --- Thinking block handling ---
            if has_thinking:
                if next_token_id == think_start_id:
                    in_thinking = True
                    thinking_token_count += 1
                    current_token[0, 0] = next_token_id
                    continue
                if next_token_id == think_end_id:
                    in_thinking = False
                    thinking_token_count += 1
                    current_token[0, 0] = next_token_id
                    continue
                if in_thinking:
                    thinking_token_count += 1
                    if thinking_token_count > max_thinking_tokens:
                        break
                    current_token[0, 0] = next_token_id
                    continue

            # --- Stop checks (before appending, so stop tokens stay out of generated_ids) ---
            step_token = tokenizer.convert_ids_to_tokens(next_token_id)
            if step_token == "<0x0A>" or next_token_id == newline_token_id:
                break
            if stop_token_ids and next_token_id in stop_token_ids:
                break

            # --- Content token ---
            generated_ids.append(next_token_id)
            content_steps += 1

            # Store attention weights for this step (detach to CPU to save GPU memory)
            step_attn = [a[0, :, -1, :].detach().cpu() for a in outputs.attentions]  # (num_heads, key_len)
            step_attentions.append(step_attn)

            if content_steps >= max_decode_steps:
                break

            current_token[0, 0] = next_token_id

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    num_total_steps = len(generated_ids)

    # --- Identify answer steps ---
    t_ans, _ = identify_answer_steps(generated_ids, answer_text, tokenizer)

    if len(t_ans) == 0:
        zeros = np.zeros((num_layers, num_heads), dtype=np.float32)
        return LogitContribTrialResult(
            S_tau=zeros,
            L_plus=zeros,
            L_minus=zeros,
            generated_text=generated_text,
            num_answer_steps=0,
            num_total_steps=num_total_steps,
        )

    # --- Compute logit contributions for answer steps ---
    # For each answer step, we need:
    #   - attention weights (stored in step_attentions)
    #   - V cache at that step (needle positions are stable from prefill)
    #   - u_{y_t} for the correct answer token at that step

    # Get unembedding vectors for answer tokens
    answer_token_ids = [generated_ids[t] for t in t_ans]
    u_vectors = W_U[answer_token_ids].to(device)  # (num_answer_steps, hidden_dim)

    # Accumulate L+ and L- across answer steps
    L_plus_accum = np.zeros((num_layers, num_heads), dtype=np.float64)
    L_minus_accum = np.zeros((num_layers, num_heads), dtype=np.float64)

    for ans_idx, t in enumerate(t_ans):
        u_y = u_vectors[ans_idx]  # (hidden_dim,)
        attn_at_t = step_attentions[t]  # list of (num_heads, key_len) per layer

        for layer_idx in range(num_layers):
            # Get V cache for this layer
            # value_states shape: (batch, num_kv_heads, seq_len, head_dim)
            # FIXME: After decode, past_kv contains the full sequence (prefill + decode).
            # The V values at needle positions (in the prefix) are unchanged across
            # decode steps, so using the final V cache is correct for needle positions.
            # However, the attention weights at step t were computed over only the
            # positions that existed at step t (prefill + steps 0..t). The V cache
            # now has extra positions from steps t+1..end. We index attention weights
            # (which have the correct key_len for step t) and only use V at those
            # positions. Since attn_at_t has shape (num_heads, key_len_at_t), and
            # V cache positions 0..key_len_at_t-1 haven't changed, this is safe.
            if hasattr(past_kv, "layers"):
                # Layer-based Cache (transformers ≥4.49): .layers[i].values
                v_cache = past_kv.layers[layer_idx].values[0]  # (num_kv_heads, cache_len, head_dim)
            elif hasattr(past_kv, "value_cache"):
                # Older DynamicCache: .value_cache[i]
                v_cache = past_kv.value_cache[layer_idx][0]  # (num_kv_heads, cache_len, head_dim)
            else:
                # Legacy tuple format: past_kv[layer_idx] is (key_states, value_states)
                v_cache = past_kv[layer_idx][1][0]  # (num_kv_heads, cache_len, head_dim)
            key_len_at_t = attn_at_t[layer_idx].shape[1]
            cache_len = v_cache.shape[1]
            # Sliding-window / hybrid caches may have fewer entries than the
            # attention attended over.  Truncate to the available cache length
            # (attention weights for evicted positions are discarded).
            effective_len = min(key_len_at_t, cache_len)
            v_cache_at_t = v_cache[:, :effective_len, :]  # (num_kv_heads, effective_len, head_dim)

            attn_weights = attn_at_t[layer_idx][:, :effective_len].to(device)  # (num_heads, effective_len)
            o_proj_w = o_proj_weights[layer_idx].to(device)  # (num_heads, head_dim, hidden_dim)

            # Apply tuned-lens correction if translators are provided
            u_y_for_layer = u_y
            layer_logit_bias = 0.0
            if translators is not None:
                A_l, b_l = translators[layer_idx]
                u_y_for_layer, layer_logit_bias = apply_tuned_lens_correction(u_y, A_l, b_l)

            phi_needle, phi_off = compute_logit_contribution_per_step(
                attn_weights=attn_weights,
                value_cache=v_cache_at_t,
                o_proj_weight=o_proj_w,
                u_y=u_y_for_layer,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                needle_start=needle_start,
                needle_end=needle_end,
                context_len=effective_len,
                logit_bias=layer_logit_bias,
            )

            L_plus_accum[layer_idx] += phi_needle
            L_minus_accum[layer_idx] += phi_off

    # Average over answer steps
    n_ans = len(t_ans)
    L_plus = (L_plus_accum / n_ans).astype(np.float32)
    L_minus = (L_minus_accum / n_ans).astype(np.float32)
    S_tau = (L_plus - L_minus).astype(np.float32)

    return LogitContribTrialResult(
        S_tau=S_tau,
        L_plus=L_plus,
        L_minus=L_minus,
        generated_text=generated_text,
        num_answer_steps=n_ans,
        num_total_steps=num_total_steps,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Logit-contribution contrastive retrieval head detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: retrieval_heads/<model>_logit_contrib[_<dataset>].json)",
    )
    parser.add_argument("--dataset", type=str, default="nolima", choices=["niah", "nolima"], help="Probing dataset")
    parser.add_argument(
        "--haystack-dir", type=Path, default=None, help="Directory with NIAH data (default: data/haystack_for_detect)"
    )
    parser.add_argument("--nolima-dir", type=Path, default=Path("data/nolima"), help="Directory with NoLiMa data")
    parser.add_argument("--question-type", type=str, default="onehop", choices=["onehop", "twohop", "twohop2"])
    parser.add_argument("--nolima-variant", type=str, default="needle_set", choices=["needle_set", "needle_set_hard"])
    parser.add_argument("--max-characters-per-entry", type=int, default=1)
    parser.add_argument("--min-length", type=int, default=1000, help="Minimum context length in tokens")
    parser.add_argument("--max-length", type=int, default=50000, help="Maximum context length in tokens")
    parser.add_argument("--num-lengths", type=int, default=20)
    parser.add_argument("--num-depths", type=int, default=10)
    parser.add_argument(
        "--num-examples", type=int, default=None, help="If set, use stratified sampling to limit trial count"
    )
    parser.add_argument("--max-decode-steps", type=int, default=50)
    parser.add_argument(
        "--debug-trials",
        type=int,
        default=0,
        help="Print raw generated text and ROUGE for the first N trials (for diagnosing model output)",
    )
    parser.add_argument(
        "--rouge-threshold", type=float, default=50.0, help="Minimum ROUGE-1 recall * 100 to accept a trial"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--prefill-attn-impl",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--use-hardcoded-periods", action="store_true", default=False)
    parser.add_argument("--force-llama2-periods", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-thinking-tokens",
        type=int,
        default=0,
        help=(
            "Max tokens to allow inside <think>…</think> blocks during decode. "
            "Set >0 for reasoning models (e.g. 512). 0 disables thinking handling."
        ),
    )
    parser.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Wrap the prompt in the tokenizer's chat template before detection. "
            "Required for models that need structured prompts (e.g. GPT-oss). "
            "Needle positions are re-located after re-tokenization."
        ),
    )
    parser.add_argument(
        "--prompt-suffix",
        type=str,
        default=None,
        help=(
            "String to append after the prompt (post chat-template if enabled). "
            "Tokenized and appended to the token sequence before detection. "
            "E.g. '<|channel|>final<|message|>' for GPT-oss to skip reasoning."
        ),
    )
    parser.add_argument(
        "--tuned-lens",
        type=str,
        default=None,
        help=(
            "HuggingFace repo ID for a pretrained tuned lens (e.g. "
            "'uzaymacar/gemma-3-27b-tuned-lens'). Downloads translators.pt "
            "and applies affine correction to unembedding projections per layer. "
            "See docs/h3_direct_path_bias.md for motivation."
        ),
    )
    parser.add_argument(
        "--tuned-lens-url",
        type=str,
        default=None,
        help=(
            "Direct URL to a tuned-lens .pt file (e.g. the AlignmentResearch lens at "
            "https://huggingface.co/spaces/AlignmentResearch/tuned-lens/resolve/main/"
            "lens/meta-llama/Meta-Llama-3-8B-Instruct/params.pt). Format (uzaymacar "
            "vs AlignmentResearch flat state-dict) is auto-detected. Mutually "
            "exclusive with --tuned-lens."
        ),
    )
    args = parser.parse_args()

    if args.tuned_lens and args.tuned_lens_url:
        parser.error("--tuned-lens and --tuned-lens-url are mutually exclusive.")

    # Resolve output path
    model_short_name = args.model.split("/")[-1]
    dataset_suffix = f"_{args.dataset}" if args.dataset != "niah" else ""
    tl_suffix = "_tuned_lens" if (args.tuned_lens or args.tuned_lens_url) else ""
    if args.output is None:
        output_path = Path("retrieval_heads") / f"{model_short_name}_logit_contrib{dataset_suffix}{tl_suffix}.json"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    if args.haystack_dir is None:
        args.haystack_dir = Path("data/haystack_for_detect")

    console.rule("[bold]Logit-Contribution Retrieval Head Detection[/bold]")

    # Build test grid
    context_lengths, depth_percents = build_context_depth_ranges(
        args.min_length,
        args.max_length,
        args.num_lengths,
        args.num_depths,
    )

    # Build dataset trials
    if args.dataset == "niah":
        needles = load_niah_needles(args.haystack_dir)
        dataset_trials = build_niah_dataset(
            args.haystack_dir,
            context_lengths,
            depth_percents,
            max_tokens=args.max_length,
        )
        dataset_info = (
            f"{len(needles)} needles, "
            f"{args.num_lengths} lengths x {args.num_depths} depths x {len(needles)} needles"
        )
    elif args.dataset == "nolima":
        nolima_entries = load_nolima_needle_set(args.nolima_dir, args.nolima_variant)
        dataset_trials = build_nolima_dataset(
            args.nolima_dir,
            context_lengths,
            depth_percents,
            question_type=args.question_type,
            variant=args.nolima_variant,
            max_tokens=args.max_length,
            max_characters_per_entry=args.max_characters_per_entry,
            seed=args.seed,
        )
        dataset_info = (
            f"{len(nolima_entries)} entries ({args.nolima_variant}), "
            f"question_type={args.question_type}, "
            f"max_chars/entry={args.max_characters_per_entry}"
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.num_examples is not None:
        dataset_trials = stratified_sample(
            dataset_trials,
            args.num_examples,
            seed=args.seed,
        )
        dataset_info += f" → sampled {len(dataset_trials)} trials"

    config_table = Table(title="Configuration", show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="bold")
    config_table.add_column("Value")
    config_table.add_row("Model", args.model)
    config_table.add_row("Method", "logit-contribution (spatial contrast)")
    config_table.add_row("Dataset", args.dataset)
    config_table.add_row("Output", str(output_path))
    config_table.add_row("Context range", f"{args.min_length} – {args.max_length} tokens")
    config_table.add_row("Dataset info", dataset_info)
    config_table.add_row("Total trials", str(len(dataset_trials)))
    config_table.add_row("Prefill backend", f"{args.prefill_attn_impl} (decode uses eager)")
    config_table.add_row("Chat template", str(args.chat_template))
    if args.prompt_suffix:
        config_table.add_row("Prompt suffix", args.prompt_suffix)
    if args.tuned_lens or args.tuned_lens_url:
        config_table.add_row("Tuned lens", args.tuned_lens or args.tuned_lens_url)
        config_table.add_row("Method", "logit-contribution (spatial contrast, tuned-lens-corrected)")
    console.print(config_table)

    # Load model
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    console.print(f"\nLoading model ({args.dtype}) ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
    tokenizer_has_bos = tokenizer_adds_bos(tokenizer)
    think_start_id, think_end_id = detect_thinking_tokens(tokenizer)
    if think_start_id is not None and args.max_thinking_tokens > 0:
        console.print(
            f"[cyan]Thinking tokens detected:[/cyan] "
            f"start={think_start_id}, end={think_end_id}, "
            f"budget={args.max_thinking_tokens}"
        )
    elif think_start_id is not None and args.max_thinking_tokens == 0:
        console.print(
            "[yellow]Thinking tokens detected but --max-thinking-tokens=0. " "Set >0 for reasoning models.[/yellow]"
        )
    if args.chat_template:
        has_template = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template
        if has_template:
            console.print("[cyan]Chat template enabled:[/cyan] prompts will be wrapped before detection")
        else:
            console.print("[yellow]--chat-template requested but tokenizer has no chat template[/yellow]")
    prompt_suffix_ids: list[int] = []
    if args.prompt_suffix:
        prompt_suffix_ids = tokenizer.encode(args.prompt_suffix, add_special_tokens=False)
        decoded_back = tokenizer.convert_ids_to_tokens(prompt_suffix_ids)
        console.print(
            f"[cyan]Prompt suffix:[/cyan] {args.prompt_suffix!r} → " f"{len(prompt_suffix_ids)} tokens {decoded_back}"
        )
    period_tokens = get_period_tokens(
        args.model,
        tokenizer,
        args.use_hardcoded_periods,
        args.force_llama2_periods,
    )

    # Token caches
    haystack_token_cache: dict[str, list[int]] = {}
    haystack_period_cache: dict[str, list[int]] = {}
    needle_token_cache: dict[str, list[int]] = {}
    question_token_cache: dict[str, list[int]] = {}

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        attn_implementation="eager",
        trust_remote_code=True,
    ).eval()

    num_layers, num_heads, num_kv_heads, head_dim = extract_head_config(model)

    # Precompute model components
    console.print("Extracting o_proj weights and unembedding matrix...")
    o_proj_weights, actual_num_heads = get_o_proj_weights(model, num_layers, num_heads, head_dim)
    if actual_num_heads != num_heads:
        console.print(f"[cyan]Overriding num_heads: {num_heads} → {actual_num_heads} (from o_proj weight shape)[/cyan]")
        num_heads = actual_num_heads
    W_U = get_unembedding_matrix(model)

    # Optionally load tuned lens translators
    translators = None
    if args.tuned_lens or args.tuned_lens_url:
        source = args.tuned_lens or args.tuned_lens_url
        console.print(f"Downloading tuned lens from [cyan]{source}[/cyan] ...")
        tl_path = _resolve_tuned_lens_path(
            repo_id=args.tuned_lens,
            url=args.tuned_lens_url,
        )
        hidden_dim = W_U.shape[1]
        translators = load_tuned_lens_translators(tl_path, num_layers, hidden_dim)
        console.print(
            f"[green]Loaded tuned lens:[/green] {num_layers} layers, "
            f"hidden_dim={hidden_dim}, "
            f"params={num_layers * hidden_dim * (hidden_dim + 1):,}"
        )

    model_table = Table(title="Model", show_header=False, box=None, padding=(0, 2))
    model_table.add_column("Key", style="bold")
    model_table.add_column("Value")
    model_table.add_row("Architecture", f"{num_layers}L × {num_heads}H (Q) × {num_kv_heads}H (KV) × {head_dim}d")
    model_table.add_row("GQA ratio", str(num_heads // num_kv_heads))
    model_table.add_row("Vocab size", str(W_U.shape[0]))
    model_table.add_row("Dtype", args.dtype)
    console.print(model_table)
    stop_token_ids = get_stop_token_ids(tokenizer, model)
    console.print(f"[dim]Stop token IDs ({len(stop_token_ids)}): {stop_token_ids}[/dim]")
    console.print("[green]Model loaded[/green]\n")

    # Resume or fresh start
    if args.resume:
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt is not None:
            head_counter = defaultdict(list)
            for k, v in ckpt.get("head_counter", {}).items():
                head_counter[k] = v
            completed_trials = ckpt.get("completed_trials", [])
            passed_trial_ids = ckpt.get("passed_trial_ids", [])
            console.print(f"Resumed: {len(completed_trials)} trials already completed")
        else:
            head_counter = defaultdict(list)
            completed_trials = []
            passed_trial_ids = []
    else:
        head_counter = defaultdict(list)
        completed_trials = []
        passed_trial_ids = []

    completed_set = set(t if isinstance(t, str) else tuple(t) for t in completed_trials)
    trials = [t for t in dataset_trials if t.trial_id not in completed_set]
    total_trials = len(dataset_trials)
    console.print(f"Trials: {total_trials} total, {len(trials)} remaining\n")

    num_passed = 0
    num_no_answer_steps = 0

    # Pooled accumulators for global aggregation
    pooled_L_plus = np.zeros((num_layers, num_heads), dtype=np.float64)
    pooled_L_minus = np.zeros((num_layers, num_heads), dtype=np.float64)
    pooled_ans_count = 0
    # Per-trial S values for bootstrap CI
    per_trial_S: list[np.ndarray] = []

    if not trials:
        console.print("[yellow]All trials already completed.[/yellow]")
    else:
        loop_start_time = time.time()
        processed = 0
        oom_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[status]}[/dim]"),
            console=console,
        ) as progress:
            ptask = progress.add_task(
                "Detecting retrieval heads (logit-contrib)",
                total=len(trials),
                status="",
            )

            for trial in trials:
                elapsed_s = time.time() - loop_start_time
                if processed > 0:
                    avg_s = elapsed_s / processed
                    eta_s = (len(trials) - processed) * avg_s
                    timing = f"elapsed={format_duration(elapsed_s)} eta={format_duration(eta_s)}"
                else:
                    timing = "elapsed=00:00 eta=estimating..."

                progress.update(
                    ptask,
                    status=(
                        f"ctx={trial.context_length} depth={trial.depth_percent}% "
                        f"id={trial.trial_id[:30]} | {timing}"
                    ),
                )

                # --- Build context with needle ---
                haystack_tokens = haystack_token_cache.get(trial.haystack_text)
                if haystack_tokens is None:
                    haystack_tokens = tokenizer.encode(trial.haystack_text, add_special_tokens=False)
                    haystack_token_cache[trial.haystack_text] = haystack_tokens
                    haystack_period_cache[trial.haystack_text] = build_period_token_positions(
                        haystack_tokens, period_tokens
                    )
                needle_tokens = needle_token_cache.get(trial.needle_text)
                if needle_tokens is None:
                    needle_tokens = tokenizer.encode(trial.needle_text, add_special_tokens=False)
                    needle_token_cache[trial.needle_text] = needle_tokens

                # Token-space composition with tracked needle position.
                question_tokens = None
                if not (trial.prompt_template is not None and "{haystack}" in trial.prompt_template):
                    question_tokens = question_token_cache.get(trial.question)
                    if question_tokens is None:
                        question_tokens = tokenizer.encode(trial.question, add_special_tokens=False)
                        question_token_cache[trial.question] = question_tokens

                input_ids, needle_start, needle_end, _, _ = build_tracked_prompt(
                    haystack_tokens=haystack_tokens,
                    needle_tokens=needle_tokens,
                    context_length=trial.context_length,
                    depth_percent=trial.depth_percent,
                    period_positions=haystack_period_cache[trial.haystack_text],
                    tokenizer=tokenizer,
                    prompt_template=trial.prompt_template if "{haystack}" in (trial.prompt_template or "") else None,
                    question_tokens=question_tokens,
                    use_chat_template=bool(args.chat_template),
                    prompt_suffix_ids=prompt_suffix_ids,
                    add_bos=tokenizer_has_bos,
                )

                if needle_start < 0 or needle_end <= needle_start:
                    console.print(f"[yellow]Warning: invalid needle span ({trial.trial_id}). Skipping.[/yellow]")
                    completed_trials.append(trial.trial_id)
                    processed += 1
                    progress.advance(ptask)
                    continue

                # --- Run logit-contribution detection ---
                try:
                    result = detect_single_trial_logit_contrib(
                        model,
                        tokenizer,
                        input_ids,
                        needle_start,
                        needle_end,
                        num_layers,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        trial.answer_text,
                        args.prefill_attn_impl,
                        o_proj_weights=o_proj_weights,
                        W_U=W_U,
                        max_decode_steps=args.max_decode_steps,
                        newline_token_id=newline_token_id,
                        stop_token_ids=stop_token_ids,
                        think_start_id=think_start_id if args.max_thinking_tokens > 0 else None,
                        think_end_id=think_end_id if args.max_thinking_tokens > 0 else None,
                        max_thinking_tokens=args.max_thinking_tokens,
                        translators=translators,
                    )
                except torch.cuda.OutOfMemoryError:
                    console.print(f"[red]OOM at ctx={trial.context_length}. Skipping.[/red]")
                    oom_count += 1
                    torch.cuda.empty_cache()
                    gc.collect()
                    completed_trials.append(trial.trial_id)
                    processed += 1
                    progress.advance(ptask)
                    continue

                # --- ROUGE gate (strip residual thinking as safety net) ---
                rouge_text = (
                    strip_thinking_content(result.generated_text)
                    if args.max_thinking_tokens > 0
                    else result.generated_text
                )
                rouge_result = scorer.score(trial.answer_text, rouge_text)["rouge1"].recall * 100

                if args.debug_trials > 0 and processed < args.debug_trials:
                    progress.console.print(
                        f"\n[bold]--- Debug trial {processed + 1}/{args.debug_trials} ---[/bold]\n"
                        f"  [dim]trial_id:[/dim] {trial.trial_id}\n"
                        f"  [dim]ctx_len:[/dim]  {trial.context_length}  "
                        f"[dim]depth:[/dim] {trial.depth_percent}%\n"
                        f"  [dim]answer:[/dim]   {trial.answer_text!r}\n"
                        f"  [dim]generated (raw, repr):[/dim]\n    {result.generated_text!r}\n"
                        f"  [dim]answer_steps:[/dim] {result.num_answer_steps}/{result.num_total_steps}\n"
                        f"  [dim]ROUGE-1 recall:[/dim] {rouge_result:.1f}  "
                        f"({'[green]PASS[/green]' if rouge_result > args.rouge_threshold else '[red]FAIL[/red]'})"
                    )

                if rouge_result > args.rouge_threshold and result.num_answer_steps > 0:
                    num_passed += 1
                    passed_trial_ids.append(trial.trial_id)

                    # Accumulate for global pooling
                    pooled_L_plus += result.L_plus * result.num_answer_steps
                    pooled_L_minus += result.L_minus * result.num_answer_steps
                    pooled_ans_count += result.num_answer_steps

                    # Store per-trial S for bootstrap
                    per_trial_S.append(result.S_tau)

                    # Also store per-trial scores in head_counter
                    for layer_idx in range(num_layers):
                        for head_idx in range(num_heads):
                            key = f"{layer_idx}-{head_idx}"
                            head_counter[key].append(float(result.S_tau[layer_idx, head_idx]))

                elif rouge_result > args.rouge_threshold and result.num_answer_steps == 0:
                    num_no_answer_steps += 1

                completed_trials.append(trial.trial_id)
                processed += 1
                progress.advance(ptask)

                # Checkpoint every 10 trials
                if len(completed_trials) % 10 == 0:
                    save_checkpoint(
                        {
                            "head_counter": dict(head_counter),
                            "completed_trials": completed_trials,
                            "passed_trial_ids": passed_trial_ids,
                        },
                        checkpoint_path,
                    )

        total_elapsed = time.time() - loop_start_time
        console.print(f"\n[green]{num_passed}[/green] / {len(trials)} trials passed ROUGE gate")
        if num_no_answer_steps > 0:
            console.print(f"[yellow]{num_no_answer_steps} trials passed ROUGE but had 0 answer steps[/yellow]")
        if oom_count > 0:
            console.print(f"[yellow]{oom_count} OOM skips[/yellow]")
        console.print(f"Total time: {format_duration(total_elapsed)}")

    # --- Compute global pooled scores + bootstrap CI ---
    pooled_ci_lower = None
    pooled_ci_upper = None

    if pooled_ans_count > 0:
        console.print(f"\n[bold]Pooled:[/bold] {pooled_ans_count} answer steps across {num_passed} trials")

        # Bootstrap CI over trials
        if len(per_trial_S) >= 10:
            n_bootstrap = 1000
            rng = np.random.RandomState(args.seed)
            S_arr = np.stack(per_trial_S)  # (num_trials, num_layers, num_heads)
            boot_means = np.empty((n_bootstrap, num_layers, num_heads), dtype=np.float32)
            for b in range(n_bootstrap):
                idx = rng.randint(0, len(S_arr), size=len(S_arr))
                boot_means[b] = S_arr[idx].mean(axis=0)
            pooled_ci_lower = np.percentile(boot_means, 2.5, axis=0).astype(np.float32)
            pooled_ci_upper = np.percentile(boot_means, 97.5, axis=0).astype(np.float32)

            significant = (pooled_ci_lower > 0).sum()
            significant_neg = (pooled_ci_upper < 0).sum()
            console.print(
                f"[bold]Bootstrap 95% CI:[/bold] "
                f"{significant}/{num_layers * num_heads} heads with CI > 0 (needle-dominant), "
                f"{significant_neg}/{num_layers * num_heads} heads with CI < 0 (off-needle-dominant)"
            )

    # --- Save results ---
    scores_dict = {}
    for layer in range(num_layers):
        for head in range(num_heads):
            key = f"{layer}-{head}"
            scores_dict[key] = head_counter.get(key, [])

    result_envelope = {
        "meta": {
            "method": "logit_contribution",
            "dataset": args.dataset,
            "num_trials_passed": num_passed,
            "num_trials_total": total_trials,
            "pooled_ans_steps": pooled_ans_count,
            "rouge_threshold": args.rouge_threshold,
            "model": args.model,
            "question_type": args.question_type if args.dataset == "nolima" else None,
            "context_range": [args.min_length, args.max_length],
            "num_lengths": args.num_lengths,
            "num_depths": args.num_depths,
            "max_decode_steps": args.max_decode_steps,
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tuned_lens": args.tuned_lens or args.tuned_lens_url,
        },
        "scores": scores_dict,
        "trial_ids": passed_trial_ids,
    }

    if pooled_ci_lower is not None:
        ci_dict = {}
        for layer in range(num_layers):
            for head in range(num_heads):
                key = f"{layer}-{head}"
                ci_dict[key] = [
                    float(pooled_ci_lower[layer, head]),
                    float(pooled_ci_upper[layer, head]),
                ]
        result_envelope["confidence_intervals"] = ci_dict

    output_path.write_text(json.dumps(result_envelope, indent=2))
    console.print(f"\n[bold green]Saved logit-contribution scores to {output_path}[/bold green]")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Print top heads
    scored = [(key, float(np.mean(scores)) if scores else 0.0) for key, scores in scores_dict.items()]
    scored.sort(key=lambda x: x[1], reverse=True)

    table = Table(title="Top 20 Retrieval Heads (logit-contribution)")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Layer-Head")
    table.add_column("Mean S", justify="right")
    table.add_column("Trials", justify="right")

    for i, (key, mean_score) in enumerate(scored[:20]):
        table.add_row(
            str(i + 1),
            key,
            f"{mean_score:.6f}",
            str(len(scores_dict[key])),
        )
    console.print(table)

    # Also print bottom 5 (most negative — heads writing answer info from non-needle)
    table_neg = Table(title="Bottom 5 Retrieval Heads (off-needle dominant)")
    table_neg.add_column("Rank", style="dim", width=4)
    table_neg.add_column("Layer-Head")
    table_neg.add_column("Mean S", justify="right")

    scored_asc = sorted(scored, key=lambda x: x[1])
    for i, (key, mean_score) in enumerate(scored_asc[:5]):
        table_neg.add_row(str(len(scored) - i), key, f"{mean_score:.6f}")
    console.print(table_neg)

    # Validate output
    from locos_eval.retrieval_heads import load_retrieval_heads

    heads = load_retrieval_heads(str(output_path), num_heads=10)
    console.print(f"\nValidation: load_retrieval_heads() returned {len(heads)} heads")
    console.print(f"Top 10: {heads}")


if __name__ == "__main__":
    main()
