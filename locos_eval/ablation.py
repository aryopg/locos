"""Native zero/mean ablation: patches attention to override retrieval-head
queries WITHOUT bypassing vLLM's paged attention.

Unlike :mod:`locos_eval.attention` (which replaces SDPA with a manual
sequential KV cache for both base and masked passes), this module mirrors
each architecture's *original* attention forward — including the call to
``self.attn(q, k, v)`` (vLLM's paged-attention layer) — and only injects
a single q-replacement step right after the qkv split.

Two ablation modes are supported, matching ``nolima_ablation.make_zero_hook``
and ``nolima_ablation.make_mean_hook`` in :mod:`locos`:

* ``"zero"`` — replace masked-head queries with zeros.
* ``"mean"`` — replace masked-head queries with a precomputed per-(layer,
  head) mean q activation, captured during a calibration pass through the
  model.

Injection happens **before** QK-norm and rotary so that subsequent transforms
apply to the replacement value (matches the HF reference implementation).
For zero mode the choice is moot — RMSNorm(0)=0 and rot(0)=0 — but a single
intervention point keeps the two modes interchangeable.

Patched forward order:
    qkv_proj → split → REPLACE Q HEADS → (QK-norm) → rotary → self.attn → o_proj

Supported architectures (must match :mod:`locos_eval.attention`):
    LlamaAttention, Gemma3Attention, Qwen3Attention, Olmo2Attention.
"""

from collections.abc import Callable

import torch

from locos_eval.attention import _get_supported_attention_classes, get_decoder_layers


def build_ablation_attention_forward(
    attn_module,
    masked_heads_local: list[int],
    layer_idx: int,
    original_forward: Callable | None = None,
    *,
    replacement: torch.Tensor | None = None,
) -> Callable:
    """Build a replacement ``forward`` that overrides masked-head queries.

    Args:
        attn_module: The vLLM attention module (Llama/Gemma3/Qwen3/Olmo2).
        masked_heads_local: Head indices to override, in the *local* per-rank
            head index space. Empty list → returned forward delegates to the
            original (no-op patch).
        layer_idx: Layer index (kept for symmetry with attention.py; unused
            here since each replacement forward is bound to one module).
        original_forward: Pre-bound original ``forward`` method. Captured
            from the class to avoid infinite recursion on re-patch.
        replacement: Replacement values for masked-head queries.

            * ``None`` → zero ablation (write 0.0).
            * ``Tensor`` of shape ``(len(masked_heads_local), head_dim)`` →
              mean ablation. Each row is the per-head mean q activation;
              broadcast across the token dimension.

            Must be on the model's device and a float dtype compatible with
            the q tensor at injection time.

    Returns:
        A callable with signature ``(positions, hidden_states, **kwargs) -> Tensor``.
    """
    del layer_idx  # not used; accepted for parity with attention.py

    # No heads on this layer (or none on this rank for TP) → nothing to do.
    # Only resolve the original forward in this branch — for the active path
    # we never call it (the patched forward replicates the full pipeline).
    if not masked_heads_local:
        if original_forward is None:
            original_forward = attn_module.__class__.forward.__get__(attn_module)
        return original_forward

    num_heads: int = attn_module.num_heads  # local per-rank count under TP
    num_kv_heads: int = attn_module.num_kv_heads  # local per-rank count under TP
    head_dim: int = attn_module.head_dim
    q_size: int = attn_module.q_size
    kv_size: int = attn_module.kv_size

    # Bounds-check head indices against the local shard size (catches
    # mis-remapped heads early instead of silently overriding the wrong slice).
    for h in masked_heads_local:
        assert 0 <= h < num_heads, (
            f"masked head {h} out of range for local num_heads={num_heads} "
            f"on attention module {type(attn_module).__name__}"
        )

    # Validate replacement shape if provided. The replacement is broadcast
    # across the token dimension by tensor assignment.
    if replacement is not None:
        expected_shape = (len(masked_heads_local), head_dim)
        assert replacement.shape == expected_shape, (
            f"replacement shape {tuple(replacement.shape)} does not match expected "
            f"{expected_shape} for {len(masked_heads_local)} masked heads × head_dim={head_dim}"
        )

    # QK normalization (architecture-specific) — same detection as attention.py.
    is_olmo2 = False
    try:
        from vllm.model_executor.models.olmo2 import Olmo2Attention

        is_olmo2 = isinstance(attn_module, Olmo2Attention)
    except ImportError:
        pass
    apply_qk_norm = attn_module._apply_qk_norm if is_olmo2 else None
    q_norm = None if is_olmo2 else getattr(attn_module, "q_norm", None)
    k_norm = None if is_olmo2 else getattr(attn_module, "k_norm", None)

    # Snapshot to a tuple so the closure captures an immutable list of indices.
    masked_heads_t = tuple(masked_heads_local)
    # ``replacement_value`` is either a scalar (zero mode) or the precomputed
    # tensor (mean mode). Tensor assignment broadcasts both correctly.
    replacement_value: float | torch.Tensor = 0.0 if replacement is None else replacement

    def ablation_forward(positions, hidden_states, **kwargs):
        if kwargs.get("has_images", False):
            raise NotImplementedError("Gemma3 VLM ablation is not supported; this implementation is text-only")

        qkv, _ = attn_module.qkv_proj(hidden_states)
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Override masked-head queries BEFORE QK-norm and rotary so subsequent
        # transforms apply to the replacement (matches nolima_ablation hooks
        # which run on q_proj output). For zero ablation this is mathematically
        # equivalent to writing zeros after rotary since RMSNorm(0)=0, rot(0)=0.
        q_3d = q.view(-1, num_heads, head_dim)
        q_3d[:, masked_heads_t, :] = replacement_value
        q = q_3d.view(-1, num_heads * head_dim)

        if apply_qk_norm is not None:
            # Olmo2/Olmo3 path: full-tensor RMSNorm with TP gather/split.
            q, k = apply_qk_norm(q, k)
        else:
            # Gemma3 / Qwen3 path: per-head RMSNorm.
            if q_norm is not None:
                q = q.unflatten(-1, (num_heads, head_dim))
                q = q_norm(q)
                q = q.flatten(-2, -1)
            if k_norm is not None:
                k = k.unflatten(-1, (num_kv_heads, head_dim))
                k = k_norm(k)
                k = k.flatten(-2, -1)

        q, k = attn_module.rotary_emb(positions, q, k)

        # Native paged attention — vLLM owns the KV cache here, so we keep
        # full continuous-batching and scheduler benefits.
        attn_output = attn_module.attn(q, k, v)
        output, _ = attn_module.o_proj(attn_output)
        return output

    return ablation_forward


def patch_model_for_ablation(
    model,
    heads_per_layer: dict[int, list[int]],
    *,
    replacements_per_layer: dict[int, torch.Tensor] | None = None,
) -> None:
    """Walk the model and replace each attention layer's forward.

    Idempotent: already-patched layers are skipped. Layers with no masked
    heads on this rank receive a no-op patch (delegates to original) so that
    :func:`unpatch_model_for_ablation` can cleanly restore everything.

    Args:
        model: The vLLM-loaded model.
        heads_per_layer: Mapping ``layer_idx -> list of local head indices``.
            For TP>1 these MUST already be remapped to the local rank's index
            space (use :func:`locos_eval.rpc_ops._remap_heads_for_tp`).
        replacements_per_layer: Optional mapping ``layer_idx -> Tensor`` of
            shape ``(len(heads_per_layer[layer_idx]), head_dim)`` — the per-
            head mean q activations for mean ablation. ``None`` → zero
            ablation (default). Must be on the model's device.
    """
    supported_classes = _get_supported_attention_classes()
    layers = get_decoder_layers(model)

    patched = 0
    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        if not isinstance(attn, supported_classes):
            continue
        if hasattr(attn, "_ablation_patched"):
            continue
        local_heads = heads_per_layer.get(layer_idx, [])
        replacement = None
        if replacements_per_layer is not None:
            replacement = replacements_per_layer.get(layer_idx)
            # Sanity: a layer with masked heads MUST have a replacement under
            # mean mode; mismatched calibration is a setup bug, not a runtime
            # condition we should silently absorb.
            if local_heads and replacement is None:
                raise ValueError(
                    f"Layer {layer_idx} has {len(local_heads)} masked heads but no "
                    f"replacement tensor in replacements_per_layer; calibration likely missed it."
                )

        original = type(attn).forward.__get__(attn)
        attn._ablation_original_forward = original
        new_forward = build_ablation_attention_forward(
            attn,
            masked_heads_local=local_heads,
            layer_idx=layer_idx,
            original_forward=original,
            replacement=replacement,
        )
        attn.forward = new_forward
        attn._ablation_patched = True
        patched += 1

    if patched == 0:
        raise RuntimeError(
            f"No supported attention layers found in {type(model).__name__}. "
            f"Supported: {[c.__name__ for c in supported_classes]}"
        )


def unpatch_model_for_ablation(model) -> None:
    """Reverse all :func:`patch_model_for_ablation` monkey-patches."""
    for module in model.modules():
        if not hasattr(module, "_ablation_patched"):
            continue
        if hasattr(module, "_ablation_original_forward"):
            module.forward = module._ablation_original_forward
            del module._ablation_original_forward
        del module._ablation_patched


# ---------------------------------------------------------------------------
# Mean-activation calibration
# ---------------------------------------------------------------------------


class _QCaptureState:
    """Per-layer running accumulator for q-portion of qkv_proj output.

    Stored as a regular attribute on each :class:`QKVParallelLinear` module to
    survive across many forward passes during calibration. ``sum`` is kept in
    float64 for numerical stability across long calibration corpora.
    """

    __slots__ = ("count", "head_dim", "num_heads", "q_size", "sum")

    def __init__(self, num_heads: int, head_dim: int, q_size: int) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_size = q_size
        self.sum: torch.Tensor | None = None
        self.count: int = 0


def _make_q_capture_hook(state: _QCaptureState):
    """Build a forward hook that accumulates q-portion of qkv_proj output."""

    def hook(module, args, output):
        del module, args
        # vLLM's QKVParallelLinear returns (qkv_tensor, bias)
        qkv = output[0] if isinstance(output, tuple) else output
        q = qkv[..., : state.q_size]
        # Flatten any leading batch/seq dims to a single token axis. q can be
        # 2D ``[N, q_size]`` (vLLM's typical packed layout) or 3D
        # ``[B, T, q_size]`` (rare, but be safe).
        q_flat = q.reshape(-1, state.num_heads, state.head_dim).detach()
        contrib = q_flat.sum(dim=0).to(torch.float64)
        if state.sum is None:
            state.sum = contrib
        else:
            state.sum.add_(contrib)
        state.count += q_flat.shape[0]

    return hook


def install_q_capture_hooks(
    model,
    layers_to_capture: list[int],
) -> tuple[list, dict[int, _QCaptureState]]:
    """Install forward hooks on ``qkv_proj`` for the given layers.

    Captures the q-portion of each forward call into per-layer running sums
    (with token counts) so the caller can compute means after running an
    arbitrary set of calibration prompts through the model.

    Args:
        model: vLLM-loaded model.
        layers_to_capture: Decoder layer indices that need calibration. Only
            layers with at least one masked head need to appear here.

    Returns:
        ``(handles, states)`` where ``handles`` is a list of removable hook
        handles (call ``.remove()`` on each when calibration finishes) and
        ``states`` maps ``layer_idx`` to the per-layer ``_QCaptureState``.
    """
    decoder_layers = get_decoder_layers(model)
    handles: list = []
    states: dict[int, _QCaptureState] = {}
    for layer_idx in layers_to_capture:
        attn = decoder_layers[layer_idx].self_attn
        state = _QCaptureState(
            num_heads=attn.num_heads,
            head_dim=attn.head_dim,
            q_size=attn.q_size,
        )
        states[layer_idx] = state
        handle = attn.qkv_proj.register_forward_hook(_make_q_capture_hook(state))
        handles.append(handle)
    return handles, states


def finalize_q_capture_means(
    states: dict[int, _QCaptureState],
    target_dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    """Compute per-layer mean q activations from accumulator states.

    Args:
        states: Output of :func:`install_q_capture_hooks` (after calibration
            forwards have finished).
        target_dtype: Cast the float64 accumulator means to this dtype before
            returning. Should match the model dtype (e.g. ``torch.bfloat16``)
            so the means can be assigned directly into q at injection time.

    Returns:
        ``layer_idx -> Tensor`` of shape ``(num_heads_local, head_dim)``.
        Layers that received zero forwards are omitted (caller should treat
        their absence as a calibration failure).
    """
    means: dict[int, torch.Tensor] = {}
    for layer_idx, state in states.items():
        if state.count == 0 or state.sum is None:
            continue
        mean = state.sum / state.count
        means[layer_idx] = mean.to(target_dtype)
    return means


def select_replacements_for_masked_heads(
    means_per_layer: dict[int, torch.Tensor],
    heads_per_layer: dict[int, list[int]],
) -> dict[int, torch.Tensor]:
    """Slice per-layer full-head means down to the masked-head subset.

    The patched forward expects a tensor of shape ``(M, head_dim)`` per layer,
    where ``M`` is the number of masked heads on this rank. Calibration
    captures the *full* per-rank q activation tensor of shape
    ``(num_heads_local, head_dim)``; this helper picks out the rows
    corresponding to ``heads_per_layer[layer_idx]``.
    """
    replacements: dict[int, torch.Tensor] = {}
    for layer_idx, masked_heads in heads_per_layer.items():
        if not masked_heads:
            continue
        full = means_per_layer.get(layer_idx)
        if full is None:
            raise ValueError(
                f"Layer {layer_idx} has {len(masked_heads)} masked heads but no "
                f"calibrated mean tensor — calibration likely did not visit this layer."
            )
        # Index by the local head indices.
        idx = torch.tensor(masked_heads, dtype=torch.long, device=full.device)
        replacements[layer_idx] = full.index_select(0, idx).contiguous()
    return replacements


def calibrate_mean_q_activations(
    llm,
    prompts: list[str],
    layers_to_capture: list[int],
    target_dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    """Drive an end-to-end mean-q calibration pass on a single-GPU vLLM LLM.

    Installs capture hooks on ``qkv_proj`` for each requested layer, runs
    ``llm.generate(prompts, max_tokens=1)`` so vLLM does its native prefill
    (where the bulk of token positions are seen), then computes the means
    and removes the hooks.

    Args:
        llm: vLLM ``LLM`` instance (TP=1; for TP>1 use the matching RPC ops).
        prompts: Calibration prompts (already chat-templated by the caller).
        layers_to_capture: Decoder-layer indices needing calibration.
        target_dtype: Dtype to cast computed means to (use the model dtype).

    Returns:
        ``layer_idx -> Tensor`` of shape ``(num_heads_local, head_dim)``.
    """
    from vllm import SamplingParams

    assert len(prompts) > 0, "calibration prompts must be non-empty"

    model_list = llm.apply_model(lambda m: m)
    assert len(model_list) == 1, f"Expected 1 model from apply_model, got {len(model_list)}"
    [model] = model_list

    handles, states = install_q_capture_hooks(model, layers_to_capture)
    try:
        # max_tokens=1 keeps the run short while ensuring vLLM still does the
        # full prefill (which is where most tokens appear).
        params = SamplingParams(max_tokens=1, temperature=0.0)
        llm.generate(prompts, params, use_tqdm=False)
    finally:
        for h in handles:
            h.remove()

    means = finalize_q_capture_means(states, target_dtype)
    missing = [layer_idx for layer_idx in layers_to_capture if layer_idx not in means]
    if missing:
        raise RuntimeError(
            f"Calibration finished but layers {missing} received zero forwards — "
            f"check that the calibration prompts actually reach those layers "
            f"(e.g. prompt list non-empty, model fully loaded)."
        )
    return means
