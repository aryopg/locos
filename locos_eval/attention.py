"""Masked-attention forward used by the ablation wrapper.

When state.active is True, ALL attention layers use F.scaled_dot_product_attention
with our own sequential KV caches, bypassing vLLM's paged attention entirely.
The base pass and masked pass differ only in:
  - Whether retrieval-head queries are zeroed
  - Which KV cache (_base_kv vs _masked_kv) is read/written

When state.active is False, the original vLLM attention forward is called
unchanged.
"""

from collections.abc import Callable

import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from locos_eval.state import DeCoreState

# Whitelist of SDPA backends safe at long contexts: Flash and memory-efficient
# attention compute O(N) memory; the math kernel materialises [Hq, n_new, seq]
# scores tensor — at 32K prefill on Qwen3-14B (TP=2) that's a single ~40 GiB
# allocation, immediately OOMing an 80 GB GPU. By passing this whitelist to
# sdpa_kernel(), the dispatcher is forbidden from silently falling back to
# math, so an unsupported shape becomes a loud error instead of a stealth OOM.
_LONG_CTX_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]

# ---------------------------------------------------------------------------
# Supported attention classes and model unwrapping
# ---------------------------------------------------------------------------


def _get_supported_attention_classes() -> tuple:
    """Import and return all vLLM attention classes we can patch.

    Returns a tuple of classes suitable for isinstance() checks.
    """
    classes = []
    try:
        from vllm.model_executor.models.llama import LlamaAttention

        classes.append(LlamaAttention)
    except ImportError:
        pass
    try:
        from vllm.model_executor.models.gemma3 import Gemma3Attention

        classes.append(Gemma3Attention)
    except ImportError:
        pass
    try:
        from vllm.model_executor.models.qwen3 import Qwen3Attention

        classes.append(Qwen3Attention)
    except ImportError:
        pass
    try:
        # Olmo2Attention covers both Olmo 2 and Olmo 3 (Olmo3ForCausalLM
        # is registered to Olmo2ForCausalLM in vLLM's model registry).
        from vllm.model_executor.models.olmo2 import Olmo2Attention

        classes.append(Olmo2Attention)
    except ImportError:
        pass
    if not classes:
        raise ImportError(
            "No supported vLLM attention classes found "
            "(tried LlamaAttention, Gemma3Attention, Qwen3Attention, Olmo2Attention)"
        )
    return tuple(classes)


def get_decoder_layers(model) -> list:
    """Extract the list of decoder layers from a vLLM model.

    Handles both plain causal LM models (model.model.layers) and
    conditional-generation wrappers like Gemma3ForConditionalGeneration
    (model.language_model.model.layers).
    """
    import torch.nn as nn

    # Try direct path first: model.model.layers (LlamaForCausalLM, Gemma3ForCausalLM, etc.)
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module) and hasattr(inner, "layers"):
        return inner.layers

    # Conditional-generation wrapper: model.language_model.model.layers
    lm = getattr(model, "language_model", None)
    if isinstance(lm, nn.Module):
        inner = getattr(lm, "model", None)
        if isinstance(inner, nn.Module) and hasattr(inner, "layers"):
            return inner.layers

    raise AttributeError(
        f"Cannot find decoder layers on {type(model).__name__}. "
        "Expected model.model.layers or model.language_model.model.layers."
    )


def get_lm_head(model):
    """Extract the lm_head (or tied embed_tokens) from a vLLM model.

    Handles plain causal LM models, conditional-generation wrappers,
    and tied-embedding models (e.g. Gemma3) where embed_tokens serves as lm_head.
    """
    import torch.nn as nn

    # Unwrap conditional-generation wrappers
    lm = getattr(model, "language_model", None)
    if not isinstance(lm, nn.Module):
        lm = model

    if hasattr(lm, "lm_head"):
        return lm.lm_head

    # Tied embeddings: some models (Gemma3) use embed_tokens as the output head
    inner = getattr(lm, "model", None)
    if isinstance(inner, nn.Module) and hasattr(inner, "embed_tokens"):
        return inner.embed_tokens

    raise AttributeError(
        f"Cannot find lm_head or embed_tokens on {type(model).__name__}. "
        "Expected model.lm_head, model.language_model.lm_head, or model.model.embed_tokens."
    )


def build_decore_attention_forward(
    attn_module,
    state: DeCoreState,
    layer_idx: int,
    original_forward: Callable | None = None,
) -> Callable:
    """Return a replacement forward function for a LlamaAttention instance.

    Args:
        attn_module: The LlamaAttention instance to wrap.
        state: Shared DeCoreState.
        layer_idx: The layer index (vLLM v0.18 removed layer_idx from attn).
        original_forward: The original (unpatched) forward method.
            Captured from the class to avoid infinite recursion on re-patch.

    Returns:
        A callable with signature (positions, hidden_states) -> Tensor.
    """
    if original_forward is None:
        # Use the class-level method to avoid capturing an already-patched instance method
        original_forward = attn_module.__class__.forward.__get__(attn_module)

    num_heads: int = attn_module.num_heads
    num_kv_heads: int = attn_module.num_kv_heads
    head_dim: int = attn_module.head_dim
    scale: float = attn_module.scaling
    assert num_heads > 0 and num_kv_heads > 0 and head_dim > 0
    assert num_heads % num_kv_heads == 0, f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    kv_groups: int = num_heads // num_kv_heads

    # QK normalization (varies by architecture):
    #   - Olmo2/Olmo3: full-tensor RMSNorm with TP gather/split, encapsulated in _apply_qk_norm.
    #     Detected via isinstance against the actual class (not duck-typed), because test
    #     mocks (MagicMock) auto-create any attribute, defeating getattr-based detection.
    #   - Gemma3 / Qwen3: per-head RMSNorm via q_norm/k_norm modules (no TP gather).
    #   - LLaMA: no QK norm (both attrs absent).
    is_olmo2 = False
    try:
        from vllm.model_executor.models.olmo2 import Olmo2Attention

        is_olmo2 = isinstance(attn_module, Olmo2Attention)
    except ImportError:
        pass
    apply_qk_norm = attn_module._apply_qk_norm if is_olmo2 else None
    q_norm = None if is_olmo2 else getattr(attn_module, "q_norm", None)
    k_norm = None if is_olmo2 else getattr(attn_module, "k_norm", None)

    def decore_forward(positions, hidden_states, **kwargs):
        if not state.active:
            return original_forward(positions, hidden_states, **kwargs)

        # --- DeCoRe active: both passes use SDPA with our KV caches ---

        # 1. QKV projection + rotary embedding
        qkv, _ = attn_module.qkv_proj(hidden_states)
        q, k, v = qkv.split([attn_module.q_size, attn_module.kv_size, attn_module.kv_size], dim=-1)

        # QK normalization (no-op for LLaMA)
        if apply_qk_norm is not None:
            # Olmo2/Olmo3: full-tensor RMSNorm with TP gather/split.
            q, k = apply_qk_norm(q, k)
        else:
            # Gemma3 / Qwen3: per-head RMSNorm.
            if q_norm is not None:
                q = q.unflatten(-1, (num_heads, head_dim))
                q = q_norm(q)
                q = q.flatten(-2, -1)
            if k_norm is not None:
                k = k.unflatten(-1, (num_kv_heads, head_dim))
                k = k_norm(k)
                k = k.flatten(-2, -1)

        q, k = attn_module.rotary_emb(positions, q, k)
        assert q.shape[-1] == num_heads * head_dim, f"Q shape mismatch: {q.shape[-1]} != {num_heads * head_dim}"

        # 2. Reshape K/V to [tokens, kv_heads, head_dim]
        k_new = k.view(-1, num_kv_heads, head_dim)
        v_new = v.view(-1, num_kv_heads, head_dim)

        if state.masked_pass_active:
            # Zero retrieval heads in Q
            masked_heads = state.masked_heads_for_layer(layer_idx)
            if masked_heads:
                q_3d = q.view(-1, num_heads, head_dim)
                q_3d[:, masked_heads, :] = 0.0
                q = q_3d.view(-1, num_heads * head_dim)

            # Append to masked KV cache
            state.update_masked_kv(layer_idx, k_new.detach(), v_new.detach())
            k_full, v_full = state.get_masked_kv(layer_idx)
        else:
            # Append to base KV cache
            state.update_base_kv(layer_idx, k_new.detach(), v_new.detach())
            k_full, v_full = state.get_base_kv(layer_idx)

        assert k_full is not None and v_full is not None, f"KV cache empty for layer {layer_idx}"

        # 3. SDPA in [B=1, H, L, D] layout. Prefill and decode take different
        #    paths because the math-kernel risk is asymmetric:
        #
        #      Prefill (n_new == seq):  the math fallback would materialise a
        #        [Hq, n_new, seq] scores tensor — ~40 GiB at 32K on Qwen3-14B
        #        TP=2. We manually expand K/V to Hq, force Flash/mem-efficient
        #        via sdpa_kernel(), and hard-ban the math kernel.
        #
        #      Decode  (n_new == 1):    the same scores tensor would be
        #        [Hq, 1, seq] — at most a few MB. Manual K/V expansion costs
        #        ~Hq/Hkv × bytes per layer per token (5x for Qwen3) which
        #        dominates allocator churn during long generations. So we
        #        skip the expansion, pass K/V at [Hkv, seq, d], and let SDPA
        #        handle the head broadcast via enable_gqa. Math fallback is
        #        permitted here because the worst case is harmless.
        num_new = q.shape[0]
        is_prefill = num_new > 1
        q_t = q.view(num_new, num_heads, head_dim).permute(1, 0, 2).contiguous().unsqueeze(0)

        if is_prefill:
            seq_len = k_full.shape[0]
            k_exp = k_full.unsqueeze(2).expand(-1, -1, kv_groups, -1).reshape(seq_len, num_heads, head_dim)
            v_exp = v_full.unsqueeze(2).expand(-1, -1, kv_groups, -1).reshape(seq_len, num_heads, head_dim)
            k_t = k_exp.permute(1, 0, 2).contiguous().unsqueeze(0)
            v_t = v_exp.permute(1, 0, 2).contiguous().unsqueeze(0)

            with sdpa_kernel(_LONG_CTX_SDPA_BACKENDS):
                attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, scale=scale)
        else:
            # Decode: [1, Hkv, seq, d] for K/V — SDPA broadcasts heads internally.
            k_t = k_full.permute(1, 0, 2).contiguous().unsqueeze(0)
            v_t = v_full.permute(1, 0, 2).contiguous().unsqueeze(0)

            attn_out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=False, scale=scale, enable_gqa=kv_groups > 1
            )
        attn_out = attn_out.squeeze(0).permute(1, 0, 2).reshape(num_new, num_heads * head_dim)

        # 5. Output projection
        output, _ = attn_module.o_proj(attn_out)
        return output

    return decore_forward


def patch_model_attention_layers(model, state: DeCoreState) -> None:
    """Walk the model's attention layers and monkey-patch their forward methods.

    Safe to call multiple times — already-patched layers are skipped.
    Supports LlamaAttention, Gemma3Attention, Qwen3Attention, and
    conditional-generation wrappers.

    Args:
        model: The vLLM-loaded model (LlamaForCausalLM, Gemma3ForCausalLM,
            Gemma3ForConditionalGeneration, or compatible).
        state: The shared DeCoreState instance.
    """
    supported_classes = _get_supported_attention_classes()
    layers = get_decoder_layers(model)

    patched = 0
    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        if not isinstance(attn, supported_classes):
            continue
        if hasattr(attn, "_decore_patched"):
            continue
        original = type(attn).forward.__get__(attn)
        attn._decore_original_forward = original
        new_forward = build_decore_attention_forward(attn, state, layer_idx=layer_idx, original_forward=original)
        attn.forward = new_forward
        attn._decore_patched = True
        patched += 1

    if patched == 0:
        raise RuntimeError(
            f"No supported attention layers found in {type(model).__name__}. "
            f"Supported: {[c.__name__ for c in supported_classes]}"
        )


def unpatch_single_layer(module) -> None:
    """Restore the original forward on a single attention module."""
    if hasattr(module, "_decore_original_forward"):
        module.forward = module._decore_original_forward
        del module._decore_original_forward
    if hasattr(module, "_decore_patched"):
        del module._decore_patched


def unpatch_model_attention_layers(model) -> None:
    """Reverse all DeCoRe monkey-patches on the model's attention layers."""
    for module in model.modules():
        if hasattr(module, "_decore_patched"):
            unpatch_single_layer(module)
