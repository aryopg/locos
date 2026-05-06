"""Unit tests for the native zero-ablation patching path (locos_eval.ablation).

These tests use mocks rather than a real model — they verify that the
replacement forward (a) calls the model's qkv_proj/rotary/attn/o_proj in the
right order, and (b) zeros queries on the configured retrieval heads (and
ONLY those) before ``self.attn(q, k, v)`` is invoked.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from locos_eval.ablation import (
    build_ablation_attention_forward,
    patch_model_for_ablation,
    unpatch_model_for_ablation,
)


def _make_mock_attn(num_heads=8, num_kv_heads=8, head_dim=4):
    """Mock vLLM attention module shaped like LlamaAttention.

    QK-norm modules are explicitly None — MagicMock auto-creates any attribute,
    so without this the patched forward would call a MagicMock as q_norm and
    contaminate q with a non-Tensor return value.
    """
    attn = MagicMock()
    attn.num_heads = num_heads
    attn.num_kv_heads = num_kv_heads
    attn.head_dim = head_dim
    attn.q_size = num_heads * head_dim
    attn.kv_size = num_kv_heads * head_dim
    attn.q_norm = None
    attn.k_norm = None
    return attn


def _wire_qkv_rotary(attn, q, k, v):
    """Wire qkv_proj → split → rotary → attn → o_proj path with deterministic returns."""
    qkv = torch.cat([q, k, v], dim=-1)
    attn.qkv_proj.return_value = (qkv, None)
    # Identity rotary so q passed into self.attn equals q post-zeroing.
    attn.rotary_emb.side_effect = lambda positions, q_in, k_in: (q_in, k_in)
    # Identity o_proj so the patched forward's return equals self.attn's output.
    attn.o_proj.side_effect = lambda x: (x, None)


def test_no_masked_heads_returns_original_forward():
    """When this layer has no masked heads, the patched forward delegates."""
    attn = _make_mock_attn()
    sentinel = MagicMock(return_value=torch.zeros(1))
    forward_fn = build_ablation_attention_forward(attn, masked_heads_local=[], layer_idx=0, original_forward=sentinel)
    assert forward_fn is sentinel


def test_zeros_only_configured_heads():
    """Only the configured heads' query slices should be zeroed."""
    num_heads, num_kv_heads, head_dim, num_tokens = 4, 4, 8, 3
    attn = _make_mock_attn(num_heads, num_kv_heads, head_dim)

    # All-ones q so we can detect zeroing by inspecting the tensor handed to self.attn.
    q = torch.ones(num_tokens, num_heads * head_dim)
    k = torch.ones(num_tokens, num_kv_heads * head_dim)
    v = torch.ones(num_tokens, num_kv_heads * head_dim)
    _wire_qkv_rotary(attn, q, k, v)

    captured = {}

    def fake_attn(q_in, k_in, v_in):
        captured["q"] = q_in.clone()
        return torch.zeros_like(q_in)

    attn.attn = fake_attn

    masked = [1, 3]
    forward_fn = build_ablation_attention_forward(attn, masked_heads_local=masked, layer_idx=0)
    forward_fn(torch.zeros(num_tokens, dtype=torch.long), torch.randn(num_tokens, 16))

    q_seen = captured["q"].view(num_tokens, num_heads, head_dim)
    for h in range(num_heads):
        if h in masked:
            assert torch.all(q_seen[:, h, :] == 0.0), f"head {h} should be zeroed"
        else:
            assert torch.all(q_seen[:, h, :] == 1.0), f"head {h} should be untouched"


def test_invokes_qkv_rotary_attn_oproj_in_order():
    """Patched forward must respect the canonical attention pipeline order."""
    num_heads, num_kv_heads, head_dim, num_tokens = 2, 2, 4, 1
    attn = _make_mock_attn(num_heads, num_kv_heads, head_dim)
    q = torch.zeros(num_tokens, num_heads * head_dim)
    k = torch.zeros(num_tokens, num_kv_heads * head_dim)
    v = torch.zeros(num_tokens, num_kv_heads * head_dim)
    _wire_qkv_rotary(attn, q, k, v)
    attn.attn = MagicMock(return_value=torch.zeros_like(q))

    forward_fn = build_ablation_attention_forward(attn, masked_heads_local=[0], layer_idx=0)
    forward_fn(torch.zeros(num_tokens, dtype=torch.long), torch.randn(num_tokens, 8))

    attn.qkv_proj.assert_called_once()
    attn.rotary_emb.assert_called_once()
    attn.attn.assert_called_once()
    attn.o_proj.assert_called_once()


def test_out_of_range_head_raises():
    """Local head indices outside [0, num_heads) must be caught at build time."""
    attn = _make_mock_attn(num_heads=4)
    with pytest.raises(AssertionError, match="out of range"):
        build_ablation_attention_forward(attn, masked_heads_local=[7], layer_idx=0)


def test_patch_and_unpatch_roundtrip():
    """patch_model_for_ablation + unpatch_model_for_ablation should restore originals."""
    import torch.nn as nn

    from locos_eval.attention import _get_supported_attention_classes

    supported = _get_supported_attention_classes()
    SupportedClass = supported[0]

    # Build a real instance (skipping __init__) so isinstance check passes.
    # Run nn.Module.__init__ so __setattr__/__delattr__ have _parameters, etc.
    attn = SupportedClass.__new__(SupportedClass)
    nn.Module.__init__(attn)
    # Set the bare minimum attrs build_ablation_attention_forward inspects.
    attn.num_heads = 4
    attn.num_kv_heads = 4
    attn.head_dim = 8
    attn.q_size = 32
    attn.kv_size = 32

    layer = MagicMock()
    layer.self_attn = attn
    model = MagicMock()
    # unpatch_model_for_ablation iterates model.modules() — wire it.
    model.modules.return_value = [attn]

    # Stub get_decoder_layers (it requires nn.Module on model.model, which a
    # MagicMock can't satisfy — but the function is straightforward enough
    # that we can short-circuit it here without losing test value).
    with patch("locos_eval.ablation.get_decoder_layers", return_value=[layer]):
        patch_model_for_ablation(model, heads_per_layer={0: [1]})

    assert hasattr(attn, "_ablation_patched")
    assert hasattr(attn, "_ablation_original_forward")
    # forward is now bound to the closure, not the unbound class method.
    assert attn.forward is not SupportedClass.forward

    unpatch_model_for_ablation(model)
    assert not hasattr(attn, "_ablation_patched")
    assert not hasattr(attn, "_ablation_original_forward")


def test_mean_replacement_writes_means_not_zeros():
    """Under mean mode the patched forward writes the precomputed means, not 0."""
    num_heads, num_kv_heads, head_dim, num_tokens = 4, 4, 8, 2
    attn = _make_mock_attn(num_heads, num_kv_heads, head_dim)
    q = torch.ones(num_tokens, num_heads * head_dim)
    k = torch.ones(num_tokens, num_kv_heads * head_dim)
    v = torch.ones(num_tokens, num_kv_heads * head_dim)
    _wire_qkv_rotary(attn, q, k, v)

    captured = {}

    def fake_attn(q_in, k_in, v_in):
        captured["q"] = q_in.clone()
        return torch.zeros_like(q_in)

    attn.attn = fake_attn

    masked = [1, 3]
    # Each masked head gets a distinct, non-zero mean vector.
    replacement = torch.stack([torch.full((head_dim,), 5.0), torch.full((head_dim,), 7.0)])
    forward_fn = build_ablation_attention_forward(
        attn,
        masked_heads_local=masked,
        layer_idx=0,
        replacement=replacement,
    )
    forward_fn(torch.zeros(num_tokens, dtype=torch.long), torch.randn(num_tokens, 16))

    q_seen = captured["q"].view(num_tokens, num_heads, head_dim)
    # Head 1 → all 5.0; head 3 → all 7.0; heads 0, 2 → unchanged (1.0).
    assert torch.all(q_seen[:, 1, :] == 5.0)
    assert torch.all(q_seen[:, 3, :] == 7.0)
    assert torch.all(q_seen[:, 0, :] == 1.0)
    assert torch.all(q_seen[:, 2, :] == 1.0)


def test_replacement_shape_mismatch_raises():
    """Replacement tensor with wrong row count must error at build time."""
    attn = _make_mock_attn(num_heads=4, num_kv_heads=4, head_dim=8)
    bad = torch.zeros(3, 8)  # 3 rows, but we ask for 2 masked heads
    with pytest.raises(AssertionError, match="replacement shape"):
        build_ablation_attention_forward(
            attn,
            masked_heads_local=[0, 2],
            layer_idx=0,
            replacement=bad,
        )


def test_select_replacements_for_masked_heads():
    """select_replacements_for_masked_heads should pick the right rows."""
    from locos_eval.ablation import select_replacements_for_masked_heads

    means = {
        0: torch.arange(4 * 3, dtype=torch.float32).view(4, 3),  # 4 heads × 3 dim
    }
    out = select_replacements_for_masked_heads(means, {0: [1, 3]})
    expected = torch.stack([means[0][1], means[0][3]])
    assert torch.equal(out[0], expected)


def test_select_replacements_missing_layer_raises():
    """A masked layer with no calibrated mean must raise."""
    from locos_eval.ablation import select_replacements_for_masked_heads

    with pytest.raises(ValueError, match="no calibrated mean"):
        select_replacements_for_masked_heads({}, {2: [0]})


def test_finalize_q_capture_means():
    """finalize_q_capture_means should average sums by token counts and cast dtype."""
    from locos_eval.ablation import _QCaptureState, finalize_q_capture_means

    state = _QCaptureState(num_heads=2, head_dim=3, q_size=6)
    # Pretend we accumulated three tokens worth of all-ones q activations.
    state.sum = torch.ones(2, 3, dtype=torch.float64) * 3.0
    state.count = 3
    means = finalize_q_capture_means({0: state}, target_dtype=torch.float32)
    assert means[0].dtype == torch.float32
    assert torch.all(means[0] == 1.0)


def test_patch_no_supported_layers_raises():
    """If no attention layers are recognisable, patching must error loudly."""
    layer = MagicMock()
    # self_attn is a plain MagicMock, not a registered vLLM Attention class.
    model = MagicMock()
    with (
        patch("locos_eval.ablation.get_decoder_layers", return_value=[layer]),
        pytest.raises(RuntimeError, match="No supported attention layers"),
    ):
        patch_model_for_ablation(model, heads_per_layer={0: [1]})
