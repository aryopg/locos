from unittest.mock import MagicMock, patch

import torch

from locos_eval.attention import build_decore_attention_forward, unpatch_single_layer
from locos_eval.state import DeCoreState


def make_mock_llama_attn(num_heads=8, num_kv_heads=8, head_dim=64):
    attn = MagicMock()
    attn.num_heads = num_heads
    attn.num_kv_heads = num_kv_heads
    attn.head_dim = head_dim
    attn.q_size = num_heads * head_dim
    attn.kv_size = num_kv_heads * head_dim
    attn.scaling = head_dim**-0.5
    attn.qkv_proj = MagicMock()
    attn.rotary_emb = MagicMock()
    attn.o_proj = MagicMock()
    return attn


def _setup_mock_attn(attn, num_heads, num_kv_heads, head_dim, num_tokens):
    """Wire up mock returns for qkv_proj, rotary_emb, o_proj.
    Returns the (q, k, v) tensors that will be used."""
    q = torch.randn(num_tokens, num_heads * head_dim)
    k = torch.randn(num_tokens, num_kv_heads * head_dim)
    v = torch.randn(num_tokens, num_kv_heads * head_dim)
    qkv = torch.cat([q, k, v], dim=-1)
    attn.qkv_proj.return_value = (qkv, None)
    attn.rotary_emb.return_value = (q.clone(), k.clone())
    # o_proj as identity (pass-through) so we can reason about outputs
    attn.o_proj.side_effect = lambda x: (x, None)
    return q, k, v


# --- Routing tests ---


def test_inactive_state_calls_original_forward():
    """When state.active is False, the original forward is called unchanged."""
    state = DeCoreState()
    state.active = False

    positions = torch.zeros(2, dtype=torch.long)
    hidden = torch.randn(2, 512)
    expected_output = torch.randn(2, 512)

    original_forward = MagicMock(return_value=expected_output)
    attn = make_mock_llama_attn()

    forward_fn = build_decore_attention_forward(attn, state, 0, original_forward=original_forward)
    result = forward_fn(positions, hidden)

    original_forward.assert_called_once_with(positions, hidden)
    assert torch.equal(result, expected_output)


def test_active_state_does_not_call_original_forward():
    """When state.active is True, original_forward should never be called."""
    state = DeCoreState()
    state.set_retrieval_heads([])
    state.active = True
    state.masked_pass_active = False

    original_forward = MagicMock()
    num_heads, head_dim, num_tokens = 4, 8, 2
    attn = make_mock_llama_attn(num_heads=num_heads, num_kv_heads=num_heads, head_dim=head_dim)
    _setup_mock_attn(attn, num_heads, num_heads, head_dim, num_tokens)

    forward_fn = build_decore_attention_forward(attn, state, 0, original_forward=original_forward)

    with patch(
        "locos_eval.attention.F.scaled_dot_product_attention", side_effect=lambda q, k, v, **kw: torch.zeros_like(q)
    ):
        forward_fn(torch.zeros(num_tokens, dtype=torch.long), torch.randn(num_tokens, 512))

    original_forward.assert_not_called()


# --- KV cache routing ---


def test_base_pass_writes_to_base_kv_only():
    """Base pass should populate _base_kv and leave _masked_kv empty."""
    state = DeCoreState()
    state.set_retrieval_heads([(0, 2)])
    state.active = True
    state.masked_pass_active = False

    num_heads, head_dim, num_tokens = 8, 64, 3
    attn = make_mock_llama_attn(num_heads=num_heads)
    _setup_mock_attn(attn, num_heads, num_heads, head_dim, num_tokens)

    forward_fn = build_decore_attention_forward(attn, state, 0, original_forward=MagicMock())

    with patch(
        "locos_eval.attention.F.scaled_dot_product_attention", side_effect=lambda q, k, v, **kw: torch.zeros_like(q)
    ):
        forward_fn(torch.zeros(num_tokens, dtype=torch.long), torch.randn(num_tokens, 512))

    bk, _ = state.get_base_kv(layer=0)
    assert bk is not None and bk.shape[0] == num_tokens
    assert state.get_masked_kv(layer=0) == (None, None)


def test_masked_pass_writes_to_masked_kv_only():
    """Masked pass should populate _masked_kv and leave _base_kv empty."""
    state = DeCoreState()
    state.set_retrieval_heads([])
    state.active = True
    state.masked_pass_active = True

    num_heads, head_dim, num_tokens = 8, 64, 3
    attn = make_mock_llama_attn(num_heads=num_heads)
    _setup_mock_attn(attn, num_heads, num_heads, head_dim, num_tokens)

    forward_fn = build_decore_attention_forward(attn, state, 0, original_forward=MagicMock())

    with patch(
        "locos_eval.attention.F.scaled_dot_product_attention", side_effect=lambda q, k, v, **kw: torch.zeros_like(q)
    ):
        forward_fn(torch.zeros(num_tokens, dtype=torch.long), torch.randn(num_tokens, 512))

    mk, _ = state.get_masked_kv(layer=0)
    assert mk is not None and mk.shape[0] == num_tokens
    assert state.get_base_kv(layer=0) == (None, None)


# --- Query zeroing ---


def test_masked_pass_zeros_specified_heads():
    """In masked mode, query states for retrieval heads should be zeroed."""
    state = DeCoreState()
    state.set_retrieval_heads([(0, 2), (0, 5)])
    state.active = True
    state.masked_pass_active = True

    num_heads, head_dim, num_tokens = 8, 64, 3
    attn = make_mock_llama_attn(num_heads=num_heads)
    _setup_mock_attn(attn, num_heads, num_heads, head_dim, num_tokens)

    captured_queries = []

    def fake_sdpa(q, k, v, **kwargs):
        captured_queries.append(q.clone())
        return torch.zeros_like(q)

    forward_fn = build_decore_attention_forward(attn, state, 0, original_forward=MagicMock())

    with patch("locos_eval.attention.F.scaled_dot_product_attention", fake_sdpa):
        forward_fn(torch.zeros(num_tokens, dtype=torch.long), torch.randn(num_tokens, 512))

    q_used = captured_queries[0]  # [B=1, heads, new_tokens, head_dim]
    # Masked heads 2 and 5 should be zero
    assert q_used[0, 2, :, :].abs().max().item() == 0.0
    assert q_used[0, 5, :, :].abs().max().item() == 0.0
    # Non-masked heads should be non-zero
    assert q_used[0, 0, :, :].abs().max().item() > 0.0
    assert q_used[0, 3, :, :].abs().max().item() > 0.0


def test_base_pass_does_not_zero_any_heads():
    """Base pass should preserve ALL query heads — none zeroed."""
    state = DeCoreState()
    state.set_retrieval_heads([(0, 2), (0, 5)])  # same heads as masked test
    state.active = True
    state.masked_pass_active = False  # base pass

    num_heads, head_dim, num_tokens = 8, 64, 3
    attn = make_mock_llama_attn(num_heads=num_heads)
    _setup_mock_attn(attn, num_heads, num_heads, head_dim, num_tokens)

    captured_queries = []

    def fake_sdpa(q, k, v, **kwargs):
        captured_queries.append(q.clone())
        return torch.zeros_like(q)

    forward_fn = build_decore_attention_forward(attn, state, 0, original_forward=MagicMock())

    with patch("locos_eval.attention.F.scaled_dot_product_attention", fake_sdpa):
        forward_fn(torch.zeros(num_tokens, dtype=torch.long), torch.randn(num_tokens, 512))

    q_used = captured_queries[0]
    # ALL heads should be non-zero (including heads 2 and 5)
    for h in range(num_heads):
        assert q_used[0, h, :, :].abs().max().item() > 0.0, f"Head {h} should not be zeroed in base pass"


def test_masked_pass_does_not_zero_heads_in_other_layers():
    """Retrieval heads for layer 0 should not affect layer 1."""
    state = DeCoreState()
    state.set_retrieval_heads([(0, 2)])  # only layer 0
    state.active = True
    state.masked_pass_active = True

    num_heads, head_dim, num_tokens = 8, 64, 3
    attn = make_mock_llama_attn(num_heads=num_heads)
    _setup_mock_attn(attn, num_heads, num_heads, head_dim, num_tokens)

    captured_queries = []

    def fake_sdpa(q, k, v, **kwargs):
        captured_queries.append(q.clone())
        return torch.zeros_like(q)

    forward_fn = build_decore_attention_forward(attn, state, 1, original_forward=MagicMock())  # layer 1!

    with patch("locos_eval.attention.F.scaled_dot_product_attention", fake_sdpa):
        forward_fn(torch.zeros(num_tokens, dtype=torch.long), torch.randn(num_tokens, 512))

    q_used = captured_queries[0]
    # Head 2 should NOT be zeroed for layer 1
    assert q_used[0, 2, :, :].abs().max().item() > 0.0


# --- GQA ---


def test_gqa_expansion_num_kv_heads_less_than_num_heads():
    """With GQA (e.g., 8 heads, 2 kv_heads), K/V are manually expanded to match
    Hq, then passed to SDPA in 4D [B=1, H, L, D] contiguous layout so Flash
    Attention dispatches reliably. We do NOT use enable_gqa=True because at
    long contexts with asymmetric head counts the dispatcher falls back to
    the math kernel and materialises a full attention matrix (~40 GiB at 32K
    prefill on Qwen3-14B), OOMing the GPU."""
    state = DeCoreState()
    state.set_retrieval_heads([])
    state.active = True
    state.masked_pass_active = False

    num_heads, num_kv_heads, head_dim, num_tokens = 8, 2, 16, 2
    attn = make_mock_llama_attn(num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim)
    _setup_mock_attn(attn, num_heads, num_kv_heads, head_dim, num_tokens)

    captured_kv = {}

    def capture_sdpa(q, k, v, **kwargs):
        captured_kv["q"] = q.clone()
        captured_kv["k"] = k.clone()
        captured_kv["v"] = v.clone()
        captured_kv["kwargs"] = dict(kwargs)
        return torch.zeros_like(q)

    forward_fn = build_decore_attention_forward(attn, state, 0, original_forward=MagicMock())

    with patch("locos_eval.attention.F.scaled_dot_product_attention", capture_sdpa):
        forward_fn(torch.zeros(num_tokens, dtype=torch.long), torch.randn(num_tokens, 512))

    # 4D [B=1, H, L, D] layout, Q/K/V at full head count, contiguous
    assert captured_kv["q"].shape == (1, num_heads, num_tokens, head_dim)
    assert captured_kv["k"].shape == (1, num_heads, num_tokens, head_dim)
    assert captured_kv["v"].shape == (1, num_heads, num_tokens, head_dim)
    assert captured_kv["q"].is_contiguous()
    assert captured_kv["k"].is_contiguous()
    assert captured_kv["v"].is_contiguous()
    # enable_gqa must NOT be passed (would force math-backend fallback at long ctx)
    assert "enable_gqa" not in captured_kv["kwargs"]

    # GQA: each kv_head is repeated `num_heads // num_kv_heads` times
    # So k[0,0] == k[0,1] == k[0,2] == k[0,3] (first kv_head group)
    # and k[0,4] == k[0,5] == k[0,6] == k[0,7] (second kv_head group)
    groups = num_heads // num_kv_heads  # 4
    for g in range(num_kv_heads):
        for i in range(1, groups):
            assert torch.equal(
                captured_kv["k"][0, g * groups + i], captured_kv["k"][0, g * groups]
            ), f"GQA group {g}: head {g * groups + i} should equal head {g * groups}"


def test_decode_step_skips_manual_gqa_expansion():
    """At decode (num_new=1) the math kernel scores tensor is tiny ([Hq, 1, seq],
    a few MB), so we skip the manual K/V expansion and pass K/V at the kv_head
    count with enable_gqa=True — saves ~Hq/Hkv× allocator churn per layer per
    decode token in long-context generation."""
    state = DeCoreState()
    state.set_retrieval_heads([])
    state.active = True
    state.masked_pass_active = False

    num_heads, num_kv_heads, head_dim = 8, 2, 16
    # Pre-populate KV cache so the decode step has something to attend over
    # (mimics the post-prefill state of a real generation loop).
    seq_len_in_cache = 5
    state.update_base_kv(
        layer=0,
        k=torch.randn(seq_len_in_cache, num_kv_heads, head_dim),
        v=torch.randn(seq_len_in_cache, num_kv_heads, head_dim),
    )

    num_tokens = 1  # decode step
    attn = make_mock_llama_attn(num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim)
    _setup_mock_attn(attn, num_heads, num_kv_heads, head_dim, num_tokens)

    captured_kv = {}

    def capture_sdpa(q, k, v, **kwargs):
        captured_kv["q"] = q.clone()
        captured_kv["k"] = k.clone()
        captured_kv["v"] = v.clone()
        captured_kv["kwargs"] = dict(kwargs)
        return torch.zeros_like(q)

    forward_fn = build_decore_attention_forward(attn, state, 0, original_forward=MagicMock())

    with patch("locos_eval.attention.F.scaled_dot_product_attention", capture_sdpa):
        forward_fn(torch.zeros(num_tokens, dtype=torch.long), torch.randn(num_tokens, 512))

    # Q is at full head count; K/V stay at kv_head count (no manual expansion).
    # Sequence length is the new token plus the pre-populated cache.
    expected_seq = seq_len_in_cache + num_tokens
    assert captured_kv["q"].shape == (1, num_heads, num_tokens, head_dim)
    assert captured_kv["k"].shape == (1, num_kv_heads, expected_seq, head_dim)
    assert captured_kv["v"].shape == (1, num_kv_heads, expected_seq, head_dim)
    # SDPA handles the head broadcast itself
    assert captured_kv["kwargs"].get("enable_gqa") is True
    assert captured_kv["kwargs"].get("is_causal") is False


# --- SDPA correctness (no mock) ---


def test_single_token_attention_output_equals_value():
    """With 1 query token and 1 KV token, SDPA output = V (softmax of single score is 1.0).
    This tests the full attention pipeline without mocking SDPA."""
    state = DeCoreState()
    state.set_retrieval_heads([])
    state.active = True
    state.masked_pass_active = False

    num_heads, head_dim = 2, 4
    attn = make_mock_llama_attn(num_heads=num_heads, num_kv_heads=num_heads, head_dim=head_dim)

    # Craft known Q, K, V
    q = torch.ones(1, num_heads * head_dim)
    k = torch.ones(1, num_heads * head_dim)
    v_val = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unsqueeze(0)  # [1, 8]
    qkv = torch.cat([q, k, v_val], dim=-1)

    attn.qkv_proj.return_value = (qkv, None)
    attn.rotary_emb.return_value = (q.clone(), k.clone())
    # o_proj is identity
    attn.o_proj.side_effect = lambda x: (x, None)

    forward_fn = build_decore_attention_forward(attn, state, 0, original_forward=MagicMock())
    # No mock on SDPA — use the real one
    output = forward_fn(torch.zeros(1, dtype=torch.long), torch.randn(1, 512))

    # With 1 token: attention output should equal V (softmax of single value = 1)
    assert torch.allclose(output, v_val, atol=1e-5), f"Expected output ≈ V={v_val}, got {output}"


# --- is_causal flag ---


def test_is_causal_true_for_multi_token_prefill():
    """Multi-token input should use is_causal=True."""
    state = DeCoreState()
    state.set_retrieval_heads([])
    state.active = True
    state.masked_pass_active = False

    num_heads, head_dim = 2, 4
    num_tokens = 3  # multi-token
    attn = make_mock_llama_attn(num_heads=num_heads, num_kv_heads=num_heads, head_dim=head_dim)
    _setup_mock_attn(attn, num_heads, num_heads, head_dim, num_tokens)

    captured_kwargs = {}

    def capture_sdpa(q, k, v, **kwargs):
        captured_kwargs.update(kwargs)
        return torch.zeros_like(q)

    forward_fn = build_decore_attention_forward(attn, state, 0, original_forward=MagicMock())

    with patch("locos_eval.attention.F.scaled_dot_product_attention", capture_sdpa):
        forward_fn(torch.zeros(num_tokens, dtype=torch.long), torch.randn(num_tokens, 512))

    assert captured_kwargs["is_causal"] is True


def test_is_causal_false_for_single_token_decode():
    """Single-token input (decode step) should use is_causal=False."""
    state = DeCoreState()
    state.set_retrieval_heads([])
    state.active = True
    state.masked_pass_active = False

    num_heads, head_dim = 2, 4
    num_tokens = 1  # single token
    attn = make_mock_llama_attn(num_heads=num_heads, num_kv_heads=num_heads, head_dim=head_dim)
    _setup_mock_attn(attn, num_heads, num_heads, head_dim, num_tokens)

    captured_kwargs = {}

    def capture_sdpa(q, k, v, **kwargs):
        captured_kwargs.update(kwargs)
        return torch.zeros_like(q)

    forward_fn = build_decore_attention_forward(attn, state, 0, original_forward=MagicMock())

    with patch("locos_eval.attention.F.scaled_dot_product_attention", capture_sdpa):
        forward_fn(torch.zeros(num_tokens, dtype=torch.long), torch.randn(num_tokens, 512))

    assert captured_kwargs["is_causal"] is False


# --- Unpatching ---


def test_unpatch_restores_original_forward():
    """unpatch_single_layer restores the original forward method."""
    state = DeCoreState()
    state.set_retrieval_heads([(0, 1)])

    attn = make_mock_llama_attn()
    original_forward = attn.forward

    new_fwd = build_decore_attention_forward(attn, state, 0, original_forward=original_forward)
    attn.forward = new_fwd
    attn._decore_patched = True
    attn._decore_original_forward = original_forward

    assert attn._decore_patched is True

    unpatch_single_layer(attn)

    assert not hasattr(attn, "_decore_patched")
    assert attn.forward is original_forward
