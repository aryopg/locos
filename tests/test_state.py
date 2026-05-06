import torch

from locos_eval.state import AblationState


def test_default_state_is_inactive():
    state = AblationState()
    assert not state.active
    assert not state.masked_pass_active


def test_set_masked_pass():
    state = AblationState()
    state.masked_pass_active = True
    assert state.masked_pass_active


def test_masked_heads_by_layer_returns_empty_list_for_unknown_layer():
    state = AblationState()
    state.set_retrieval_heads([(0, 1), (0, 3), (2, 5)])
    assert state.masked_heads_for_layer(99) == []
    assert state.masked_heads_for_layer(0) == [1, 3]
    assert state.masked_heads_for_layer(2) == [5]


# --- Base KV cache ---


def test_base_kv_initially_none():
    state = AblationState()
    assert state.get_base_kv(layer=0) == (None, None)


def test_base_kv_update_and_retrieve():
    state = AblationState()
    k = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # [1, 2, 2]
    v = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
    state.update_base_kv(layer=0, k=k, v=v)
    k2, v2 = state.get_base_kv(layer=0)
    assert torch.equal(k2, k)
    assert torch.equal(v2, v)


def test_base_kv_append_preserves_values():
    """Appending to KV cache should concatenate along dim 0, preserving all values."""
    state = AblationState()
    k1 = torch.tensor([[[1.0, 2.0]]])  # [1, 1, 2]
    v1 = torch.tensor([[[3.0, 4.0]]])
    state.update_base_kv(layer=0, k=k1, v=v1)

    k2 = torch.tensor([[[5.0, 6.0]]])
    v2 = torch.tensor([[[7.0, 8.0]]])
    state.update_base_kv(layer=0, k=k2, v=v2)

    k_full, v_full = state.get_base_kv(layer=0)
    assert k_full.shape == (2, 1, 2)
    # First row from k1, second row from k2
    assert torch.equal(k_full[0], k1[0])
    assert torch.equal(k_full[1], k2[0])
    assert torch.equal(v_full[0], v1[0])
    assert torch.equal(v_full[1], v2[0])


# --- Masked KV cache ---


def test_masked_kv_initially_none():
    state = AblationState()
    assert state.get_masked_kv(layer=0) == (None, None)


def test_masked_kv_update_and_retrieve():
    state = AblationState()
    k = torch.zeros(5, 4, 64)
    v = torch.ones(5, 4, 64)
    state.update_masked_kv(layer=0, k=k, v=v)
    k2, v2 = state.get_masked_kv(layer=0)
    assert torch.equal(k2, k)
    assert torch.equal(v2, v)


# --- Copy and reset ---


def test_copy_base_to_masked_kv_values_match():
    state = AblationState()
    k = torch.randn(5, 4, 64)
    v = torch.randn(5, 4, 64)
    state.update_base_kv(layer=0, k=k, v=v)
    state.copy_base_to_masked_kv()
    mk, mv = state.get_masked_kv(layer=0)
    assert torch.equal(mk, k)
    assert torch.equal(mv, v)


def test_copy_base_to_masked_kv_is_independent():
    """After copying, mutating the masked cache should NOT affect the base cache."""
    state = AblationState()
    k = torch.zeros(3, 2, 4)
    v = torch.ones(3, 2, 4)
    state.update_base_kv(layer=0, k=k, v=v)
    state.copy_base_to_masked_kv()

    # Append to masked — should NOT appear in base
    extra_k = torch.full((1, 2, 4), 99.0)
    extra_v = torch.full((1, 2, 4), 99.0)
    state.update_masked_kv(layer=0, k=extra_k, v=extra_v)

    base_k, _ = state.get_base_kv(layer=0)
    masked_k, _ = state.get_masked_kv(layer=0)
    assert base_k.shape[0] == 3  # unchanged
    assert masked_k.shape[0] == 4  # grew by 1


def test_copy_base_to_masked_clears_old_masked():
    """Copy should replace any pre-existing masked cache, not merge."""
    state = AblationState()
    # Pre-populate masked with layer 5
    state.update_masked_kv(layer=5, k=torch.zeros(1, 1, 1), v=torch.zeros(1, 1, 1))
    # Now set base for layer 0 only
    state.update_base_kv(layer=0, k=torch.randn(2, 2, 4), v=torch.randn(2, 2, 4))
    state.copy_base_to_masked_kv()

    # Layer 5 should be gone from masked cache
    assert state.get_masked_kv(layer=5) == (None, None)
    # Layer 0 should be present
    mk, _ = state.get_masked_kv(layer=0)
    assert mk is not None


def test_reset_kv_caches_clears_both():
    state = AblationState()
    state.update_base_kv(layer=0, k=torch.zeros(3, 2, 8), v=torch.zeros(3, 2, 8))
    state.update_masked_kv(layer=1, k=torch.zeros(3, 2, 8), v=torch.zeros(3, 2, 8))
    state.reset_kv_caches()
    assert state.get_base_kv(layer=0) == (None, None)
    assert state.get_masked_kv(layer=1) == (None, None)


def test_layers_are_independent():
    """KV updates to one layer should not affect another."""
    state = AblationState()
    state.update_base_kv(layer=0, k=torch.zeros(2, 1, 4), v=torch.zeros(2, 1, 4))
    state.update_base_kv(layer=1, k=torch.ones(3, 1, 4), v=torch.ones(3, 1, 4))
    k0, _ = state.get_base_kv(layer=0)
    k1, _ = state.get_base_kv(layer=1)
    assert k0.shape[0] == 2
    assert k1.shape[0] == 3
    assert k0.sum().item() == 0.0
    assert k1.sum().item() > 0.0


# --- Pre-allocated KV cache (capacity-based slice-copy path) ---


def test_prealloc_get_returns_only_populated_slice():
    """With set_kv_capacity, get_*_kv returns buf[:seq_len], not the whole buffer."""
    state = AblationState()
    state.set_kv_capacity(16)
    state.update_base_kv(layer=0, k=torch.ones(3, 2, 4), v=torch.ones(3, 2, 4) * 2)

    k, v = state.get_base_kv(layer=0)
    # Only the populated 3 rows are returned
    assert k.shape == (3, 2, 4)
    assert v.shape == (3, 2, 4)
    assert torch.all(k == 1.0)
    assert torch.all(v == 2.0)


def test_prealloc_append_writes_into_existing_buffer():
    """Two consecutive updates share one preallocated buffer (no torch.cat realloc)."""
    state = AblationState()
    state.set_kv_capacity(16)
    state.update_base_kv(layer=0, k=torch.ones(3, 2, 4), v=torch.ones(3, 2, 4))
    # Capture the buffer identity after first write
    buf_k_before = state._base_kv[0][0]
    state.update_base_kv(layer=0, k=torch.full((1, 2, 4), 7.0), v=torch.full((1, 2, 4), 7.0))
    buf_k_after = state._base_kv[0][0]

    # Same underlying storage — no reallocation
    assert buf_k_before.data_ptr() == buf_k_after.data_ptr()

    k, _ = state.get_base_kv(layer=0)
    assert k.shape == (4, 2, 4)
    # Rows 0-2 from the first write, row 3 from the second
    assert torch.all(k[:3] == 1.0)
    assert torch.all(k[3] == 7.0)


def test_prealloc_overflow_raises():
    """Writing past capacity should fail with a clear error."""
    state = AblationState()
    state.set_kv_capacity(4)
    state.update_base_kv(layer=0, k=torch.zeros(3, 1, 2), v=torch.zeros(3, 1, 2))

    import pytest

    with pytest.raises(AssertionError, match="base KV overflow"):
        state.update_base_kv(layer=0, k=torch.zeros(2, 1, 2), v=torch.zeros(2, 1, 2))


def test_prealloc_copy_base_to_masked_is_independent():
    """copy_base_to_masked_kv allocates an independent masked buffer in the prealloc path."""
    state = AblationState()
    state.set_kv_capacity(8)
    state.update_base_kv(layer=0, k=torch.zeros(3, 2, 4), v=torch.ones(3, 2, 4))
    state.copy_base_to_masked_kv()

    # Mutating masked must not affect base
    state.update_masked_kv(layer=0, k=torch.full((1, 2, 4), 99.0), v=torch.full((1, 2, 4), 99.0))

    base_k, _ = state.get_base_kv(layer=0)
    masked_k, _ = state.get_masked_kv(layer=0)
    assert base_k.shape[0] == 3
    assert masked_k.shape[0] == 4
    assert masked_k[3].max().item() == 99.0
    assert base_k.max().item() == 0.0


def test_prealloc_reset_clears_buffers_and_seq_lens():
    """reset_kv_caches drops both buffers and length counters in the prealloc path."""
    state = AblationState()
    state.set_kv_capacity(8)
    state.update_base_kv(layer=0, k=torch.zeros(3, 2, 4), v=torch.zeros(3, 2, 4))
    state.update_masked_kv(layer=1, k=torch.zeros(3, 2, 4), v=torch.zeros(3, 2, 4))
    state.reset_kv_caches()
    assert state.get_base_kv(layer=0) == (None, None)
    assert state.get_masked_kv(layer=1) == (None, None)
    assert state._base_seq_len == {}
    assert state._masked_seq_len == {}
