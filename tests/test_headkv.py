"""Unit tests for locos/detectors/headkv.py.

Strict HeadKV scoring (no spatial contrast, no rescaling):

  phi_t = sum_{j in needle} alpha_{t,j}     [per anchor step]
  S_tau = max_t phi_t                        [over anchor window]

No GPU required -- CPU tensors only.
"""

import numpy as np
import torch

from locos.detectors.headkv import (
    compute_needle_attention_per_step,
)


class TestHeadKVKernel:
    def test_pure_needle_attention_returns_one(self):
        """All mass on needle => phi_t = 1 for every head."""
        num_heads = 3
        key_len = 10
        needle_start, needle_end = 3, 6  # needle_len = 3

        attn = torch.zeros(num_heads, key_len)
        attn[:, needle_start:needle_end] = 1.0 / 3

        phi = compute_needle_attention_per_step(
            attn_weights=attn,
            needle_start=needle_start,
            needle_end=needle_end,
        )

        np.testing.assert_allclose(phi, np.ones(num_heads), atol=1e-6)

    def test_pure_off_needle_attention_returns_zero(self):
        """All mass off-needle => phi_t = 0."""
        num_heads = 2
        key_len = 8
        needle_start, needle_end = 2, 5

        attn = torch.zeros(num_heads, key_len)
        off = [i for i in range(key_len) if not (needle_start <= i < needle_end)]
        attn[:, off] = 1.0 / len(off)

        phi = compute_needle_attention_per_step(
            attn_weights=attn,
            needle_start=needle_start,
            needle_end=needle_end,
        )

        np.testing.assert_allclose(phi, np.zeros(num_heads), atol=1e-6)

    def test_uniform_attention_returns_needle_fraction(self):
        """Uniform attention => phi_t = needle_len / key_len (no rescaling here, unlike attention_spatial)."""
        num_heads = 4
        key_len = 20
        needle_start, needle_end = 5, 9  # needle_len = 4

        attn = torch.full((num_heads, key_len), 1.0 / key_len)

        phi = compute_needle_attention_per_step(
            attn_weights=attn,
            needle_start=needle_start,
            needle_end=needle_end,
        )

        expected = (needle_end - needle_start) / key_len  # 4/20 = 0.2
        np.testing.assert_allclose(phi, np.full(num_heads, expected), atol=1e-6)

    def test_per_head_independence(self):
        """Each head's score depends only on its own row."""
        num_heads = 3
        key_len = 8
        needle_start, needle_end = 2, 5

        attn = torch.zeros(num_heads, key_len)
        # head 0: full needle mass
        attn[0, needle_start:needle_end] = 1.0 / 3
        # head 1: full off-needle
        off = [i for i in range(key_len) if not (needle_start <= i < needle_end)]
        attn[1, off] = 1.0 / 5
        # head 2: half-half
        attn[2, needle_start:needle_end] = 0.5 / 3
        attn[2, off] = 0.5 / 5

        phi = compute_needle_attention_per_step(
            attn_weights=attn,
            needle_start=needle_start,
            needle_end=needle_end,
        )

        np.testing.assert_allclose(phi[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(phi[1], 0.0, atol=1e-6)
        np.testing.assert_allclose(phi[2], 0.5, atol=1e-6)

    def test_returns_numpy_float32(self):
        """Output dtype matches sibling detectors for downstream compatibility."""
        attn = torch.rand(2, 10)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        phi = compute_needle_attention_per_step(attn_weights=attn, needle_start=2, needle_end=5)
        assert isinstance(phi, np.ndarray)
        assert phi.dtype == np.float32
        assert phi.shape == (2,)


class TestHeadKVTrialResultDataclass:
    """Smoke test for the dataclass shape. Real model behaviour runs on GPU."""

    def test_dataclass_shape(self):
        from locos.detectors.headkv import HeadKVTrialResult

        num_layers, num_heads, anchor_window = 2, 3, 8
        result = HeadKVTrialResult(
            S_tau=np.zeros((num_layers, num_heads), dtype=np.float32),
            per_step_phi=np.zeros((num_layers, num_heads, anchor_window), dtype=np.float32),
            anchor_window=anchor_window,
        )
        assert result.S_tau.shape == (num_layers, num_heads)
        assert result.per_step_phi.shape == (num_layers, num_heads, anchor_window)
        assert result.anchor_window == anchor_window


class TestHeadKVMaxAggregation:
    def test_max_over_anchor_window(self):
        """S_tau = max over anchor window of per-step needle sums."""
        # Build a fake per_step_phi: 2 layers, 3 heads, 4 anchor steps
        per_step_phi = np.array(
            [
                # layer 0
                [
                    [0.1, 0.5, 0.2, 0.3],  # head 0: max = 0.5
                    [0.0, 0.0, 0.0, 0.9],  # head 1: max = 0.9
                    [0.4, 0.4, 0.4, 0.4],  # head 2: max = 0.4
                ],
                # layer 1
                [
                    [0.7, 0.1, 0.1, 0.1],  # head 0: max = 0.7
                    [0.0, 0.0, 0.0, 0.0],  # head 1: max = 0.0
                    [0.2, 0.6, 0.6, 0.2],  # head 2: max = 0.6
                ],
            ],
            dtype=np.float32,
        )

        S_tau = per_step_phi.max(axis=-1)
        expected = np.array([[0.5, 0.9, 0.4], [0.7, 0.0, 0.6]], dtype=np.float32)
        np.testing.assert_allclose(S_tau, expected, atol=1e-6)
