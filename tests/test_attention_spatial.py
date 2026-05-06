"""Unit tests for locos/detectors/attention_spatial.py.

Verifies the attention-only baseline (drop OV projection, keep spatial contrast):

  phi_{t,j} = alpha_{t,j}
  Phi+ = sum_{j in needle} alpha_j
  Phi- = (needle_len / off_needle_len) * sum_{j not in needle} alpha_j
  S    = Phi+ - Phi-

No GPU required -- all tensors are CPU.
"""

import numpy as np
import torch

from locos.detectors.attention_spatial import (
    compute_attention_spatial_per_step,
)


class TestAttentionSpatialKernel:
    def test_pure_needle_attention(self):
        """All mass on needle positions => Phi+ = 1, Phi- = 0."""
        num_heads = 2
        key_len = 10
        needle_start, needle_end = 3, 6  # needle_len = 3

        attn = torch.zeros(num_heads, key_len)
        attn[:, needle_start:needle_end] = 1.0 / (needle_end - needle_start)
        # row sums to 1.0

        phi_needle, phi_off = compute_attention_spatial_per_step(
            attn_weights=attn,
            needle_start=needle_start,
            needle_end=needle_end,
        )

        np.testing.assert_allclose(phi_needle, np.ones(num_heads), atol=1e-6)
        np.testing.assert_allclose(phi_off, np.zeros(num_heads), atol=1e-6)

    def test_pure_off_needle_attention(self):
        """All mass off-needle => Phi+ = 0; Phi- = (needle_len/off_len) * 1.0."""
        num_heads = 1
        key_len = 10
        needle_start, needle_end = 3, 6  # needle_len = 3, off_len = 7

        attn = torch.zeros(num_heads, key_len)
        # Place 1.0 total mass uniformly over off-needle positions
        off_positions = [i for i in range(key_len) if not (needle_start <= i < needle_end)]
        attn[:, off_positions] = 1.0 / len(off_positions)

        phi_needle, phi_off = compute_attention_spatial_per_step(
            attn_weights=attn,
            needle_start=needle_start,
            needle_end=needle_end,
        )

        np.testing.assert_allclose(phi_needle, np.zeros(num_heads), atol=1e-6)
        # phi_off = 1.0 * (needle_len / off_len) = 3/7
        np.testing.assert_allclose(phi_off, np.full(num_heads, 3.0 / 7.0), atol=1e-6)

    def test_uniform_attention_zero_score(self):
        """Uniform attention => S = Phi+ - Phi- = 0 after rescaling."""
        num_heads = 4
        key_len = 20
        needle_start, needle_end = 5, 9

        attn = torch.full((num_heads, key_len), 1.0 / key_len)

        phi_needle, phi_off = compute_attention_spatial_per_step(
            attn_weights=attn,
            needle_start=needle_start,
            needle_end=needle_end,
        )

        # Phi+ = needle_len/key_len, Phi- = (needle_len/off_len) * (off_len/key_len) = needle_len/key_len
        np.testing.assert_allclose(phi_needle, phi_off, atol=1e-6)
        np.testing.assert_allclose(phi_needle - phi_off, np.zeros(num_heads), atol=1e-6)

    def test_per_head_independence(self):
        """Each head's score depends only on its own row."""
        num_heads = 3
        key_len = 8
        needle_start, needle_end = 2, 5  # needle_len = 3, off_len = 5

        attn = torch.zeros(num_heads, key_len)
        # head 0: pure needle
        attn[0, needle_start:needle_end] = 1.0 / 3
        # head 1: pure off-needle
        off = [i for i in range(key_len) if not (needle_start <= i < needle_end)]
        attn[1, off] = 1.0 / 5
        # head 2: half-half
        attn[2, needle_start:needle_end] = 0.5 / 3
        attn[2, off] = 0.5 / 5

        phi_needle, phi_off = compute_attention_spatial_per_step(
            attn_weights=attn,
            needle_start=needle_start,
            needle_end=needle_end,
        )

        np.testing.assert_allclose(phi_needle[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(phi_off[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(phi_needle[1], 0.0, atol=1e-6)
        np.testing.assert_allclose(phi_off[1], 1.0 * 3 / 5, atol=1e-6)
        np.testing.assert_allclose(phi_needle[2], 0.5, atol=1e-6)
        np.testing.assert_allclose(phi_off[2], 0.5 * 3 / 5, atol=1e-6)

    def test_returns_numpy_float32(self):
        """Output dtype matches logit_contrib for downstream compatibility."""
        attn = torch.rand(2, 10)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        phi_needle, phi_off = compute_attention_spatial_per_step(attn_weights=attn, needle_start=2, needle_end=5)
        assert isinstance(phi_needle, np.ndarray)
        assert isinstance(phi_off, np.ndarray)
        assert phi_needle.dtype == np.float32
        assert phi_off.dtype == np.float32


class TestSingleTrialDataclass:
    """Smoke test for the trial-result dataclass shape.

    Real model behaviour is exercised on GPU servers (see deploy/jobs/).
    """

    def test_dataclass_shape(self):
        from locos.detectors.attention_spatial import (
            AttentionSpatialTrialResult,
        )

        num_layers, num_heads = 2, 3
        result = AttentionSpatialTrialResult(
            S_tau=np.zeros((num_layers, num_heads), dtype=np.float32),
            L_plus=np.zeros((num_layers, num_heads), dtype=np.float32),
            L_minus=np.zeros((num_layers, num_heads), dtype=np.float32),
            generated_text="",
            num_answer_steps=0,
            num_total_steps=0,
        )
        assert result.S_tau.shape == (num_layers, num_heads)
        assert result.L_plus.shape == (num_layers, num_heads)
        assert result.L_minus.shape == (num_layers, num_heads)
        assert result.num_answer_steps == 0
        assert result.num_total_steps == 0
        assert result.generated_text == ""
