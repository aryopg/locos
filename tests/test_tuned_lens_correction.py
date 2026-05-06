"""Tests for tuned-lens translator loading (H3 direct-path bias experiment)."""

import numpy as np
import pytest
import torch


def _make_translators_pt(tmp_path, num_layers: int, hidden_dim: int, *, wrapped: bool = True):
    """Helper: create a fake translators.pt file and return its path."""
    translators = {}
    for layer in range(num_layers):
        translators[layer] = {
            "A": torch.randn(hidden_dim, hidden_dim, dtype=torch.float32),
            "b": torch.randn(hidden_dim, dtype=torch.float32),
        }
    payload = {"translators": translators} if wrapped else translators
    path = tmp_path / "translators.pt"
    torch.save(payload, path)
    return str(path)


class TestLoadTunedLensTranslators:
    """Tests for load_tuned_lens_translators()."""

    def test_load_returns_list_of_tuples(self, tmp_path):
        """Load a valid .pt with 4 layers, hidden_dim=16. Check length and shapes."""
        from locos.detectors.logit_contrib import load_tuned_lens_translators

        num_layers, hidden_dim = 4, 16
        path = _make_translators_pt(tmp_path, num_layers=num_layers, hidden_dim=hidden_dim)

        result = load_tuned_lens_translators(path, num_layers=num_layers, hidden_dim=hidden_dim)

        assert isinstance(result, list)
        assert len(result) == num_layers
        for i, (A, b) in enumerate(result):
            assert isinstance(A, torch.Tensor), f"Layer {i}: A should be a Tensor"
            assert isinstance(b, torch.Tensor), f"Layer {i}: b should be a Tensor"
            assert A.shape == (hidden_dim, hidden_dim), f"Layer {i}: A shape mismatch"
            assert b.shape == (hidden_dim,), f"Layer {i}: b shape mismatch"
            assert A.dtype == torch.float32, f"Layer {i}: A should be float32"
            assert b.dtype == torch.float32, f"Layer {i}: b should be float32"

    def test_layer_count_mismatch_raises(self, tmp_path):
        """Request more layers than the file contains -> ValueError mentioning 'layers'."""
        from locos.detectors.logit_contrib import load_tuned_lens_translators

        path = _make_translators_pt(tmp_path, num_layers=2, hidden_dim=16)

        with pytest.raises(ValueError, match="layers"):
            load_tuned_lens_translators(path, num_layers=4, hidden_dim=16)

    def test_hidden_dim_mismatch_raises(self, tmp_path):
        """A matrix dim doesn't match requested hidden_dim -> ValueError mentioning 'hidden'."""
        from locos.detectors.logit_contrib import load_tuned_lens_translators

        path = _make_translators_pt(tmp_path, num_layers=4, hidden_dim=32)

        with pytest.raises(ValueError, match="hidden"):
            load_tuned_lens_translators(path, num_layers=4, hidden_dim=16)

    def test_identity_translator_converts_to_zero_residual(self, tmp_path):
        """A=I in file (uzaymacar identity init) should become A_res=0 after loading."""
        from locos.detectors.logit_contrib import load_tuned_lens_translators

        num_layers, hidden_dim = 2, 8
        # Create a file with A=I (uzaymacar's identity initialization)
        translators = {}
        for layer in range(num_layers):
            translators[layer] = {
                "A": torch.eye(hidden_dim, dtype=torch.float32),
                "b": torch.zeros(hidden_dim, dtype=torch.float32),
            }
        path = tmp_path / "translators.pt"
        torch.save({"translators": translators}, path)

        result = load_tuned_lens_translators(str(path), num_layers=num_layers, hidden_dim=hidden_dim)

        for i, (A_res, b) in enumerate(result):
            torch.testing.assert_close(
                A_res, torch.zeros(hidden_dim, hidden_dim), msg=f"Layer {i}: identity A should become zero residual"
            )
            torch.testing.assert_close(b, torch.zeros(hidden_dim), msg=f"Layer {i}: b should be unchanged")

    def test_end_to_end_identity_translator_preserves_u_y(self, tmp_path):
        """End-to-end: load identity translator -> apply correction -> u_y unchanged."""
        from locos.detectors.logit_contrib import (
            apply_tuned_lens_correction,
            load_tuned_lens_translators,
        )

        num_layers, hidden_dim = 2, 8
        translators_data = {}
        for layer in range(num_layers):
            translators_data[layer] = {
                "A": torch.eye(hidden_dim, dtype=torch.float32),
                "b": torch.zeros(hidden_dim, dtype=torch.float32),
            }
        path = tmp_path / "translators.pt"
        torch.save({"translators": translators_data}, path)

        translators = load_tuned_lens_translators(str(path), num_layers=num_layers, hidden_dim=hidden_dim)
        u_y = torch.randn(hidden_dim)

        for layer_idx in range(num_layers):
            A_res, b = translators[layer_idx]
            u_corrected, bias = apply_tuned_lens_correction(u_y, A_res, b)
            torch.testing.assert_close(u_corrected, u_y)
            assert bias == pytest.approx(0.0, abs=1e-6)

    def test_end_to_end_known_translator(self, tmp_path):
        """End-to-end: file has A_full=1.5*I, b=1 -> u_corrected = 1.5*u_y, bias = sum(u_y)."""
        from locos.detectors.logit_contrib import (
            apply_tuned_lens_correction,
            load_tuned_lens_translators,
        )

        hidden_dim = 4
        translators_data = {
            0: {
                "A": torch.eye(hidden_dim) * 1.5,  # Full transform: 1.5*I
                "b": torch.ones(hidden_dim),
            }
        }
        path = tmp_path / "translators.pt"
        torch.save(translators_data, path)

        translators = load_tuned_lens_translators(str(path), num_layers=1, hidden_dim=hidden_dim)
        A_res, b = translators[0]

        # A_res should be 1.5*I - I = 0.5*I
        torch.testing.assert_close(A_res, torch.eye(hidden_dim) * 0.5)

        u_y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        u_corrected, bias = apply_tuned_lens_correction(u_y, A_res, b)

        # (I + 0.5*I)^T @ u_y = 1.5 * u_y
        torch.testing.assert_close(u_corrected, 1.5 * u_y)
        assert bias == pytest.approx(10.0, abs=1e-6)


class TestApplyTunedLensCorrection:
    """Test the affine correction of the unembedding vector."""

    def test_identity_translator_returns_original(self):
        """A = 0, b = 0 should give u_y_corrected = u_y, bias = 0."""
        from locos.detectors.logit_contrib import apply_tuned_lens_correction

        hidden_dim = 8
        u_y = torch.randn(hidden_dim)
        A = torch.zeros(hidden_dim, hidden_dim)
        b = torch.zeros(hidden_dim)
        u_corrected, bias = apply_tuned_lens_correction(u_y, A, b)
        torch.testing.assert_close(u_corrected, u_y)
        assert bias == pytest.approx(0.0, abs=1e-6)

    def test_known_correction(self):
        """Verify: u_corrected = (I + A)^T @ u_y, bias = u_y . b."""
        from locos.detectors.logit_contrib import apply_tuned_lens_correction

        hidden_dim = 4
        u_y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        A = torch.eye(hidden_dim) * 0.5  # A = 0.5 * I
        b = torch.ones(hidden_dim)
        u_corrected, bias = apply_tuned_lens_correction(u_y, A, b)
        # (I + 0.5*I)^T @ u_y = 1.5 * u_y
        expected_u = 1.5 * u_y
        torch.testing.assert_close(u_corrected, expected_u)
        # bias = u_y . b = 1 + 2 + 3 + 4 = 10
        assert bias == pytest.approx(10.0, abs=1e-6)

    def test_device_propagation(self):
        """Output should be on the same device as u_y."""
        from locos.detectors.logit_contrib import apply_tuned_lens_correction

        u_y = torch.randn(8)  # CPU
        A = torch.randn(8, 8)
        b = torch.randn(8)
        u_corrected, _ = apply_tuned_lens_correction(u_y, A, b)
        assert u_corrected.device == u_y.device


class TestLogitBiasInSpatialContrast:
    """Test that logit_bias shifts spatial contrast correctly."""

    def test_zero_bias_matches_original(self):
        """With logit_bias=0, output should match the original function."""
        from locos.detectors.logit_contrib import compute_logit_contribution_per_step

        num_heads, key_len, head_dim, hidden_dim = 2, 10, 4, 8
        torch.manual_seed(42)
        attn_weights = torch.softmax(torch.randn(num_heads, key_len), dim=-1)
        value_cache = torch.randn(num_heads, key_len, head_dim)
        o_proj_weight = torch.randn(num_heads, head_dim, hidden_dim)
        u_y = torch.randn(hidden_dim)
        phi_n_orig, phi_o_orig = compute_logit_contribution_per_step(
            attn_weights,
            value_cache,
            o_proj_weight,
            u_y,
            num_heads,
            num_heads,
            3,
            6,
            key_len,
        )
        phi_n_bias, phi_o_bias = compute_logit_contribution_per_step(
            attn_weights,
            value_cache,
            o_proj_weight,
            u_y,
            num_heads,
            num_heads,
            3,
            6,
            key_len,
            logit_bias=0.0,
        )
        np.testing.assert_allclose(phi_n_orig, phi_n_bias, atol=1e-5)
        np.testing.assert_allclose(phi_o_orig, phi_o_bias, atol=1e-5)

    def test_nonzero_bias_shifts_contributions(self):
        """A positive bias should increase both needle and off-needle sums."""
        from locos.detectors.logit_contrib import compute_logit_contribution_per_step

        num_heads, key_len, head_dim, hidden_dim = 2, 10, 4, 8
        torch.manual_seed(42)
        attn_weights = torch.softmax(torch.randn(num_heads, key_len), dim=-1)
        value_cache = torch.randn(num_heads, key_len, head_dim)
        o_proj_weight = torch.randn(num_heads, head_dim, hidden_dim)
        u_y = torch.randn(hidden_dim)
        needle_start, needle_end = 3, 6
        phi_n_orig, phi_o_orig = compute_logit_contribution_per_step(
            attn_weights,
            value_cache,
            o_proj_weight,
            u_y,
            num_heads,
            num_heads,
            needle_start,
            needle_end,
            key_len,
        )
        phi_n_bias, phi_o_bias = compute_logit_contribution_per_step(
            attn_weights,
            value_cache,
            o_proj_weight,
            u_y,
            num_heads,
            num_heads,
            needle_start,
            needle_end,
            key_len,
            logit_bias=100.0,
        )
        assert np.all(phi_n_bias > phi_n_orig)
        assert np.all(phi_o_bias > phi_o_orig)

    def test_bias_effect_on_spatial_contrast_uniform_attention(self):
        """With uniform attention, bias cancels in S = L+ - L-."""
        from locos.detectors.logit_contrib import compute_logit_contribution_per_step

        num_heads, key_len = 1, 10
        head_dim, hidden_dim = 4, 8
        attn_weights = torch.ones(num_heads, key_len) / key_len
        value_cache = torch.zeros(num_heads, key_len, head_dim)
        o_proj_weight = torch.zeros(num_heads, head_dim, hidden_dim)
        u_y = torch.zeros(hidden_dim)
        needle_start, needle_end = 3, 6
        bias = 5.0
        phi_n, phi_o = compute_logit_contribution_per_step(
            attn_weights,
            value_cache,
            o_proj_weight,
            u_y,
            num_heads,
            num_heads,
            needle_start,
            needle_end,
            key_len,
            logit_bias=bias,
        )
        # With uniform attention: S = L+ - L- = 0 (bias cancels)
        np.testing.assert_allclose(phi_n - phi_o, [0.0], atol=1e-5)
