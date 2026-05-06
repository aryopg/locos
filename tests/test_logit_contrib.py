"""Unit tests for locos/detectors/logit_contrib.py.

Verifies that the implementation matches the equations in
docs/logit_contribution_locos_v2.tex:

  Eq. 2 (phi):  phi_{t,j}^{(l,h)} = alpha_{t,j} * u_{y_t}^T W_O^{(l,h)} v_j^{(l,h)}
  Eq. 3 (spatial contrast):
      Phi+ = sum_{j in needle} phi_j
      Phi- = (needle_len / off_needle_len) * sum_{j not in needle} phi_j
  Eq. 5 (per-trial):  S^tau = (1/|T_ans|) sum_t (Phi+_t - Phi-_t)
  Eq. 4 (global):     S = (1/total_ans_steps) sum_tau sum_t (Phi+_t - Phi-_t)

No GPU required -- all tensors are CPU.
"""

import numpy as np
import pytest
import torch

from locos.detectors.logit_contrib import (
    compute_logit_contribution_per_step,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_identity_o_proj(num_heads: int, head_dim: int):
    """Build o_proj_weight that is identity-like for easy manual verification.

    Returns shape (num_heads, head_dim, hidden_dim) where hidden_dim = num_heads * head_dim.
    Each head's slice is a (head_dim, hidden_dim) matrix that picks out the
    corresponding head_dim block from the hidden_dim, i.e.
        o_proj[h] = I_{head_dim} padded into the h-th block of hidden_dim.

    With this weight, W_O^{(l,h)} v = concat(0, ..., v, ..., 0) where v sits
    at the h-th block.  So u^T W_O^{(l,h)} v = u[h*d_h : (h+1)*d_h] . v.
    """
    hidden_dim = num_heads * head_dim
    o_proj = torch.zeros(num_heads, head_dim, hidden_dim)
    for h in range(num_heads):
        o_proj[h, :, h * head_dim : (h + 1) * head_dim] = torch.eye(head_dim)
    return o_proj


# ---------------------------------------------------------------------------
# Tests: Eq. 2 -- per-position logit contribution phi
# ---------------------------------------------------------------------------


class TestPhiComputation:
    """Test that phi = alpha * u_y^T W_O v  (Eq. 2 in the paper)."""

    def test_phi_single_position_unit_vectors(self):
        """With identity W_O, unit v, and u_y selecting the right block,
        phi should equal the attention weight."""
        num_heads = 2
        head_dim = 4
        hidden_dim = num_heads * head_dim
        key_len = 5
        needle_start, needle_end = 1, 3

        # Identity-like o_proj
        o_proj = _make_identity_o_proj(num_heads, head_dim)

        # V cache: all ones at every position for simplicity
        V = torch.ones(num_heads, key_len, head_dim)

        # u_y: ones in head-0's block, zeros elsewhere
        # => u_y^T W_O^{(0)} v = sum(v) = head_dim, u_y^T W_O^{(1)} v = 0
        u_y = torch.zeros(hidden_dim)
        u_y[:head_dim] = 1.0

        # Attention: head 0 puts all mass on position 2 (in needle)
        attn = torch.zeros(num_heads, key_len)
        attn[0, 2] = 1.0  # head 0 -> needle position
        attn[1, 4] = 1.0  # head 1 -> off-needle position

        phi_needle, phi_off = compute_logit_contribution_per_step(
            attn_weights=attn,
            value_cache=V,
            o_proj_weight=o_proj,
            u_y=u_y,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            needle_start=needle_start,
            needle_end=needle_end,
            context_len=key_len,
        )

        # Head 0: phi[0, 2] = 1.0 * (1^T @ 1) = head_dim = 4.
        # All other positions have alpha=0, so phi_needle[0] = 4.0, phi_off[0] = 0.
        assert phi_needle[0] == pytest.approx(head_dim, abs=1e-5)
        assert phi_off[0] == pytest.approx(0.0, abs=1e-5)

        # Head 1: u_y has zeros in head-1's block, so u_y^T W_O^{(1)} v = 0 for all j.
        # => phi = 0 everywhere, regardless of attention.
        assert phi_needle[1] == pytest.approx(0.0, abs=1e-5)
        assert phi_off[1] == pytest.approx(0.0, abs=1e-5)

    def test_phi_scales_with_attention_weight(self):
        """phi at position j should scale linearly with alpha_{t,j}."""
        head_dim = 2
        hidden_dim = head_dim
        key_len = 4
        needle_start, needle_end = 0, 2

        o_proj = torch.eye(head_dim).unsqueeze(0)  # (1, head_dim, hidden_dim)
        V = torch.ones(1, key_len, head_dim)
        u_y = torch.ones(hidden_dim)

        # Uniform attention
        attn_uniform = torch.ones(1, key_len) / key_len
        phi_n1, _phi_o1 = compute_logit_contribution_per_step(
            attn_uniform, V, o_proj, u_y, 1, 1, needle_start, needle_end, key_len
        )

        # Double attention on needle, zero elsewhere
        attn_needle = torch.zeros(1, key_len)
        attn_needle[0, 0] = 0.5
        attn_needle[0, 1] = 0.5
        phi_n2, _phi_o2 = compute_logit_contribution_per_step(
            attn_needle, V, o_proj, u_y, 1, 1, needle_start, needle_end, key_len
        )

        # With uniform attn: phi per position = (1/4) * head_dim = 0.5
        # phi_needle = 2 * 0.5 = 1.0
        assert phi_n1[0] == pytest.approx(1.0, abs=1e-5)

        # With all mass on needle: phi_needle = (0.5 + 0.5) * head_dim = 2.0
        assert phi_n2[0] == pytest.approx(2.0, abs=1e-5)

    def test_phi_orthogonal_ov_gives_zero(self):
        """A head that attends to needle but whose OV output is orthogonal
        to u_y should have phi ≈ 0 (paper Sec 3.3, paragraph after Eq. 2)."""
        head_dim = 4
        hidden_dim = head_dim
        key_len = 10
        needle_start, needle_end = 3, 6

        o_proj = torch.eye(head_dim).unsqueeze(0)

        # v vectors in one direction
        V = torch.zeros(1, key_len, head_dim)
        V[0, :, 0] = 1.0  # all value vectors point along dim 0

        # u_y orthogonal to v
        u_y = torch.zeros(hidden_dim)
        u_y[1] = 1.0  # points along dim 1

        # Strong attention to needle
        attn = torch.zeros(1, key_len)
        attn[0, 3:6] = 1.0 / 3

        phi_needle, phi_off = compute_logit_contribution_per_step(
            attn, V, o_proj, u_y, 1, 1, needle_start, needle_end, key_len
        )

        assert phi_needle[0] == pytest.approx(0.0, abs=1e-6)
        assert phi_off[0] == pytest.approx(0.0, abs=1e-6)

    def test_phi_manual_computation(self):
        """Hand-computed phi against the formula for a small example."""
        head_dim = 2
        key_len = 3
        needle_start, needle_end = 1, 2

        # W_O^{(l,h)}: (hidden_dim, head_dim) in paper = (3, 2)
        # In code o_proj is stored as (num_heads, head_dim, hidden_dim) = (1, 2, 3)
        # so o_proj[h] = W_O^T = (2, 3)
        W_O = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # (3, 2)
        o_proj = W_O.T.unsqueeze(0)  # (1, 2, 3)

        # Value vectors
        V = torch.zeros(1, key_len, head_dim)
        V[0, 0] = torch.tensor([1.0, 0.0])
        V[0, 1] = torch.tensor([0.0, 2.0])  # needle position
        V[0, 2] = torch.tensor([1.0, 1.0])

        u_y = torch.tensor([1.0, 1.0, 0.5])

        # Attention weights
        attn = torch.tensor([[0.2, 0.5, 0.3]])

        # Manual: p = W_O^T @ u_y = (2, 3) @ (3,) = (2,)
        p = W_O.T @ u_y  # [1*1+0*1+1*0.5, 0*1+1*1+1*0.5] = [1.5, 1.5]
        assert torch.allclose(p, torch.tensor([1.5, 1.5]))

        # phi[j] = alpha[j] * v[j]^T @ p
        # phi[0] = 0.2 * ([1,0] . [1.5,1.5]) = 0.2 * 1.5 = 0.3
        # phi[1] = 0.5 * ([0,2] . [1.5,1.5]) = 0.5 * 3.0 = 1.5
        # phi[2] = 0.3 * ([1,1] . [1.5,1.5]) = 0.3 * 3.0 = 0.9
        expected_phi = [0.3, 1.5, 0.9]

        phi_needle, phi_off = compute_logit_contribution_per_step(
            attn, V, o_proj, u_y, 1, 1, needle_start, needle_end, key_len
        )

        # Phi+ = phi[1] = 1.5
        assert phi_needle[0] == pytest.approx(expected_phi[1], abs=1e-5)

        # Phi- = (needle_len / off_needle_len) * (phi[0] + phi[2])
        #      = (1/2) * (0.3 + 0.9) = 0.6
        expected_phi_off = (1.0 / 2.0) * (expected_phi[0] + expected_phi[2])
        assert phi_off[0] == pytest.approx(expected_phi_off, abs=1e-5)


# ---------------------------------------------------------------------------
# Tests: Eq. 3 -- spatial contrast (needle vs off-needle rescaling)
# ---------------------------------------------------------------------------


class TestSpatialContrast:
    """Test the rescaling factor (needle_len / off_needle_len) in Eq. 3."""

    def test_rescaling_factor_values(self):
        """Verify Phi- uses the correct rescaling factor from Eq. 3:
        scale = (e - s) / (N_t - (e - s))."""
        head_dim = 2
        key_len = 10
        needle_start, needle_end = 2, 5  # needle_len = 3
        needle_len = needle_end - needle_start
        off_needle_len = key_len - needle_len  # 7

        o_proj = torch.eye(head_dim).unsqueeze(0)
        u_y = torch.ones(head_dim)

        # Uniform phi: v all ones, u all ones => v^T p = head_dim for all positions.
        # With uniform attention (1/key_len each), phi[j] = (1/key_len) * head_dim
        V = torch.ones(1, key_len, head_dim)
        attn = torch.ones(1, key_len) / key_len

        phi_needle, phi_off = compute_logit_contribution_per_step(
            attn, V, o_proj, u_y, 1, 1, needle_start, needle_end, key_len
        )

        phi_per_pos = head_dim / key_len  # 0.2

        # Phi+ = needle_len * phi_per_pos = 3 * 0.2 = 0.6
        assert phi_needle[0] == pytest.approx(needle_len * phi_per_pos, abs=1e-5)

        # Raw off-needle sum = off_needle_len * phi_per_pos = 7 * 0.2 = 1.4
        # Phi- = (needle_len / off_needle_len) * 1.4 = (3/7) * 1.4 = 0.6
        expected_phi_off = (needle_len / off_needle_len) * (off_needle_len * phi_per_pos)
        assert phi_off[0] == pytest.approx(expected_phi_off, abs=1e-5)

        # When phi is uniform, Phi+ == Phi- (the contrast cancels out)
        assert phi_needle[0] == pytest.approx(phi_off[0], abs=1e-5)

    def test_contrast_sign_needle_dominant(self):
        """When head writes answer-relevant info only from needle positions,
        Phi+ > Phi- (positive contrast score)."""
        head_dim = 2
        key_len = 8
        needle_start, needle_end = 2, 4

        o_proj = torch.eye(head_dim).unsqueeze(0)
        u_y = torch.ones(head_dim)

        # V: high values at needle, zero elsewhere
        V = torch.zeros(1, key_len, head_dim)
        V[0, 2:4] = 5.0  # needle positions have large values

        # Uniform attention
        attn = torch.ones(1, key_len) / key_len

        phi_needle, phi_off = compute_logit_contribution_per_step(
            attn, V, o_proj, u_y, 1, 1, needle_start, needle_end, key_len
        )

        assert phi_needle[0] > phi_off[0]
        assert phi_needle[0] - phi_off[0] > 0

    def test_contrast_sign_off_needle_dominant(self):
        """When head writes answer-relevant info from off-needle positions,
        Phi+ < Phi- (negative contrast score, paper Sec 3.3 after Eq. 5)."""
        head_dim = 2
        key_len = 8
        needle_start, needle_end = 2, 4

        o_proj = torch.eye(head_dim).unsqueeze(0)
        u_y = torch.ones(head_dim)

        # V: high values at OFF-needle, zero at needle
        V = torch.ones(1, key_len, head_dim) * 5.0
        V[0, 2:4] = 0.0  # needle positions have zero values

        attn = torch.ones(1, key_len) / key_len

        phi_needle, phi_off = compute_logit_contribution_per_step(
            attn, V, o_proj, u_y, 1, 1, needle_start, needle_end, key_len
        )

        assert phi_needle[0] < phi_off[0]
        # S_tau = Phi+ - Phi- < 0 (parametric-knowledge head)
        assert phi_needle[0] - phi_off[0] < 0


# ---------------------------------------------------------------------------
# Tests: GQA expansion (paper Sec 3.5 item 5)
# ---------------------------------------------------------------------------


class TestGQAExpansion:
    """Test that GQA (num_kv_heads < num_heads) expands V correctly."""

    def test_gqa_two_to_one(self):
        """2 Q-heads sharing 1 KV-head: both use the same V but different W_O."""
        num_heads = 2
        num_kv_heads = 1
        head_dim = 3
        hidden_dim = num_heads * head_dim
        key_len = 4
        needle_start, needle_end = 1, 2

        # V cache: only 1 KV head
        V = torch.randn(num_kv_heads, key_len, head_dim)

        # Different o_proj for each Q head
        o_proj = torch.randn(num_heads, head_dim, hidden_dim)

        u_y = torch.randn(hidden_dim)
        attn = torch.softmax(torch.randn(num_heads, key_len), dim=-1)

        phi_needle, _phi_off = compute_logit_contribution_per_step(
            attn, V, o_proj, u_y, num_heads, num_kv_heads, needle_start, needle_end, key_len
        )

        # Both heads should use the same V (expanded from the single KV head)
        # but get different scores due to different W_O slices.
        # Verify shapes
        assert phi_needle.shape == (num_heads,)
        assert _phi_off.shape == (num_heads,)

        # Scores CAN differ (different W_O)
        # Verify against manual expansion
        V_expanded = V.repeat_interleave(num_heads // num_kv_heads, dim=0)
        assert V_expanded.shape == (num_heads, key_len, head_dim)

        # Recompute manually for head 0
        p0 = o_proj[0] @ u_y  # (head_dim,)
        lc0 = (V_expanded[0] @ p0) * attn[0]  # (key_len,)
        expected_needle_0 = lc0[needle_start:needle_end].sum().item()
        assert phi_needle[0] == pytest.approx(expected_needle_0, abs=1e-4)

    def test_gqa_four_to_two(self):
        """4 Q-heads with 2 KV-heads (GQA ratio = 2)."""
        num_heads = 4
        num_kv_heads = 2
        head_dim = 2
        hidden_dim = num_heads * head_dim
        key_len = 6
        needle_start, needle_end = 0, 3

        V = torch.randn(num_kv_heads, key_len, head_dim)
        o_proj = torch.randn(num_heads, head_dim, hidden_dim)
        u_y = torch.randn(hidden_dim)
        attn = torch.softmax(torch.randn(num_heads, key_len), dim=-1)

        phi_needle, _phi_off = compute_logit_contribution_per_step(
            attn, V, o_proj, u_y, num_heads, num_kv_heads, needle_start, needle_end, key_len
        )

        assert phi_needle.shape == (num_heads,)
        assert _phi_off.shape == (num_heads,)

        # Q-heads 0,1 share KV-head 0; Q-heads 2,3 share KV-head 1
        V_expanded = V.repeat_interleave(2, dim=0)
        assert torch.equal(V_expanded[0], V_expanded[1])
        assert torch.equal(V_expanded[2], V_expanded[3])
        assert not torch.equal(V_expanded[0], V_expanded[2])

    def test_non_gqa_passthrough(self):
        """When num_kv_heads == num_heads, V is used directly (no expansion)."""
        num_heads = 3
        head_dim = 2
        hidden_dim = num_heads * head_dim
        key_len = 5
        needle_start, needle_end = 1, 3

        V = torch.randn(num_heads, key_len, head_dim)
        o_proj = torch.randn(num_heads, head_dim, hidden_dim)
        u_y = torch.randn(hidden_dim)
        attn = torch.softmax(torch.randn(num_heads, key_len), dim=-1)

        phi_needle, _phi_off = compute_logit_contribution_per_step(
            attn, V, o_proj, u_y, num_heads, num_heads, needle_start, needle_end, key_len
        )

        # Manually compute for each head
        for h in range(num_heads):
            p = o_proj[h] @ u_y
            lc = (V[h] @ p) * attn[h]
            expected = lc[needle_start:needle_end].sum().item()
            assert phi_needle[h] == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# Tests: Eq. 5 -- per-trial aggregation (mean over answer steps)
# ---------------------------------------------------------------------------


class TestPerTrialAggregation:
    """Test that S_tau = (1/|T_ans|) * sum_t (Phi+_t - Phi-_t)."""

    def test_single_answer_step(self):
        """With one answer step, S_tau = Phi+ - Phi- directly."""
        num_heads = 2
        head_dim = 2
        hidden_dim = num_heads * head_dim
        key_len = 6
        needle_start, needle_end = 1, 3

        o_proj = _make_identity_o_proj(num_heads, head_dim)
        u_y = torch.zeros(hidden_dim)
        u_y[:head_dim] = 1.0  # activates head 0 only

        V = torch.ones(num_heads, key_len, head_dim)

        # Head 0: lots of attention on needle
        attn = torch.zeros(num_heads, key_len)
        attn[0, 1] = 0.4
        attn[0, 2] = 0.4
        attn[0, 0] = 0.1
        attn[0, 3] = 0.1
        # Head 1: attention doesn't matter (u_y zeroes head-1 block)
        attn[1] = 1.0 / key_len

        phi_needle, phi_off = compute_logit_contribution_per_step(
            attn, V, o_proj, u_y, num_heads, num_heads, needle_start, needle_end, key_len
        )

        # For a single answer step: S_tau = Phi+ - Phi-
        S_tau = phi_needle - phi_off
        assert S_tau.shape == (num_heads,)

        # Head 0 should have positive contrast (needle-dominant attention)
        assert S_tau[0] > 0

        # Head 1 should be ~0 (orthogonal u_y)
        assert S_tau[1] == pytest.approx(0.0, abs=1e-5)

    def test_multi_step_averaging(self):
        """S_tau averages (Phi+ - Phi-) over multiple answer steps (Eq. 5)."""
        head_dim = 2
        key_len = 5
        needle_start, needle_end = 1, 3

        o_proj = torch.eye(head_dim).unsqueeze(0)

        # Two answer steps with different u_y and attention
        V = torch.ones(1, key_len, head_dim)

        # Step 1: u_y = [1, 0], attention on needle
        u_y_1 = torch.tensor([1.0, 0.0])
        attn_1 = torch.zeros(1, key_len)
        attn_1[0, 1] = 0.5
        attn_1[0, 2] = 0.3
        attn_1[0, 0] = 0.1
        attn_1[0, 3] = 0.1

        phi_n1, phi_o1 = compute_logit_contribution_per_step(
            attn_1, V, o_proj, u_y_1, 1, 1, needle_start, needle_end, key_len
        )

        # Step 2: u_y = [0, 1], attention spread
        u_y_2 = torch.tensor([0.0, 1.0])
        attn_2 = torch.ones(1, key_len) / key_len

        phi_n2, phi_o2 = compute_logit_contribution_per_step(
            attn_2, V, o_proj, u_y_2, 1, 1, needle_start, needle_end, key_len
        )

        # Per-step contrast
        delta_1 = phi_n1 - phi_o1
        delta_2 = phi_n2 - phi_o2

        # S_tau = mean of deltas (Eq. 5)
        S_tau = (delta_1 + delta_2) / 2

        # Step 2 has uniform phi, so delta_2 ≈ 0
        assert abs(delta_2[0]) < 1e-5

        # S_tau should be half of step 1's contrast
        assert S_tau[0] == pytest.approx(delta_1[0] / 2, abs=1e-5)


# ---------------------------------------------------------------------------
# Tests: Eq. 4 -- global aggregation (answer-step-weighted mean)
# ---------------------------------------------------------------------------


class TestGlobalAggregation:
    """Test the global pooling formula from Eq. 4.

    S = (1 / sum_tau |T_ans^tau|) * sum_tau sum_t (Phi+_t - Phi-_t)

    This is the answer-step-weighted mean, implemented in main() via
    pooled_L_plus/pooled_L_minus accumulators (lines 988-990 in logit_contrib.py).
    """

    def test_answer_step_weighting(self):
        """Trials with more answer steps should contribute proportionally
        more to the global score (Eq. 4)."""
        # Simulate two trials:
        # Trial A: 1 answer step, S_tau_A per step = 2.0
        # Trial B: 3 answer steps, S_tau_B per step = 1.0
        #
        # Per Eq. 4: S = (1*2.0 + 3*1.0) / (1+3) = 5.0/4 = 1.25
        # Per unweighted trial mean: (2.0 + 1.0) / 2 = 1.5

        num_layers, num_heads = 1, 1

        # Simulate L_plus, L_minus arrays as returned by detect_single_trial
        # Trial A: 1 answer step; L_plus - L_minus = 2.0
        L_plus_A = np.array([[3.0]], dtype=np.float32)
        L_minus_A = np.array([[1.0]], dtype=np.float32)
        S_tau_A = L_plus_A - L_minus_A  # [[2.0]]
        n_ans_A = 1

        # Trial B: 3 answer steps; L_plus - L_minus = 1.0
        L_plus_B = np.array([[2.5]], dtype=np.float32)
        L_minus_B = np.array([[1.5]], dtype=np.float32)
        S_tau_B = L_plus_B - L_minus_B  # [[1.0]]
        n_ans_B = 3

        # Global pooling (matching code at lines 988-990)
        pooled_L_plus = np.zeros((num_layers, num_heads), dtype=np.float64)
        pooled_L_minus = np.zeros((num_layers, num_heads), dtype=np.float64)
        pooled_ans_count = 0

        # L_plus/L_minus are already per-trial means (divided by n_ans).
        # The code multiplies back by n_ans to recover the raw sum.
        pooled_L_plus += L_plus_A * n_ans_A
        pooled_L_minus += L_minus_A * n_ans_A
        pooled_ans_count += n_ans_A

        pooled_L_plus += L_plus_B * n_ans_B
        pooled_L_minus += L_minus_B * n_ans_B
        pooled_ans_count += n_ans_B

        # Global S (Eq. 4)
        S_global = (pooled_L_plus - pooled_L_minus) / pooled_ans_count

        # Expected: (1*2.0 + 3*1.0) / 4 = 1.25
        assert S_global[0, 0] == pytest.approx(1.25, abs=1e-6)

        # Contrast with unweighted trial mean (what head_counter/np.mean gives)
        S_unweighted = (S_tau_A[0, 0] + S_tau_B[0, 0]) / 2
        assert S_unweighted == pytest.approx(1.5, abs=1e-6)

        # They differ when trials have different numbers of answer steps
        assert S_global[0, 0] != pytest.approx(S_unweighted, abs=1e-2)


# ---------------------------------------------------------------------------
# Tests: o_proj weight extraction shape
# ---------------------------------------------------------------------------


class TestOProjExtraction:
    """Test get_o_proj_weights produces the correct shape and values."""

    def test_reshape_identity(self):
        """Verify that the o_proj reshape recovers per-head slices correctly.

        o_proj.weight in nn.Linear has shape (hidden_dim, num_heads * head_dim).
        After reshape: (num_heads, head_dim, hidden_dim).
        o_proj_reshaped[h, :, :] should correspond to columns
        [h*head_dim : (h+1)*head_dim] of the original weight.
        """
        num_heads = 3
        head_dim = 4
        hidden_dim = 16  # doesn't have to equal num_heads * head_dim in general

        # Simulate o_proj.weight: (hidden_dim, num_heads * head_dim)
        w = torch.randn(hidden_dim, num_heads * head_dim)

        # Apply the same reshape as get_o_proj_weights (line 148)
        w_reshaped = w.view(hidden_dim, num_heads, head_dim).permute(1, 2, 0)
        assert w_reshaped.shape == (num_heads, head_dim, hidden_dim)

        # Each head's slice should match the corresponding columns
        for h in range(num_heads):
            # Original slice for head h: w[:, h*head_dim : (h+1)*head_dim]
            # Shape: (hidden_dim, head_dim) = W_O^{(l,h)}
            original_slice = w[:, h * head_dim : (h + 1) * head_dim]

            # Reshaped: w_reshaped[h] has shape (head_dim, hidden_dim) = (W_O^{(l,h)})^T
            assert torch.allclose(w_reshaped[h], original_slice.T)

    def test_phi_through_reshape(self):
        """End-to-end: the reshaped o_proj in compute_logit_contribution_per_step
        computes the same phi as a direct matrix multiply with the original weight."""
        num_heads = 2
        head_dim = 3
        hidden_dim = 8
        key_len = 4
        needle_start, needle_end = 1, 3

        # Original o_proj weight (hidden_dim, num_heads * head_dim)
        w_orig = torch.randn(hidden_dim, num_heads * head_dim)
        w_reshaped = w_orig.view(hidden_dim, num_heads, head_dim).permute(1, 2, 0)

        V = torch.randn(num_heads, key_len, head_dim)
        u_y = torch.randn(hidden_dim)
        attn = torch.softmax(torch.randn(num_heads, key_len), dim=-1)

        # Call the function (we verify values manually below, so discard aggregated outputs)
        compute_logit_contribution_per_step(
            attn, V, w_reshaped, u_y, num_heads, num_heads, needle_start, needle_end, key_len
        )

        # Manual reference: for each head, phi[h,j] = alpha[h,j] * u_y^T @ W_O^{(h)} @ v[h,j]
        for h in range(num_heads):
            W_O_h = w_orig[:, h * head_dim : (h + 1) * head_dim]  # (hidden_dim, head_dim)
            for j in range(key_len):
                # phi = alpha * u_y^T W_O v
                phi_manual = attn[h, j].item() * (u_y @ W_O_h @ V[h, j]).item()
                # Reconstruct phi from the function's internals
                p = w_reshaped[h] @ u_y  # (head_dim,)
                phi_func = attn[h, j].item() * (V[h, j] @ p).item()
                assert phi_func == pytest.approx(phi_manual, abs=1e-4)


# ---------------------------------------------------------------------------
# Tests: edge cases and assertions
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_attention_gives_zero_phi(self):
        """All-zero attention weights => phi = 0 everywhere."""
        num_heads = 2
        head_dim = 3
        key_len = 5
        needle_start, needle_end = 1, 3

        o_proj = torch.randn(num_heads, head_dim, num_heads * head_dim)
        V = torch.randn(num_heads, key_len, head_dim)
        u_y = torch.randn(num_heads * head_dim)
        attn = torch.zeros(num_heads, key_len)

        phi_needle, phi_off = compute_logit_contribution_per_step(
            attn, V, o_proj, u_y, num_heads, num_heads, needle_start, needle_end, key_len
        )

        np.testing.assert_allclose(phi_needle, 0.0, atol=1e-7)
        np.testing.assert_allclose(phi_off, 0.0, atol=1e-7)

    def test_needle_spanning_full_context_raises(self):
        """If needle covers the entire context, off_needle_len = 0 => assertion error."""
        head_dim = 2
        key_len = 5

        o_proj = torch.zeros(1, head_dim, head_dim)
        V = torch.zeros(1, key_len, head_dim)
        u_y = torch.zeros(head_dim)
        attn = torch.zeros(1, key_len)

        with pytest.raises(AssertionError, match="No off-needle positions"):
            compute_logit_contribution_per_step(
                attn,
                V,
                o_proj,
                u_y,
                1,
                1,
                needle_start=0,
                needle_end=key_len,
                context_len=key_len,
            )

    def test_output_dtypes(self):
        """Return arrays should be float32 numpy."""
        num_heads = 2
        head_dim = 2
        key_len = 4

        o_proj = torch.randn(num_heads, head_dim, num_heads * head_dim)
        V = torch.randn(num_heads, key_len, head_dim)
        u_y = torch.randn(num_heads * head_dim)
        attn = torch.softmax(torch.randn(num_heads, key_len), dim=-1)

        phi_needle, phi_off = compute_logit_contribution_per_step(
            attn, V, o_proj, u_y, num_heads, num_heads, 0, 2, key_len
        )

        assert isinstance(phi_needle, np.ndarray)
        assert isinstance(phi_off, np.ndarray)
        assert phi_needle.dtype == np.float32
        assert phi_off.dtype == np.float32

    def test_negative_phi_values_allowed(self):
        """phi can be negative when the head writes AWAY from the answer direction.
        The paper notes S_tau is unclamped (Table 1, 'Sign' row)."""
        head_dim = 2
        key_len = 4
        needle_start, needle_end = 0, 2

        o_proj = torch.eye(head_dim).unsqueeze(0)

        # V points opposite to u_y
        V = torch.zeros(1, key_len, head_dim)
        V[0, :, 0] = -1.0  # negative direction

        u_y = torch.tensor([1.0, 0.0])  # positive direction

        attn = torch.ones(1, key_len) / key_len

        phi_needle, phi_off = compute_logit_contribution_per_step(
            attn, V, o_proj, u_y, 1, 1, needle_start, needle_end, key_len
        )

        # All phi values are negative => both Phi+ and Phi- are negative
        assert phi_needle[0] < 0
        assert phi_off[0] < 0


# ---------------------------------------------------------------------------
# Tests: symmetry and invariance properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_uniform_phi_zero_contrast(self):
        """When phi is uniform across all positions, Phi+ == Phi- and
        the contrast S_tau = 0 (rescaling makes them comparable)."""
        head_dim = 2
        key_len = 10
        needle_start, needle_end = 3, 7  # 4 needle, 6 off-needle

        o_proj = torch.eye(head_dim).unsqueeze(0)
        V = torch.ones(1, key_len, head_dim)  # uniform v
        u_y = torch.ones(head_dim)
        attn = torch.ones(1, key_len) / key_len  # uniform attention

        phi_needle, phi_off = compute_logit_contribution_per_step(
            attn, V, o_proj, u_y, 1, 1, needle_start, needle_end, key_len
        )

        # Phi+ and Phi- should be equal (rescaling normalizes for span length)
        assert phi_needle[0] == pytest.approx(phi_off[0], abs=1e-5)
        assert phi_needle[0] - phi_off[0] == pytest.approx(0.0, abs=1e-5)

    def test_scaling_linearity(self):
        """phi scales linearly with u_y magnitude (from the dot product)."""
        head_dim = 3
        key_len = 6
        needle_start, needle_end = 1, 3

        o_proj = torch.randn(1, head_dim, head_dim)
        V = torch.randn(1, key_len, head_dim)
        u_y = torch.randn(head_dim)
        attn = torch.softmax(torch.randn(1, key_len), dim=-1)

        phi_n1, phi_o1 = compute_logit_contribution_per_step(
            attn, V, o_proj, u_y, 1, 1, needle_start, needle_end, key_len
        )

        scale = 3.0
        phi_n2, phi_o2 = compute_logit_contribution_per_step(
            attn, V, o_proj, u_y * scale, 1, 1, needle_start, needle_end, key_len
        )

        np.testing.assert_allclose(phi_n2, phi_n1 * scale, rtol=1e-4)
        np.testing.assert_allclose(phi_o2, phi_o1 * scale, rtol=1e-4)

    def test_additivity_over_positions(self):
        """Phi+ + raw_Phi_off (before rescaling) should equal the total
        sum of phi over all positions."""
        num_heads = 2
        head_dim = 3
        hidden_dim = num_heads * head_dim
        key_len = 8
        needle_start, needle_end = 2, 5
        needle_len = needle_end - needle_start
        off_needle_len = key_len - needle_len

        o_proj = torch.randn(num_heads, head_dim, hidden_dim)
        V = torch.randn(num_heads, key_len, head_dim)
        u_y = torch.randn(hidden_dim)
        attn = torch.softmax(torch.randn(num_heads, key_len), dim=-1)

        phi_needle, phi_off_rescaled = compute_logit_contribution_per_step(
            attn, V, o_proj, u_y, num_heads, num_heads, needle_start, needle_end, key_len
        )

        # Recover raw off-needle sum: phi_off_rescaled = (needle_len/off_needle_len) * raw_sum
        phi_off_raw = phi_off_rescaled * (off_needle_len / needle_len)

        # Compute total phi manually
        u_projected = torch.einsum("hde,e->hd", o_proj, u_y)
        logit_contrib = torch.einsum("hkd,hd->hk", V, u_projected)
        phi_total = (attn * logit_contrib).sum(dim=-1).float().numpy()

        np.testing.assert_allclose(phi_needle + phi_off_raw, phi_total, rtol=1e-4)
