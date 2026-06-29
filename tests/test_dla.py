import numpy as np
import pytest
import torch

from locos.detectors.dla import compute_dla_per_step


def test_compute_dla_per_step_sums_all_positions():
    o_proj = torch.eye(2).unsqueeze(0)
    values = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    u_y = torch.tensor([1.0, 0.5])
    attn = torch.tensor([[0.2, 0.5, 0.3]])

    result = compute_dla_per_step(
        attn_weights=attn,
        value_cache=values,
        o_proj_weight=o_proj,
        u_y=u_y,
        num_heads=1,
        num_kv_heads=1,
    )

    # Per-position contributions:
    # 0.2 * dot([1, 0], [1, 0.5]) = 0.2
    # 0.5 * dot([0, 2], [1, 0.5]) = 0.5
    # 0.3 * dot([1, 1], [1, 0.5]) = 0.45
    assert result.shape == (1,)
    assert result[0] == pytest.approx(1.15, abs=1e-6)


def test_compute_dla_per_step_expands_gqa_values():
    o_proj = torch.zeros(2, 1, 2)
    o_proj[0, 0, 0] = 1.0
    o_proj[1, 0, 1] = 1.0
    values = torch.tensor([[[2.0], [4.0]]])
    u_y = torch.tensor([1.0, 3.0])
    attn = torch.tensor([[1.0, 0.0], [0.25, 0.75]])

    result = compute_dla_per_step(
        attn_weights=attn,
        value_cache=values,
        o_proj_weight=o_proj,
        u_y=u_y,
        num_heads=2,
        num_kv_heads=1,
    )

    np.testing.assert_allclose(result, np.array([2.0, 10.5], dtype=np.float32))

