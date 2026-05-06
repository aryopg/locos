"""Unit tests for locos/analysis/direct_path_bias.py.

Verifies rank-agreement correctness on two synthetic rankings and the
disagreement-quadrant selection logic.
"""

from __future__ import annotations

import pytest

from locos.analysis import direct_path_bias as dpb


def test_compute_agreement_identical_rankings_have_rho_one():
    means = {f"0-{h}": float(h) for h in range(20)}
    agreement = dpb.compute_agreement(means, means, model="fake/toy")
    assert agreement["spearman"] == pytest.approx(1.0)
    assert agreement["kendall"] == pytest.approx(1.0)
    assert agreement["overlap@10"] == 10
    assert agreement["n_heads"] == 20
    assert agreement["model"] == "fake/toy"


def test_compute_agreement_reversed_rankings_have_rho_minus_one():
    a = {f"0-{h}": float(h) for h in range(20)}
    b = {f"0-{h}": float(19 - h) for h in range(20)}
    agreement = dpb.compute_agreement(a, b, model="fake/toy")
    assert agreement["spearman"] == pytest.approx(-1.0)
    assert agreement["kendall"] == pytest.approx(-1.0)
    # Reversed rankings share nothing in top-k (k<n/2)
    assert agreement["overlap@10"] == 0


def test_layer_bucket_three_equal_bins():
    # 12 layers → bins of size 4: early 0-3, mid 4-7, late 8-11
    assert dpb.layer_bucket(0, 12) == "early"
    assert dpb.layer_bucket(3, 12) == "early"
    assert dpb.layer_bucket(4, 12) == "mid"
    assert dpb.layer_bucket(7, 12) == "mid"
    assert dpb.layer_bucket(8, 12) == "late"
    assert dpb.layer_bucket(11, 12) == "late"


def test_identify_disagreement_finds_high_cri_low_lc_heads():
    # 20 heads. We'll plant one head that is top-CRI but bottom-LC.
    rows = []
    for i in range(20):
        rows.append(
            {
                "layer": 0,
                "head": i,
                "cri_rank": i,  # 0 is top
                "lc_rank": 19 - i,  # 19 is bottom for i=0 (the planted one)
                "cri_score": 0.0,
                "lc_score": 0.0,
                "bucket": "mid",
                "layer_depth": 0.5,
            }
        )
    disagreement = dpb.identify_disagreement(rows)
    # top-10% CRI (cri_rank < 2) AND bottom-50% LC (lc_rank >= 10).
    # Heads matching: i=0 (cri=0, lc=19), i=1 (cri=1, lc=18).
    assert len(disagreement) == 2
    assert {r["head"] for r in disagreement} == {0, 1}


def test_compute_rows_assigns_correct_layer_depth():
    means_cri = {"0-0": 1.0, "1-0": 0.5, "2-0": 0.0}
    means_lc = {"0-0": 0.2, "1-0": 0.5, "2-0": 0.9}
    rows = dpb.compute_rows(means_cri, means_lc, num_layers=3)
    depths = {r["layer"]: r["layer_depth"] for r in rows}
    assert depths[0] == pytest.approx(0.0)
    assert depths[1] == pytest.approx(0.5)
    assert depths[2] == pytest.approx(1.0)


def test_compute_rows_raises_on_no_shared_keys():
    with pytest.raises(ValueError):
        dpb.compute_rows({"0-0": 1.0}, {"1-1": 1.0}, num_layers=2)
