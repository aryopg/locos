"""Unit tests for locos/analysis/kv_group_analysis.py.

No GPU required. Verifies top-k → unique-KV-group mapping, mean aggregation,
and envelope-JSON emission on a hand-constructed fixture.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from locos.analysis import kv_group_analysis as kga


@pytest.fixture()
def toy_scores() -> dict[str, list[float]]:
    """2-layer, 8-Q-head / 2-KV-head model (GQA ratio 4:1).

    Scores are hand-designed so that the top 4 Q-heads all fall inside the
    same KV group (layer 0, group 0 → heads 0..3), which is the pathological
    case the KV-group reporting is meant to surface.
    """
    return {
        # Layer 0, Q-heads 0..7  →  KV groups 0,0,0,0, 1,1,1,1
        "0-0": [1.0, 1.0],
        "0-1": [0.95, 0.95],
        "0-2": [0.9, 0.9],
        "0-3": [0.85, 0.85],
        "0-4": [0.2, 0.2],
        "0-5": [0.19, 0.19],
        "0-6": [0.18, 0.18],
        "0-7": [0.17, 0.17],
        # Layer 1 — low scores, different KV group coverage
        "1-0": [0.10, 0.10],
        "1-1": [0.09, 0.09],
        "1-2": [0.08, 0.08],
        "1-3": [0.07, 0.07],
        "1-4": [0.06, 0.06],
        "1-5": [0.05, 0.05],
        "1-6": [0.04, 0.04],
        "1-7": [0.03, 0.03],
    }


def test_head_to_kv_group_matches_hf_layout():
    assert kga.head_to_kv_group(0, gqa_ratio=4) == 0
    assert kga.head_to_kv_group(3, gqa_ratio=4) == 0
    assert kga.head_to_kv_group(4, gqa_ratio=4) == 1
    assert kga.head_to_kv_group(7, gqa_ratio=4) == 1
    # Non-GQA (ratio 1): head == group
    for h in range(8):
        assert kga.head_to_kv_group(h, gqa_ratio=1) == h


def test_count_unique_kv_groups_collapses_inflated_topk():
    # Top-4 by score = heads (0,1,2,3) of layer 0 → all in KV group 0
    top_k = [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert kga.count_unique_kv_groups(top_k, gqa_ratio=4) == 1
    # Mix layers and groups: 4 heads, 3 distinct (layer, group) cells
    mixed = [(0, 0), (0, 4), (1, 0), (1, 4)]
    assert kga.count_unique_kv_groups(mixed, gqa_ratio=4) == 4


def test_compute_kv_group_stats_reports_coverage_and_concentration(toy_scores):
    rows = kga.compute_kv_group_stats(
        toy_scores,
        num_heads=8,
        num_kv_heads=2,
        model="fake/toy",
        method="logit_contrib",
        k_values=[1, 4, 8],
    )
    by_k = {r["k"]: r for r in rows}

    # k=1 → one head → one group
    assert by_k[1]["unique_groups"] == 1
    assert by_k[1]["concentration"] == pytest.approx(1.0)
    assert by_k[1]["coverage"] == pytest.approx(1.0)

    # k=4 → top-4 all in layer-0 group-0 → unique_groups=1, concentration=4
    assert by_k[4]["unique_groups"] == 1
    assert by_k[4]["concentration"] == pytest.approx(4.0)
    # Ceiling at k=4 is min(4, 2 layers × 2 groups = 4) = 4
    assert by_k[4]["coverage"] == pytest.approx(0.25)

    # k=8 → all layer-0 heads → both groups of layer 0 covered (2)
    assert by_k[8]["unique_groups"] == 2
    # Ceiling = min(8, 4) = 4 → coverage 2/4
    assert by_k[8]["coverage"] == pytest.approx(0.5)


def test_aggregate_scores_by_kv_group_averages_per_trial(toy_scores):
    grouped = kga.aggregate_scores_by_kv_group(toy_scores, gqa_ratio=4)
    # Layer 0 group 0 = mean over heads 0..3 per trial.
    expected_l0_g0 = np.mean([[1.0, 1.0], [0.95, 0.95], [0.9, 0.9], [0.85, 0.85]], axis=0).tolist()
    assert grouped["0-0"] == pytest.approx(expected_l0_g0)
    # Layer 0 group 1 = mean over heads 4..7 per trial.
    expected_l0_g1 = np.mean([[0.2, 0.2], [0.19, 0.19], [0.18, 0.18], [0.17, 0.17]], axis=0).tolist()
    assert grouped["0-1"] == pytest.approx(expected_l0_g1)
    # Number of group cells = num_layers × num_kv_heads = 2 × 2 = 4.
    assert set(grouped.keys()) == {"0-0", "0-1", "1-0", "1-1"}


def test_write_csv_and_envelope_json_roundtrip(tmp_path, toy_scores):
    rows = kga.compute_kv_group_stats(
        toy_scores,
        num_heads=8,
        num_kv_heads=2,
        model="fake/toy",
        method="logit_contrib",
        k_values=[1, 4],
    )
    csv_path = tmp_path / "out.csv"
    kga.write_csv(rows, csv_path)
    text = csv_path.read_text()
    assert "unique_groups" in text
    assert "concentration" in text

    grouped = kga.aggregate_scores_by_kv_group(toy_scores, gqa_ratio=4)
    json_path = tmp_path / "out_kvgroup.json"
    kga.write_envelope_json(grouped, {"method": "logit_contrib"}, json_path)
    reloaded = json.loads(json_path.read_text())
    assert reloaded["meta"]["method"] == "logit_contrib"
    assert "0-0" in reloaded["scores"]


def test_sweep_k_values_truncates_to_head_total():
    assert kga.sweep_k_values(10) == [1, 2, 5, 10]
    assert kga.sweep_k_values(3) == [1, 2]  # skips 5 and above
    assert kga.sweep_k_values(0) == [0]  # degenerate, returns fallback


def test_non_gqa_ratio_one_matches_head_level_coverage(toy_scores):
    """With ratio=1, each Q-head is its own group; unique_groups == k."""
    rows = kga.compute_kv_group_stats(
        toy_scores,
        num_heads=8,
        num_kv_heads=8,
        model="fake/toy",
        method="logit_contrib",
        k_values=[1, 4, 8],
    )
    by_k = {r["k"]: r for r in rows}
    assert by_k[1]["unique_groups"] == 1
    assert by_k[4]["unique_groups"] == 4
    assert by_k[8]["unique_groups"] == 8
    # Concentration is always 1.0 when every head is its own group.
    for k in (1, 4, 8):
        assert by_k[k]["concentration"] == pytest.approx(1.0)


def test_assertion_on_non_gqa_divisibility(toy_scores):
    # num_heads=8, num_kv_heads=3 is not divisible — should fail fast.
    with pytest.raises(AssertionError):
        kga.compute_kv_group_stats(
            toy_scores,
            num_heads=8,
            num_kv_heads=3,
            model="fake/toy",
            method="logit_contrib",
            k_values=[1],
        )
