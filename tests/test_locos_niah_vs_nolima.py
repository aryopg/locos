"""Tests for locos/plotting/locos_niah_vs_nolima.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from locos.plotting.locos_niah_vs_nolima import extract_sweep_stats


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_trials(path: Path, scores: list[float], metric: str = "rouge_l") -> None:
    rows = [{metric: score, "trial_id": str(idx)} for idx, score in enumerate(scores)]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_extract_sweep_stats_bootstraps_curve_and_baseline_sidecars(tmp_path):
    cache_path = tmp_path / "nolima_ablation_TestModel_logit_contrib_nolima.json"
    cache = {
        "TestModel__baseline": {"mode": "baseline", "rouge_l_mean": 0.5, "n_samples": 2},
        "TestModel__topk_5_mean": {
            "mode": "top-k",
            "ablation_mode": "mean",
            "rouge_l_mean": 0.75,
            "n_samples": 4,
            "value": 5.0,
        },
    }
    _write_json(cache_path, cache)
    _write_trials(tmp_path / "nolima_ablation_TestModel_logit_contrib_nolima_baseline_trials.jsonl", [0.0, 1.0])
    _write_trials(
        tmp_path / "nolima_ablation_TestModel_logit_contrib_nolima_top-k_5p0_trials.jsonl", [0.0, 1.0, 1.0, 1.0]
    )

    stats = extract_sweep_stats(
        cache=cache,
        cache_path=cache_path,
        ablation_mode="mean",
        metric="rouge_l_mean",
        bootstrap_samples=500,
        bootstrap_seed=0,
    )

    assert stats.ks == [5]
    assert stats.scores == pytest.approx([0.75])
    assert stats.ci_lows[0] < stats.scores[0] < stats.ci_highs[0]
    assert stats.baseline == pytest.approx(0.5)
    assert stats.baseline_ci_low is not None
    assert stats.baseline_ci_low < stats.baseline < stats.baseline_ci_high


def test_extract_sweep_stats_keeps_point_estimates_without_sidecars(tmp_path):
    cache_path = tmp_path / "nolima_ablation_TestModel_logit_contrib_nolima.json"
    cache = {
        "TestModel__topk_5_mean": {
            "mode": "top-k",
            "ablation_mode": "mean",
            "rouge_l_mean": 0.75,
            "value": 5.0,
        },
    }

    stats = extract_sweep_stats(
        cache=cache,
        cache_path=cache_path,
        ablation_mode="mean",
        metric="rouge_l_mean",
        bootstrap_samples=500,
    )

    assert stats.ks == [5]
    assert stats.scores == pytest.approx([0.75])
    assert stats.ci_lows == pytest.approx([0.75])
    assert stats.ci_highs == pytest.approx([0.75])
