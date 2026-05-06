"""Tests for locos/plotting/ablation_comparison.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from locos.plotting.ablation_comparison import extract_sweep_stats


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_trials(path: Path, scores: list[float], metric: str = "rouge_l") -> None:
    rows = [{metric: score, "trial_id": str(idx)} for idx, score in enumerate(scores)]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_extract_sweep_stats_bootstraps_trial_sidecar_for_single_cache(tmp_path):
    cache_path = tmp_path / "nolima_ablation_TestModel_method.json"
    _write_json(
        cache_path,
        {
            "TestModel__baseline": {
                "mode": "baseline",
                "rouge_l_mean": 0.5,
                "value": 0,
            },
            "TestModel__topk_5_mean": {
                "mode": "top-k",
                "ablation_mode": "mean",
                "rouge_l_mean": 0.75,
                "value": 5.0,
            },
        },
    )
    _write_trials(tmp_path / "nolima_ablation_TestModel_method_baseline_trials.jsonl", [0.0, 1.0])
    _write_trials(tmp_path / "nolima_ablation_TestModel_method_top-k_5p0_trials.jsonl", [0.0, 1.0, 1.0, 1.0])

    stats = extract_sweep_stats(
        caches=[json.loads(cache_path.read_text(encoding="utf-8"))],
        cache_paths=[cache_path],
        ablation_mode="mean",
        metric="rouge_l_mean",
        bootstrap_samples=500,
        bootstrap_seed=0,
    )

    assert stats.ks == [5]
    assert stats.means == pytest.approx([0.75])
    assert stats.ci_lows[0] < stats.means[0] < stats.ci_highs[0]
    assert stats.baseline_mean == pytest.approx(0.5)
    assert stats.baseline_ci_low is not None
    assert stats.baseline_ci_low < stats.baseline_mean < stats.baseline_ci_high
    assert stats.uncertainty_source == "bootstrap"


def test_extract_sweep_stats_uses_seed_std_for_cache_groups(tmp_path):
    cache_paths = [
        tmp_path / "nolima_ablation_TestModel_method_seed1.json",
        tmp_path / "nolima_ablation_TestModel_method_seed2.json",
    ]
    caches = []
    for cache_path, mean in zip(cache_paths, [0.25, 0.75]):
        cache = {
            "TestModel__topk_5_mean": {
                "mode": "top-k",
                "ablation_mode": "mean",
                "rouge_l_mean": mean,
                "value": 5.0,
            }
        }
        _write_json(cache_path, cache)
        caches.append(cache)

    stats = extract_sweep_stats(
        caches=caches,
        cache_paths=cache_paths,
        ablation_mode="mean",
        metric="rouge_l_mean",
        bootstrap_samples=0,
    )

    assert stats.means == pytest.approx([0.5])
    assert stats.seed_stds == pytest.approx([0.3535533905932738])
    assert stats.ci_lows == pytest.approx([0.1464466094067262])
    assert stats.ci_highs == pytest.approx([0.8535533905932737])
    assert stats.uncertainty_source == "seed_std"


def test_extract_sweep_stats_falls_back_to_matching_sidecar_when_cache_was_renamed(tmp_path):
    cache_path = tmp_path / "nolima_ablation_TestModel_renamed_method.json"
    cache = {
        "TestModel__topk_5_mean": {
            "mode": "top-k",
            "ablation_mode": "mean",
            "rouge_l_mean": 0.75,
            "n_samples": 4,
            "value": 5.0,
        }
    }
    _write_json(cache_path, cache)
    _write_trials(tmp_path / "nolima_ablation_TestModel_original_method_top-k_5p0_trials.jsonl", [0.0, 1.0, 1.0, 1.0])

    stats = extract_sweep_stats(
        caches=[cache],
        cache_paths=[cache_path],
        ablation_mode="mean",
        metric="rouge_l_mean",
        bootstrap_samples=200,
        bootstrap_seed=0,
    )

    assert stats.uncertainty_source == "bootstrap"
    assert stats.ci_lows[0] < stats.means[0] < stats.ci_highs[0]
