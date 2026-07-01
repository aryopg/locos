"""Tests for locos.analysis._utils."""

from __future__ import annotations

import json

import numpy as np
import pytest

from locos.analysis import _utils


def test_candidate_filenames_cover_current_and_legacy_names():
    assert _utils._candidate_filenames("Qwen3-8B", "locos", "niah") == ["Qwen3-8B_logit_contrib.json"]
    assert _utils._candidate_filenames("Qwen3-8B", "locos", "nolima") == ["Qwen3-8B_logit_contrib_nolima.json"]
    assert _utils._candidate_filenames("Qwen3-8B", "alpha_spatial", "nolima") == [
        "Qwen3-8B_attention_spatial_nolima.json"
    ]
    assert _utils._candidate_filenames("Qwen3-8B", "wu", "niah") == ["Qwen3-8B_wu_niah.json", "Qwen3-8B.json"]
    assert _utils._candidate_filenames("Qwen3-8B", "wu", "nolima") == [
        "Qwen3-8B_wu_nolima.json",
        "Qwen3-8B_nolima.json",
    ]


def test_candidate_filenames_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        _utils._candidate_filenames("Qwen3-8B", "bogus", "niah")


def test_hf_path_uses_canonical_score_filename():
    assert _utils._hf_path("Qwen/Qwen3-8B", "locos", "nolima") == "retrieval_heads/Qwen3-8B_logit_contrib_nolima.json"


def test_load_score_file_reads_envelope_format_without_download(tmp_path, monkeypatch):
    score_dir = tmp_path / "retrieval_heads"
    score_dir.mkdir()
    (score_dir / "Qwen3-8B_logit_contrib_nolima.json").write_text(
        json.dumps(
            {
                "scores": {"0-0": [1.0, 0.0], "1-2": [0.5]},
                "trial_ids": ["a", "b"],
                "meta": {"dataset": "nolima"},
            }
        )
    )
    monkeypatch.setattr(_utils, "_SCORE_DIRS", [score_dir])

    loaded = _utils.load_score_file("Qwen/Qwen3-8B", "locos", "nolima", download=False)

    assert loaded == _utils.ScoreFile(
        scores={"0-0": [1.0, 0.0], "1-2": [0.5]},
        trial_ids=["a", "b"],
        meta={"dataset": "nolima"},
    )


def test_load_score_file_reads_legacy_behavioral_format(tmp_path, monkeypatch):
    score_dir = tmp_path / "retrieval_heads"
    score_dir.mkdir()
    (score_dir / "Qwen3-8B.json").write_text(json.dumps({"0-0": [0.25], "1-1": [0.75, 1.0]}))
    monkeypatch.setattr(_utils, "_SCORE_DIRS", [score_dir])

    loaded = _utils.load_score_file("Qwen/Qwen3-8B", "wu", "niah", download=False)

    assert loaded == _utils.ScoreFile(scores={"0-0": [0.25], "1-1": [0.75, 1.0]}, trial_ids=[], meta={})


def test_load_score_file_returns_none_when_missing_and_download_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(_utils, "_SCORE_DIRS", [tmp_path])

    assert _utils.load_score_file("Qwen/Qwen3-8B", "locos", "niah", download=False) is None


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    lo, hi = _utils.bootstrap_ci([0.0, 1.0, 1.0, 1.0], B=500, seed=0)

    assert 0.0 <= lo <= np.mean([0.0, 1.0, 1.0, 1.0]) <= hi <= 1.0
    assert (lo, hi) == pytest.approx(_utils.bootstrap_ci([0.0, 1.0, 1.0, 1.0], B=500, seed=0))


def test_bootstrap_ci_returns_nan_pair_for_empty_values():
    lo, hi = _utils.bootstrap_ci([], B=10)

    assert np.isnan(lo)
    assert np.isnan(hi)


def test_load_eval_rows_prefers_latest_timestamped_result(tmp_path, monkeypatch):
    variant_dir = tmp_path / "task" / "org_Model" / "greedy_s1"
    variant_dir.mkdir(parents=True)
    (variant_dir / "results_20250101.jsonl").write_text(json.dumps({"id": "old"}) + "\n")
    (variant_dir / "results_20250102.jsonl").write_text(json.dumps({"id": "new"}) + "\n\n")
    monkeypatch.setattr(_utils, "_DOWNSTREAM_DIRS", [tmp_path])

    rows = _utils.load_eval_rows("task", "org/Model", ["greedy"], seeds=("_s1",))

    assert rows == [{"id": "new"}]


def test_load_eval_rows_falls_back_to_results_jsonl(tmp_path, monkeypatch):
    variant_dir = tmp_path / "task" / "org_Model" / "greedy"
    variant_dir.mkdir(parents=True)
    (variant_dir / "results.jsonl").write_text(json.dumps({"id": 1}) + "\n")
    monkeypatch.setattr(_utils, "_DOWNSTREAM_DIRS", [tmp_path])

    assert _utils.load_eval_rows("task", "org/Model", ["greedy"], seeds=("",)) == [{"id": 1}]


def test_load_eval_rows_returns_none_when_no_rows_found(tmp_path, monkeypatch):
    monkeypatch.setattr(_utils, "_DOWNSTREAM_DIRS", [tmp_path])

    assert _utils.load_eval_rows("task", "org/Model", ["greedy"], seeds=("",)) is None
