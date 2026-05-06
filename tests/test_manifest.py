"""Tests for ExperimentManifest."""

from __future__ import annotations

import json
from pathlib import Path

from locos_eval.evals.manifest import ExperimentManifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_manifest(**overrides) -> ExperimentManifest:
    defaults = dict(
        experiment_key="nq_swap/meta-llama_Meta-Llama-3-8B-Instruct/greedy",
        task="nq_swap",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        decoding="greedy",
        variant="greedy",
    )
    defaults.update(overrides)
    return ExperimentManifest(**defaults)


def _sample_run(
    timestamp: str = "20260409_120000",
    n_samples: int = 100,
    limit: int | None = None,
    metrics: dict | None = None,
    config_hash: str = "abc123",
    files: list[str] | None = None,
) -> dict:
    return dict(
        timestamp=timestamp,
        n_samples=n_samples,
        limit=limit,
        metrics=metrics or {"sub_em": 0.85, "org_em": 0.42},
        config_hash=config_hash,
        files=files or ["generations.jsonl", "scores.jsonl"],
    )


# ---------------------------------------------------------------------------
# test_create_empty
# ---------------------------------------------------------------------------
class TestCreateEmpty:
    def test_status_is_empty(self):
        m = _make_manifest()
        assert m.status == "empty"

    def test_runs_is_empty_list(self):
        m = _make_manifest()
        assert m.runs == []

    def test_latest_timestamp_is_none(self):
        m = _make_manifest()
        assert m.latest_timestamp is None


# ---------------------------------------------------------------------------
# test_add_run
# ---------------------------------------------------------------------------
class TestAddRun:
    def test_returns_true_on_success(self):
        m = _make_manifest()
        result = m.add_run(**_sample_run())
        assert result is True

    def test_status_becomes_complete(self):
        m = _make_manifest()
        m.add_run(**_sample_run())
        assert m.status == "complete"

    def test_latest_timestamp_is_set(self):
        m = _make_manifest()
        m.add_run(**_sample_run(timestamp="20260409_120000"))
        assert m.latest_timestamp == "20260409_120000"

    def test_run_is_appended(self):
        m = _make_manifest()
        m.add_run(**_sample_run())
        assert len(m.runs) == 1
        assert m.runs[0]["timestamp"] == "20260409_120000"
        assert m.runs[0]["n_samples"] == 100
        assert m.runs[0]["metrics"] == {"sub_em": 0.85, "org_em": 0.42}

    def test_multiple_runs(self):
        m = _make_manifest()
        m.add_run(**_sample_run(timestamp="20260409_120000", config_hash="aaa"))
        m.add_run(**_sample_run(timestamp="20260409_130000", config_hash="bbb"))
        assert len(m.runs) == 2
        assert m.latest_timestamp == "20260409_130000"


# ---------------------------------------------------------------------------
# test_add_duplicate_timestamp_is_skipped
# ---------------------------------------------------------------------------
class TestAddDuplicateTimestamp:
    def test_returns_false_on_duplicate(self):
        m = _make_manifest()
        m.add_run(**_sample_run(timestamp="20260409_120000"))
        result = m.add_run(**_sample_run(timestamp="20260409_120000"))
        assert result is False

    def test_does_not_duplicate_run(self):
        m = _make_manifest()
        m.add_run(**_sample_run(timestamp="20260409_120000"))
        m.add_run(**_sample_run(timestamp="20260409_120000"))
        assert len(m.runs) == 1


# ---------------------------------------------------------------------------
# test_add_run_flags_duplicate_config
# ---------------------------------------------------------------------------
class TestAddRunFlagsDuplicateConfig:
    def test_duplicate_config_flagged(self):
        m = _make_manifest()
        m.add_run(**_sample_run(timestamp="20260409_120000", config_hash="same"))
        m.add_run(**_sample_run(timestamp="20260409_130000", config_hash="same"))
        assert "duplicate_config" not in m.runs[0]
        assert m.runs[1]["duplicate_config"] is True

    def test_different_config_not_flagged(self):
        m = _make_manifest()
        m.add_run(**_sample_run(timestamp="20260409_120000", config_hash="aaa"))
        m.add_run(**_sample_run(timestamp="20260409_130000", config_hash="bbb"))
        assert "duplicate_config" not in m.runs[0]
        assert "duplicate_config" not in m.runs[1]


# ---------------------------------------------------------------------------
# test_to_dict_and_from_dict_roundtrip
# ---------------------------------------------------------------------------
class TestRoundtrip:
    def test_empty_manifest_roundtrip(self):
        m = _make_manifest()
        restored = ExperimentManifest.from_dict(m.to_dict())
        assert restored.to_dict() == m.to_dict()

    def test_manifest_with_runs_roundtrip(self):
        m = _make_manifest()
        m.add_run(**_sample_run(timestamp="20260409_120000"))
        m.add_run(**_sample_run(timestamp="20260409_130000", config_hash="xyz"))
        restored = ExperimentManifest.from_dict(m.to_dict())
        assert restored.to_dict() == m.to_dict()

    def test_all_fields_preserved(self):
        m = _make_manifest()
        m.add_run(**_sample_run())
        d = m.to_dict()
        assert d["experiment_key"] == "nq_swap/meta-llama_Meta-Llama-3-8B-Instruct/greedy"
        assert d["task"] == "nq_swap"
        assert d["model"] == "meta-llama/Meta-Llama-3-8B-Instruct"
        assert d["decoding"] == "greedy"
        assert d["variant"] == "greedy"
        assert d["status"] == "complete"
        assert d["latest_timestamp"] == "20260409_120000"
        assert len(d["runs"]) == 1


# ---------------------------------------------------------------------------
# test_save_and_load_json
# ---------------------------------------------------------------------------
class TestSaveAndLoad:
    def test_save_creates_file(self, tmp_path: Path):
        m = _make_manifest()
        m.add_run(**_sample_run())
        path = tmp_path / "subdir" / "manifest.json"
        m.save(path)
        assert path.exists()

    def test_load_restores_manifest(self, tmp_path: Path):
        m = _make_manifest()
        m.add_run(**_sample_run(timestamp="20260409_120000"))
        m.add_run(**_sample_run(timestamp="20260409_130000", config_hash="xyz"))
        path = tmp_path / "manifest.json"
        m.save(path)
        loaded = ExperimentManifest.load(path)
        assert loaded.to_dict() == m.to_dict()

    def test_saved_json_is_valid(self, tmp_path: Path):
        m = _make_manifest()
        m.add_run(**_sample_run())
        path = tmp_path / "manifest.json"
        m.save(path)
        with open(path) as f:
            data = json.load(f)
        assert data["task"] == "nq_swap"
        assert len(data["runs"]) == 1


# ---------------------------------------------------------------------------
# test_has_timestamp
# ---------------------------------------------------------------------------
class TestHasTimestamp:
    def test_false_before_add(self):
        m = _make_manifest()
        assert m.has_timestamp("20260409_120000") is False

    def test_true_after_add(self):
        m = _make_manifest()
        m.add_run(**_sample_run(timestamp="20260409_120000"))
        assert m.has_timestamp("20260409_120000") is True

    def test_false_for_different_timestamp(self):
        m = _make_manifest()
        m.add_run(**_sample_run(timestamp="20260409_120000"))
        assert m.has_timestamp("20260409_999999") is False


# ---------------------------------------------------------------------------
# test_is_complete
# ---------------------------------------------------------------------------
class TestLatestTimestamp:
    def test_tracks_most_recent_when_added_in_order(self):
        m = _make_manifest()
        m.add_run(**_sample_run(timestamp="20260401_000000"))
        m.add_run(**_sample_run(timestamp="20260402_000000", config_hash="different"))
        assert m.latest_timestamp == "20260402_000000"

    def test_tracks_most_recent_when_added_out_of_order(self):
        m = _make_manifest()
        m.add_run(**_sample_run(timestamp="20260402_000000"))
        m.add_run(**_sample_run(timestamp="20260401_000000", config_hash="different"))
        # Should keep the later timestamp, not the last-added one
        assert m.latest_timestamp == "20260402_000000"


class TestIsComplete:
    def test_false_when_empty(self):
        m = _make_manifest()
        assert m.is_complete is False

    def test_true_after_add_run(self):
        m = _make_manifest()
        m.add_run(**_sample_run())
        assert m.is_complete is True
