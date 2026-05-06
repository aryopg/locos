"""Tests for check_experiment.py — experiment completeness checking.

All tests mock _download_manifest to avoid HuggingFace dependency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

# scripts/ is not a package, so add it to sys.path for import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_experiment import is_experiment_complete


class TestReturnsTrue:
    def test_returns_true_when_manifest_complete(self, tmp_path: Path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "runs": [{"timestamp": "20260409_120000", "n_samples": 100}],
                }
            )
        )
        with patch("check_experiment._download_manifest", return_value=manifest):
            assert is_experiment_complete("repo/id", "nq_swap/model/greedy") is True


class TestReturnsFalseNoManifest:
    def test_returns_false_when_no_manifest(self):
        with patch("check_experiment._download_manifest", return_value=None):
            assert is_experiment_complete("repo/id", "nq_swap/model/greedy") is False


class TestReturnsFalseNotComplete:
    def test_returns_false_when_manifest_not_complete(self, tmp_path: Path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "status": "empty",
                    "runs": [],
                }
            )
        )
        with patch("check_experiment._download_manifest", return_value=manifest):
            assert is_experiment_complete("repo/id", "nq_swap/model/greedy") is False


class TestMinSamplesCheck:
    def test_min_samples_met(self, tmp_path: Path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "runs": [{"timestamp": "20260409_120000", "n_samples": 50}],
                }
            )
        )
        with patch("check_experiment._download_manifest", return_value=manifest):
            assert (
                is_experiment_complete(
                    "repo/id",
                    "nq_swap/model/greedy",
                    min_samples=50,
                )
                is True
            )

    def test_min_samples_not_met(self, tmp_path: Path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "runs": [{"timestamp": "20260409_120000", "n_samples": 50}],
                }
            )
        )
        with patch("check_experiment._download_manifest", return_value=manifest):
            assert (
                is_experiment_complete(
                    "repo/id",
                    "nq_swap/model/greedy",
                    min_samples=100,
                )
                is False
            )


# ---------------------------------------------------------------------------
# CLI tests (subprocess-based)
# ---------------------------------------------------------------------------
import subprocess


class TestCheckExperimentCLI:
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent / "scripts" / "check_experiment.py"), *args],
            capture_output=True,
            text=True,
        )

    def test_force_always_exits_1(self):
        r = self._run("--force", "--repo-id", "fake/repo", "--task", "t", "--model", "m/M", "--decoding", "greedy")
        assert r.returncode == 1
        assert "FORCE" in r.stdout

    def test_force_quiet_no_output(self):
        r = self._run(
            "--force", "--quiet", "--repo-id", "fake/repo", "--task", "t", "--model", "m/M", "--decoding", "greedy"
        )
        assert r.returncode == 1
        assert r.stdout.strip() == ""

    def test_missing_required_args_fails(self):
        r = self._run("--repo-id", "fake/repo")
        assert r.returncode != 0

    def test_empty_heads_normalized_to_none(self):
        """Passing --heads '' should not crash (normalized to None for greedy)."""
        r = self._run(
            "--force", "--repo-id", "fake/repo", "--task", "t", "--model", "m/M", "--decoding", "greedy", "--heads", ""
        )
        assert r.returncode == 1  # force always exits 1, but should not crash
