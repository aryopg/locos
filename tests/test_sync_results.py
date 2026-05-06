"""Tests for sync_results scanning logic.

Only tests the pure scanning functions — no HuggingFace dependency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# scripts/ is not a package, so add it to sys.path for import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from sync_results import (
    compute_metrics_summary,
    config_hash,
    scan_local_experiments,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_CONFIG = {
    "task": "nq_swap",
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "heads": "retrieval_heads/Meta-Llama-3-8B-Instruct.json",
    "decoding": "ablation",
    "max_tokens": 512,
    "temperature": 0.0,
    "max_model_len": 8192,
    "tensor_parallel_size": 4,
    "gpu_memory_utilization": 0.5,
    "limit": 100,
    "timestamp": "20260409_120000",
}


def _make_result_line(
    sample_id: int,
    scores: dict[str, float] | None = None,
) -> str:
    """Create a single JSONL line for a results file."""
    record = {
        "sample_id": sample_id,
        "prompt": f"Prompt for sample {sample_id}",
        "output": f"Output for sample {sample_id}",
        "target": f"Target for sample {sample_id}",
        "scores": scores or {"sub_em": 0.8, "org_em": 0.4},
        "metadata": {"source": "test"},
    }
    return json.dumps(record)


def _make_experiment(
    base: Path,
    task: str,
    model_slug: str,
    variant: str,
    timestamps: list[str] | None = None,
    n_samples: int = 5,
    add_generations: bool = True,
    add_config: bool = True,
    scores: dict[str, float] | None = None,
) -> Path:
    """Create a realistic experiment directory structure.

    Creates::
        base/{task}/{model_slug}/{variant}/
            results_{timestamp}.jsonl
            results_{timestamp}_config.json
            generations.jsonl (if add_generations)
    """
    if timestamps is None:
        timestamps = ["20260409_120000"]

    variant_dir = base / task / model_slug / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    for ts in timestamps:
        # Results file
        results_path = variant_dir / f"results_{ts}.jsonl"
        lines = [_make_result_line(i, scores=scores) for i in range(n_samples)]
        results_path.write_text("\n".join(lines) + "\n")

        # Config sidecar
        if add_config:
            cfg = dict(_SAMPLE_CONFIG, timestamp=ts)
            config_path = variant_dir / f"results_{ts}_config.json"
            config_path.write_text(json.dumps(cfg, indent=2))

    # Generations checkpoint
    if add_generations:
        gen_path = variant_dir / "generations.jsonl"
        gen_lines = [json.dumps({"sample_id": i, "output": f"Gen {i}"}) for i in range(n_samples)]
        gen_path.write_text("\n".join(gen_lines) + "\n")

    return variant_dir


# ---------------------------------------------------------------------------
# test_scan_finds_experiments
# ---------------------------------------------------------------------------
class TestScanFindsExperiments:
    def test_finds_two_experiments(self, tmp_path: Path):
        _make_experiment(tmp_path, "nq_swap", "meta-llama_Llama-3-8B", "greedy")
        _make_experiment(tmp_path, "aci_bench", "meta-llama_Llama-3-8B", "decore_niah")
        experiments = scan_local_experiments(tmp_path)
        assert len(experiments) == 2

    def test_experiment_keys_correct(self, tmp_path: Path):
        _make_experiment(tmp_path, "nq_swap", "meta-llama_Llama-3-8B", "greedy")
        _make_experiment(tmp_path, "aci_bench", "meta-llama_Llama-3-8B", "decore_niah")
        experiments = scan_local_experiments(tmp_path)
        keys = {e["experiment_key"] for e in experiments}
        assert keys == {
            "nq_swap/meta-llama_Llama-3-8B/greedy",
            "aci_bench/meta-llama_Llama-3-8B/decore_niah",
        }


# ---------------------------------------------------------------------------
# test_scan_finds_timestamped_runs
# ---------------------------------------------------------------------------
class TestScanFindsTimestampedRuns:
    def test_extracts_timestamp(self, tmp_path: Path):
        _make_experiment(
            tmp_path,
            "nq_swap",
            "model_a",
            "greedy",
            timestamps=["20260409_120000"],
        )
        experiments = scan_local_experiments(tmp_path)
        assert len(experiments) == 1
        runs = experiments[0]["runs"]
        assert len(runs) == 1
        assert runs[0]["timestamp"] == "20260409_120000"

    def test_run_has_n_samples(self, tmp_path: Path):
        _make_experiment(
            tmp_path,
            "nq_swap",
            "model_a",
            "greedy",
            timestamps=["20260409_120000"],
            n_samples=10,
        )
        experiments = scan_local_experiments(tmp_path)
        assert experiments[0]["runs"][0]["n_samples"] == 10


# ---------------------------------------------------------------------------
# test_scan_finds_multiple_runs
# ---------------------------------------------------------------------------
class TestScanFindsMultipleRuns:
    def test_two_runs_in_one_dir(self, tmp_path: Path):
        _make_experiment(
            tmp_path,
            "nq_swap",
            "model_a",
            "greedy",
            timestamps=["20260409_120000", "20260410_080000"],
        )
        experiments = scan_local_experiments(tmp_path)
        assert len(experiments) == 1
        runs = experiments[0]["runs"]
        assert len(runs) == 2
        timestamps = {r["timestamp"] for r in runs}
        assert timestamps == {"20260409_120000", "20260410_080000"}


# ---------------------------------------------------------------------------
# test_scan_skips_dirs_without_results
# ---------------------------------------------------------------------------
class TestScanSkipsDirsWithoutResults:
    def test_dir_with_only_generations_skipped(self, tmp_path: Path):
        _make_experiment(
            tmp_path,
            "nq_swap",
            "model_a",
            "greedy",
            timestamps=[],
            add_generations=True,
        )
        experiments = scan_local_experiments(tmp_path)
        assert len(experiments) == 0

    def test_empty_dir_skipped(self, tmp_path: Path):
        variant_dir = tmp_path / "task" / "model" / "variant"
        variant_dir.mkdir(parents=True)
        experiments = scan_local_experiments(tmp_path)
        assert len(experiments) == 0


# ---------------------------------------------------------------------------
# test_scan_collects_all_files
# ---------------------------------------------------------------------------
class TestScanCollectsAllFiles:
    def test_results_and_config_in_files(self, tmp_path: Path):
        _make_experiment(
            tmp_path,
            "nq_swap",
            "model_a",
            "greedy",
            timestamps=["20260409_120000"],
            add_config=True,
        )
        experiments = scan_local_experiments(tmp_path)
        run = experiments[0]["runs"][0]
        assert "results_20260409_120000.jsonl" in run["files"]
        assert "results_20260409_120000_config.json" in run["files"]

    def test_results_without_config(self, tmp_path: Path):
        _make_experiment(
            tmp_path,
            "nq_swap",
            "model_a",
            "greedy",
            timestamps=["20260409_120000"],
            add_config=False,
        )
        experiments = scan_local_experiments(tmp_path)
        run = experiments[0]["runs"][0]
        assert "results_20260409_120000.jsonl" in run["files"]
        assert len(run["files"]) == 1

    def test_config_hash_present_when_config_exists(self, tmp_path: Path):
        _make_experiment(
            tmp_path,
            "nq_swap",
            "model_a",
            "greedy",
            timestamps=["20260409_120000"],
            add_config=True,
        )
        experiments = scan_local_experiments(tmp_path)
        run = experiments[0]["runs"][0]
        assert run["config_hash"] is not None
        assert isinstance(run["config_hash"], str)
        assert len(run["config_hash"]) == 64  # SHA-256 hex digest

    def test_config_hash_none_without_config(self, tmp_path: Path):
        _make_experiment(
            tmp_path,
            "nq_swap",
            "model_a",
            "greedy",
            timestamps=["20260409_120000"],
            add_config=False,
        )
        experiments = scan_local_experiments(tmp_path)
        run = experiments[0]["runs"][0]
        assert run["config_hash"] is None


# ---------------------------------------------------------------------------
# test_scan_includes_generations
# ---------------------------------------------------------------------------
class TestScanIncludesGenerations:
    def test_generation_files_populated(self, tmp_path: Path):
        _make_experiment(
            tmp_path,
            "nq_swap",
            "model_a",
            "greedy",
            timestamps=["20260409_120000"],
            add_generations=True,
        )
        experiments = scan_local_experiments(tmp_path)
        gen_files = experiments[0]["generation_files"]
        assert "generations.jsonl" in gen_files

    def test_no_generations(self, tmp_path: Path):
        _make_experiment(
            tmp_path,
            "nq_swap",
            "model_a",
            "greedy",
            timestamps=["20260409_120000"],
            add_generations=False,
        )
        experiments = scan_local_experiments(tmp_path)
        assert experiments[0]["generation_files"] == []


# ---------------------------------------------------------------------------
# test_descriptor_fields
# ---------------------------------------------------------------------------
class TestDescriptorFields:
    def test_task_model_variant_fields(self, tmp_path: Path):
        _make_experiment(tmp_path, "nq_swap", "meta-llama_Llama-3-8B", "decore_niah")
        experiments = scan_local_experiments(tmp_path)
        exp = experiments[0]
        assert exp["task"] == "nq_swap"
        assert exp["model_slug"] == "meta-llama_Llama-3-8B"
        assert exp["variant"] == "decore_niah"

    def test_local_path_is_absolute(self, tmp_path: Path):
        _make_experiment(tmp_path, "nq_swap", "model_a", "greedy")
        experiments = scan_local_experiments(tmp_path)
        assert experiments[0]["local_path"].is_absolute()


# ---------------------------------------------------------------------------
# test_metrics_summary
# ---------------------------------------------------------------------------
class TestMetricsSummary:
    def test_computes_mean_of_scores(self, tmp_path: Path):
        _make_experiment(
            tmp_path,
            "nq_swap",
            "model_a",
            "greedy",
            timestamps=["20260409_120000"],
            n_samples=3,
            scores={"sub_em": 0.6, "org_em": 0.3},
        )
        experiments = scan_local_experiments(tmp_path)
        metrics = experiments[0]["runs"][0]["metrics"]
        # All samples have identical scores, so mean == value
        assert metrics["sub_em"] == pytest.approx(0.6)
        assert metrics["org_em"] == pytest.approx(0.3)

    def test_metrics_with_varying_scores(self, tmp_path: Path):
        variant_dir = tmp_path / "nq_swap" / "model_a" / "greedy"
        variant_dir.mkdir(parents=True)
        # Write results with varying scores
        lines = [
            json.dumps(
                {"sample_id": 0, "prompt": "p", "output": "o", "target": "t", "scores": {"acc": 1.0}, "metadata": {}}
            ),
            json.dumps(
                {"sample_id": 1, "prompt": "p", "output": "o", "target": "t", "scores": {"acc": 0.0}, "metadata": {}}
            ),
        ]
        results_path = variant_dir / "results_20260409_120000.jsonl"
        results_path.write_text("\n".join(lines) + "\n")
        experiments = scan_local_experiments(tmp_path)
        assert experiments[0]["runs"][0]["metrics"]["acc"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# test_config_hash helper
# ---------------------------------------------------------------------------
class TestConfigHash:
    def test_excludes_timestamp(self, tmp_path: Path):
        cfg1 = dict(_SAMPLE_CONFIG, timestamp="20260409_120000")
        cfg2 = dict(_SAMPLE_CONFIG, timestamp="20260410_080000")
        p1 = tmp_path / "cfg1.json"
        p2 = tmp_path / "cfg2.json"
        p1.write_text(json.dumps(cfg1))
        p2.write_text(json.dumps(cfg2))
        assert config_hash(p1) == config_hash(p2)

    def test_different_config_different_hash(self, tmp_path: Path):
        cfg1 = dict(_SAMPLE_CONFIG)
        cfg2 = dict(_SAMPLE_CONFIG, limit=200)
        p1 = tmp_path / "cfg1.json"
        p2 = tmp_path / "cfg2.json"
        p1.write_text(json.dumps(cfg1))
        p2.write_text(json.dumps(cfg2))
        assert config_hash(p1) != config_hash(p2)

    def test_returns_64_char_hex(self, tmp_path: Path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(_SAMPLE_CONFIG))
        h = config_hash(p)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# test_compute_metrics_summary helper
# ---------------------------------------------------------------------------
class TestComputeMetricsSummary:
    def test_mean_of_single_metric(self, tmp_path: Path):
        lines = [
            json.dumps({"scores": {"acc": 1.0}}),
            json.dumps({"scores": {"acc": 0.5}}),
            json.dumps({"scores": {"acc": 0.0}}),
        ]
        p = tmp_path / "results.jsonl"
        p.write_text("\n".join(lines) + "\n")
        metrics = compute_metrics_summary(p)
        assert metrics["acc"] == pytest.approx(0.5)

    def test_multiple_metrics(self, tmp_path: Path):
        lines = [
            json.dumps({"scores": {"acc": 1.0, "f1": 0.8}}),
            json.dumps({"scores": {"acc": 0.0, "f1": 0.6}}),
        ]
        p = tmp_path / "results.jsonl"
        p.write_text("\n".join(lines) + "\n")
        metrics = compute_metrics_summary(p)
        assert metrics["acc"] == pytest.approx(0.5)
        assert metrics["f1"] == pytest.approx(0.7)

    def test_empty_file_returns_empty_dict(self, tmp_path: Path):
        p = tmp_path / "results.jsonl"
        p.write_text("")
        metrics = compute_metrics_summary(p)
        assert metrics == {}
