"""Tests for ExperimentKey dataclass."""

import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest

from locos_eval.evals.experiment_key import ExperimentKey


# ---------------------------------------------------------------------------
# Model slug
# ---------------------------------------------------------------------------
class TestExperimentKeyModelSlug:
    def test_replaces_slash(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="greedy",
        )
        assert ek.model_slug == "meta-llama_Meta-Llama-3-8B-Instruct"

    def test_preserves_casing(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="Qwen/Qwen3-32B",
            decoding="greedy",
        )
        assert ek.model_slug == "Qwen_Qwen3-32B"


# ---------------------------------------------------------------------------
# Job slug
# ---------------------------------------------------------------------------
class TestExperimentKeyJobSlug:
    def test_lowercase_dns_safe(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="greedy",
        )
        assert ek.job_slug == "meta-llama-3-8b-instruct"

    def test_dots_replaced(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Llama-3.1-8B-Instruct",
            decoding="greedy",
        )
        assert ek.job_slug == "llama-3-1-8b-instruct"


# ---------------------------------------------------------------------------
# Variant
# ---------------------------------------------------------------------------
class TestExperimentKeyVariant:
    def test_greedy(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="greedy",
        )
        assert ek.variant == "greedy"

    def test_ablation_niah_default(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="ablation",
            heads_path="retrieval_heads/Meta-Llama-3-8B-Instruct.json",
        )
        assert ek.variant == "ablation_niah"

    def test_ablation_nolima(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="ablation",
            heads_path="retrieval_heads/Meta-Llama-3-8B-Instruct_nolima.json",
        )
        assert ek.variant == "ablation_nolima"

    def test_ablation_niah(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="ablation",
            heads_path="retrieval_heads/Meta-Llama-3-8B-Instruct.json",
        )
        assert ek.variant == "ablation_niah"

    def test_ablation_random(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="ablation",
            heads_path="random",
            random_seed=42,
            num_heads=20,
        )
        assert ek.variant == "ablation_random_s42_n20"

    def test_ablation_mean_default_label(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="ablation",
            heads_path="retrieval_heads/Meta-Llama-3-8B-Instruct.json",
            ablation_mode="mean",
        )
        assert ek.variant == "ablation_mean_niah"

    def test_ablation_mean_explicit_label(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="ablation",
            heads_path="retrieval_heads/Meta-Llama-3-8B-Instruct_nolima.json",
            ablation_mode="mean",
        )
        assert ek.variant == "ablation_mean_nolima"

    def test_ablation_zero_default_unchanged(self):
        """Default ablation_mode='zero' must not change existing variant names."""
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="ablation",
            heads_path="retrieval_heads/Meta-Llama-3-8B-Instruct.json",
        )
        assert ek.variant == "ablation_niah"
        assert ek.ablation_mode == "zero"

    def test_explicit_heads_label_overrides(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="ablation",
            heads_path="retrieval_heads/Meta-Llama-3-8B-Instruct.json",
            heads_label="ori",
        )
        assert ek.variant == "ablation_ori"

    def test_random_default_seed_and_count(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="ablation",
            heads_path="random",
        )
        # Default seed=42, num_heads=None → 50
        assert ek.variant == "ablation_random_s42_n50"

    def test_empty_string_heads_label_produces_bare_decoding(self):
        """heads_label="" is an explicit override producing no label suffix."""
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="ablation",
            heads_path="retrieval_heads/Model.json",
            heads_label="",
        )
        # Empty label → bare decoding mode (no "_niah" suffix)
        assert ek.variant == "ablation"


# ---------------------------------------------------------------------------
# Key
# ---------------------------------------------------------------------------
class TestExperimentKeyKey:
    def test_full_key(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="ablation",
            heads_path="retrieval_heads/Meta-Llama-3-8B-Instruct_nolima.json",
        )
        assert ek.key == "nq_swap/meta-llama_Meta-Llama-3-8B-Instruct/ablation_nolima"

    def test_greedy_key(self):
        ek = ExperimentKey(
            task="aci_bench",
            model="Qwen/Qwen3-32B",
            decoding="greedy",
        )
        assert ek.key == "aci_bench/Qwen_Qwen3-32B/greedy"


# ---------------------------------------------------------------------------
# Local dir
# ---------------------------------------------------------------------------
class TestExperimentKeyLocalDir:
    def test_default_output_dir(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="greedy",
        )
        expected = Path("eval_results") / "nq_swap" / "meta-llama_Meta-Llama-3-8B-Instruct" / "greedy"
        assert ek.local_dir() == expected

    def test_custom_output_dir(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="ablation",
            heads_path="retrieval_heads/Meta-Llama-3-8B-Instruct.json",
        )
        expected = Path("/tmp/results") / "nq_swap" / "meta-llama_Meta-Llama-3-8B-Instruct" / "ablation_niah"
        assert ek.local_dir("/tmp/results") == expected


# ---------------------------------------------------------------------------
# Frozen / hashable
# ---------------------------------------------------------------------------
class TestExperimentKeyFrozen:
    def test_is_immutable(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="greedy",
        )
        with pytest.raises(AttributeError):
            ek.task = "other"  # type: ignore[misc]

    def test_is_hashable(self):
        ek1 = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="greedy",
        )
        ek2 = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="greedy",
        )
        assert hash(ek1) == hash(ek2)
        assert ek1 == ek2
        assert len({ek1, ek2}) == 1


# ---------------------------------------------------------------------------
# CLI (__main__)
# ---------------------------------------------------------------------------
class TestExperimentKeyCLI:
    _BASE_CMD: ClassVar[list[str]] = [
        sys.executable,
        "-m",
        "locos_eval.evals.experiment_key",
        "--task",
        "nq_swap",
        "--model",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "--decoding",
        "ablation",
        "--heads",
        "retrieval_heads/Meta-Llama-3-8B-Instruct_nolima.json",
    ]

    def _run(self, extra_args: list[str]) -> str:
        result = subprocess.run(
            self._BASE_CMD + extra_args,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def test_variant_output(self):
        assert self._run(["--variant"]) == "ablation_nolima"

    def test_key_output(self):
        assert self._run(["--key"]) == "nq_swap/meta-llama_Meta-Llama-3-8B-Instruct/ablation_nolima"

    def test_model_slug_output(self):
        assert self._run(["--model-slug"]) == "meta-llama_Meta-Llama-3-8B-Instruct"

    def test_local_dir_output(self):
        out = self._run(["--local-dir"])
        expected = str(Path("eval_results") / "nq_swap" / "meta-llama_Meta-Llama-3-8B-Instruct" / "ablation_nolima")
        assert out == expected

    def test_mutually_exclusive_flags(self):
        result = subprocess.run(
            [*self._BASE_CMD, "--variant", "--key"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Sampling seed
# ---------------------------------------------------------------------------
class TestExperimentKeyVariantSamplingSeed:
    def test_greedy_with_seed(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="greedy",
            sampling_seed=1,
        )
        assert ek.variant == "greedy_s1"

    def test_greedy_without_seed(self):
        """Default sampling_seed=None produces bare 'greedy'."""
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="greedy",
        )
        assert ek.variant == "greedy"

    def test_ablation_nolima_with_seed(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="ablation",
            heads_path="retrieval_heads/Meta-Llama-3-8B-Instruct_nolima.json",
            sampling_seed=2,
        )
        assert ek.variant == "ablation_nolima_s2"

    def test_ablation_with_seed(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="ablation",
            heads_path="retrieval_heads/Meta-Llama-3-8B-Instruct_nolima.json",
            sampling_seed=3,
        )
        assert ek.variant == "ablation_nolima_s3"

    def test_ablation_random_with_seed(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="ablation",
            heads_path="random",
            random_seed=42,
            num_heads=20,
            sampling_seed=1,
        )
        assert ek.variant == "ablation_random_s42_n20_s1"

    def test_key_includes_seed(self):
        ek = ExperimentKey(
            task="nq_swap",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            decoding="greedy",
            sampling_seed=1,
        )
        assert ek.key == "nq_swap/meta-llama_Meta-Llama-3-8B-Instruct/greedy_s1"

    def test_sampling_seed_must_be_positive(self):
        with pytest.raises(AssertionError):
            ExperimentKey(
                task="nq_swap",
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                decoding="greedy",
                sampling_seed=0,
            )


class TestExperimentKeyCLISamplingSeed:
    def test_variant_with_seed(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "locos_eval.evals.experiment_key",
                "--task",
                "nq_swap",
                "--model",
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "--decoding",
                "greedy",
                "--sampling-seed",
                "1",
                "--variant",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "greedy_s1"
