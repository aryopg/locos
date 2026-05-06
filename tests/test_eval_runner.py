"""Tests for the standalone eval runner base class."""

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from locos_eval.evals.runner import EvalResult, EvalRunner, EvalSample, _extract_layer_head_counts


# ---------------------------------------------------------------------------
# EvalSample
# ---------------------------------------------------------------------------
class TestEvalSample:
    def test_creation(self):
        """EvalSample stores prompt, target, and optional metadata."""
        sample = EvalSample(prompt="What is 2+2?", target="4")
        assert sample.prompt == "What is 2+2?"
        assert sample.target == "4"
        assert sample.metadata == {}

    def test_creation_with_metadata(self):
        sample = EvalSample(prompt="p", target="t", metadata={"source": "test"})
        assert sample.metadata["source"] == "test"


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------
class TestEvalResult:
    def test_creation(self):
        """EvalResult stores all fields including scores dict."""
        result = EvalResult(
            sample_id=0,
            output="output",
            target="target",
            scores={"accuracy": 1.0},
        )
        assert result.sample_id == 0
        assert result.scores["accuracy"] == 1.0
        assert result.metadata == {}

    def test_save_load_jsonl(self, tmp_path: Path):
        """Round-trip save and load preserves data."""
        results = [
            EvalResult(
                sample_id=0,
                output="o0",
                target="t0",
                scores={"acc": 0.5, "f1": 0.8},
                metadata={"key": "val"},
            ),
            EvalResult(
                sample_id=1,
                output="o1",
                target="t1",
                scores={"acc": 1.0, "f1": 0.9},
            ),
        ]
        path = tmp_path / "results.jsonl"

        EvalResult.save_jsonl(results, path)
        assert path.exists()

        loaded = EvalResult.load_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0]["sample_id"] == 0
        assert loaded[0]["scores"]["acc"] == 0.5
        assert loaded[0]["metadata"]["key"] == "val"
        assert loaded[1]["sample_id"] == 1
        assert loaded[1]["scores"]["f1"] == 0.9
        assert loaded[1]["metadata"] == {}

    def test_save_jsonl_creates_parent_dirs(self, tmp_path: Path):
        """save_jsonl creates missing parent directories."""
        results = [EvalResult(sample_id=0, output="o", target="t", scores={"a": 1.0})]
        path = tmp_path / "sub" / "dir" / "results.jsonl"

        EvalResult.save_jsonl(results, path)
        assert path.exists()

    def test_save_jsonl_empty_raises(self, tmp_path: Path):
        """save_jsonl raises on empty results list."""
        with pytest.raises(AssertionError, match="Cannot save empty"):
            EvalResult.save_jsonl([], tmp_path / "empty.jsonl")

    def test_load_jsonl_missing_raises(self, tmp_path: Path):
        """load_jsonl raises when file does not exist."""
        with pytest.raises(AssertionError, match="not found"):
            EvalResult.load_jsonl(tmp_path / "nonexistent.jsonl")


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------
class TestEvalRunner:
    def _make_runner(self, **overrides):
        """Create a minimal EvalRunner with defaults."""
        defaults = dict(model="test/model", heads="heads.json")
        defaults.update(overrides)
        return EvalRunner(**defaults)

    def test_format_prompt_with_system_message(self):
        """_format_prompt uses apply_chat_template when available."""
        runner = self._make_runner()

        # Mock tokenizer with apply_chat_template
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        )
        runner._tokenizer = tokenizer

        # Override system_message
        runner.system_message = lambda: "You are helpful."

        sample = EvalSample(prompt="Hello", target="Hi")
        result = runner._format_prompt(sample)

        tokenizer.apply_chat_template.assert_called_once()
        call_args = tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
        assert call_args[1]["tokenize"] is False
        assert call_args[1]["add_generation_prompt"] is True
        assert "<|im_start|>" in result

    def test_format_prompt_no_system_message(self):
        """_format_prompt omits system message when system_message() returns None."""
        runner = self._make_runner()

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|user|>Hello<|assistant|>"
        runner._tokenizer = tokenizer

        sample = EvalSample(prompt="Hello", target="Hi")
        runner._format_prompt(sample)

        messages = tokenizer.apply_chat_template.call_args[0][0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_format_prompt_passes_enable_thinking_when_template_supports_it(self):
        """If the chat template references enable_thinking, we pin it to True."""
        runner = self._make_runner()

        tokenizer = MagicMock()
        # Stand-in chat template that exposes the Jinja variable.
        tokenizer.chat_template = "{% if enable_thinking %}<think>{% endif %}{{ messages[0].content }}"
        tokenizer.apply_chat_template.return_value = "<think>Hello"
        runner._tokenizer = tokenizer

        sample = EvalSample(prompt="Hello", target="x")
        runner._format_prompt(sample)

        kwargs = tokenizer.apply_chat_template.call_args.kwargs
        assert kwargs.get("enable_thinking") is True

    def test_format_prompt_omits_enable_thinking_when_template_lacks_it(self):
        """Plain templates (Llama-3 etc.) must not get enable_thinking — would error."""
        runner = self._make_runner()

        tokenizer = MagicMock()
        tokenizer.chat_template = "{{ messages[0].content }}"  # no thinking support
        tokenizer.apply_chat_template.return_value = "Hello"
        runner._tokenizer = tokenizer

        sample = EvalSample(prompt="Hello", target="x")
        runner._format_prompt(sample)

        kwargs = tokenizer.apply_chat_template.call_args.kwargs
        assert "enable_thinking" not in kwargs

    def test_format_prompt_fallback(self):
        """_format_prompt falls back to plain text when apply_chat_template is absent."""
        runner = self._make_runner()

        # Tokenizer without apply_chat_template
        tokenizer = MagicMock(spec=[])  # empty spec => no attributes
        runner._tokenizer = tokenizer

        runner.system_message = lambda: "Be concise."

        sample = EvalSample(prompt="What is AI?", target="answer")
        result = runner._format_prompt(sample)

        assert "System: Be concise." in result
        assert "User: What is AI?" in result

    def test_add_common_args(self):
        """add_common_args registers all expected CLI arguments."""
        parser = argparse.ArgumentParser()
        EvalRunner.add_common_args(parser)

        # Parse with required args only — configurable params default to None
        args = parser.parse_args(["--model", "m", "--heads", "h"])
        assert args.model == "m"
        assert args.heads == "h"
        # Configurable params are None before resolve_args
        assert args.max_tokens is None
        assert args.temperature is None
        assert args.sampling_top_p is None
        assert args.sampling_top_k is None
        assert args.max_model_len is None
        assert args.tp is None
        assert args.gpu_mem is None
        assert args.limit is None
        assert args.output_dir == "eval_results"
        assert args.decoding == "greedy"
        assert args.num_heads is None
        assert args.random_seed == 42

    def test_resolve_args_defaults(self):
        """resolve_args fills None params from _default.yaml when no model YAML exists."""
        from locos_eval.evals.model_config import load_model_config

        parser = argparse.ArgumentParser()
        EvalRunner.add_common_args(parser)
        args = parser.parse_args(["--model", "nonexistent/FakeModel", "--heads", "h"])
        EvalRunner.resolve_args(args)
        # Should match _default.yaml (which layers on top of hardcoded DEFAULTS)
        expected = load_model_config("nonexistent/FakeModel")
        assert args.max_tokens == expected["max_tokens"]
        assert args.temperature == expected["temperature"]
        assert args.sampling_top_p == expected["sampling_top_p"]
        assert args.sampling_top_k == expected["sampling_top_k"]
        assert args.tp == expected["tensor_parallel_size"]
        assert args.gpu_mem == expected["gpu_memory_utilization"]

    def test_resolve_args_cli_overrides(self):
        """CLI args take priority over YAML/defaults."""
        parser = argparse.ArgumentParser()
        EvalRunner.add_common_args(parser)
        args = parser.parse_args(
            ["--model", "nonexistent/FakeModel", "--heads", "h", "--max-tokens", "256", "--temperature", "0.7"]
        )
        EvalRunner.resolve_args(args)
        # Explicitly set via CLI — should be the CLI values
        assert args.max_tokens == 256
        assert args.temperature == 0.7

    def test_resolve_args_ablation_defaults_to_top_50_heads(self):
        """Ablation CLI runs should default to the top 50 ranked heads."""
        parser = argparse.ArgumentParser()
        EvalRunner.add_common_args(parser)
        args = parser.parse_args(["--model", "nonexistent/FakeModel", "--heads", "h", "--decoding", "ablation"])
        EvalRunner.resolve_args(args)
        assert args.num_heads == 50

    def test_resolve_args_preserves_explicit_num_heads(self):
        """Explicit --num-heads should override the ablation default."""
        parser = argparse.ArgumentParser()
        EvalRunner.add_common_args(parser)
        args = parser.parse_args(
            ["--model", "nonexistent/FakeModel", "--heads", "h", "--decoding", "ablation", "--num-heads", "20"]
        )
        EvalRunner.resolve_args(args)
        assert args.num_heads == 20

    def test_task_name_default(self):
        """Default task_name returns class name."""
        runner = self._make_runner()
        assert runner.task_name() == "EvalRunner"

    def test_load_samples_not_implemented(self):
        """Base class load_samples raises NotImplementedError."""
        runner = self._make_runner()
        with pytest.raises(NotImplementedError):
            runner.load_samples()

    def test_score_not_implemented(self):
        """Base class score raises NotImplementedError."""
        runner = self._make_runner()
        with pytest.raises(NotImplementedError):
            runner.score("output", EvalSample(prompt="p", target="t"))

    def test_init_validation(self):
        """Constructor validates parameter ranges."""
        with pytest.raises(AssertionError):
            self._make_runner(max_tokens=0)
        with pytest.raises(AssertionError):
            self._make_runner(temperature=-1)
        with pytest.raises(AssertionError):
            self._make_runner(gpu_memory_utilization=0)
        with pytest.raises(AssertionError):
            self._make_runner(gpu_memory_utilization=1.5)
        with pytest.raises(AssertionError):
            self._make_runner(limit=0)

    def test_kwargs_stored(self):
        """Extra kwargs are stored for passing to ablation()."""
        runner = self._make_runner(decoding="greedy")
        assert runner._ablation_kwargs["decoding"] == "greedy"

    def test_heads_required_for_ablation_mode(self):
        """ablation decoding mode requires --heads."""
        with pytest.raises(ValueError, match="--heads is required"):
            EvalRunner(model="test/model", heads=None, decoding="ablation")

    def test_greedy_mode_does_not_require_heads(self):
        """greedy decoding mode should work without heads."""
        runner = EvalRunner(model="test/model", heads=None, decoding="greedy")
        assert runner._heads_path is None

    def test_experiment_key_property(self):
        """Runner should expose an ExperimentKey."""
        from locos_eval.evals.experiment_key import ExperimentKey

        runner = self._make_runner(decoding="ablation")
        ek = runner.experiment_key
        assert isinstance(ek, ExperimentKey)
        assert ek.task == "EvalRunner"
        assert ek.model == "test/model"
        assert ek.variant == "ablation_niah"


# ---------------------------------------------------------------------------
# Heads label and variant naming
# ---------------------------------------------------------------------------
class TestHeadsLabel:
    def _make_runner(self, **overrides):
        defaults = dict(model="test/model", heads="heads.json")
        defaults.update(overrides)
        return EvalRunner(**defaults)

    def test_niah_default(self):
        """Heads file without suffix → 'niah' label."""
        runner = self._make_runner(heads="retrieval_heads/Qwen3-32B.json")
        assert runner._heads_label() == "niah"

    def test_nolima_suffix(self):
        """Heads file with _nolima suffix → 'nolima' label."""
        runner = self._make_runner(heads="retrieval_heads/Qwen3-32B_nolima.json")
        assert runner._heads_label() == "nolima"

    def test_explicit_override(self):
        """--heads-label overrides auto-inference."""
        runner = self._make_runner(heads="retrieval_heads/Qwen3-32B.json", heads_label="custom")
        assert runner._heads_label() == "custom"

    def test_greedy_empty_label(self):
        """Greedy mode (no heads) → empty label."""
        runner = EvalRunner(model="test/model", heads=None, decoding="greedy")
        assert runner._heads_label() == ""

    def test_run_variant_greedy(self):
        """Greedy decoding → variant name 'greedy'."""
        runner = EvalRunner(model="test/model", heads=None, decoding="greedy")
        assert runner._run_variant() == "greedy"

    def test_run_variant_ablation_niah(self):
        """Ablation with NIAH heads → variant name 'ablation_niah'."""
        runner = self._make_runner(heads="retrieval_heads/Qwen3-32B.json", decoding="ablation")
        assert runner._run_variant() == "ablation_niah"


# ---------------------------------------------------------------------------
# Generation checkpointing
# ---------------------------------------------------------------------------
class TestCheckpointing:
    def test_save_and_load_generations(self, tmp_path):
        """Generations should round-trip through save/load."""
        runner = EvalRunner(model="test/model", heads="h.json")
        gen_path = tmp_path / "generations.jsonl"

        sample = EvalSample(prompt="What is 2+2?", target="4", metadata={"source": "test"})
        runner._save_generation(gen_path, 0, "four", sample)
        runner._save_generation(gen_path, 1, "five", sample)

        records = runner._load_generations(gen_path)
        assert len(records) == 2
        assert records[0]["sample_id"] == 0
        assert records[0]["output"] == "four"
        assert records[1]["sample_id"] == 1
        assert records[1]["output"] == "five"

    def test_load_nonexistent_returns_empty(self, tmp_path):
        """Loading from a nonexistent file should return empty list."""
        runner = EvalRunner(model="test/model", heads="h.json")
        records = runner._load_generations(tmp_path / "nope.jsonl")
        assert records == []


# ---------------------------------------------------------------------------
# Full run() pipeline with concrete subclass
# ---------------------------------------------------------------------------
class TestRunPipeline:
    """Test the full run() loop with a trivial concrete subclass."""

    def test_score_only_pipeline(self, tmp_path):
        """Score-only mode should load generations and score without model."""

        class TrivialTask(EvalRunner):
            def task_name(self):
                return "trivial"

            def load_samples(self):
                return [
                    EvalSample(prompt="q1", target="a1"),
                    EvalSample(prompt="q2", target="a2"),
                ]

            def score(self, output, sample):
                # Simple exact-match scorer
                return {"exact_match": 1.0 if output.strip() == sample.target else 0.0}

        runner = TrivialTask(model="test/model", heads="h.json", output_dir=str(tmp_path))

        # Create a fake generations file
        gen_path = tmp_path / "gens.jsonl"
        for i, (out, tgt) in enumerate([("a1", "a1"), ("wrong", "a2")]):
            record = {
                "sample_id": i,
                "prompt": f"q{i+1}",
                "output": out,
                "target": tgt,
                "metadata": {},
            }
            with open(gen_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        results = runner.run(score_only=str(gen_path))

        assert len(results) == 2
        assert results[0].scores["exact_match"] == 1.0
        assert results[1].scores["exact_match"] == 0.0
        assert results[0].output == "a1"
        assert results[1].output == "wrong"

        # Verify output files were created
        run_dir = runner._run_dir()
        assert run_dir.exists()
        result_files = list(run_dir.glob("results_*.jsonl"))
        config_files = list(run_dir.glob("results_*_config.json"))
        assert len(result_files) == 1
        assert len(config_files) == 1

    def _make_task_with_mock_model(self, task_cls, tmp_path, generate_side_effect=None):
        """Helper to create a task with mocked _init_model."""
        runner = task_cls(
            model="test/model",
            heads="h.json",
            output_dir=str(tmp_path),
        )
        mock_wrapper = MagicMock()
        # MagicMock auto-creates truthy attributes; the runner branches on
        # ``supports_batch`` so we must pin it to False to keep these tests on
        # the per-sample loop. Wrappers that opt into batching (Greedy,
        # Ablation, AblationRPC) set the flag to True explicitly.
        mock_wrapper.supports_batch = False
        if generate_side_effect:
            mock_wrapper.generate.side_effect = generate_side_effect
        else:
            mock_wrapper.generate.return_value = "output"
        mock_tokenizer = MagicMock(spec=[])  # no apply_chat_template

        def fake_init_model():
            runner._wrapper = mock_wrapper
            runner._tokenizer = mock_tokenizer

        runner._init_model = fake_init_model
        return runner, mock_wrapper

    def test_generation_pipeline_with_mocked_wrapper(self, tmp_path):
        """Full generation + scoring pipeline with mocked model."""

        class CountingTask(EvalRunner):
            def task_name(self):
                return "counting"

            def load_samples(self):
                return [
                    EvalSample(prompt="q1", target="answer1"),
                    EvalSample(prompt="q2", target="answer2"),
                    EvalSample(prompt="q3", target="answer3"),
                ]

            def score(self, output, sample):
                return {"match": 1.0 if output == sample.target else 0.0}

        call_count = 0

        def fake_generate(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            targets = ["answer1", "wrong", "answer3"]
            return targets[call_count - 1]

        runner, mock_wrapper = self._make_task_with_mock_model(
            CountingTask,
            tmp_path,
            generate_side_effect=fake_generate,
        )

        results = runner.run()

        assert len(results) == 3
        assert results[0].scores["match"] == 1.0
        assert results[1].scores["match"] == 0.0
        assert results[2].scores["match"] == 1.0
        assert mock_wrapper.generate.call_count == 3

        # Verify checkpoint was written
        gen_path = runner._generations_path()
        assert gen_path.exists()
        records = runner._load_generations(gen_path)
        assert len(records) == 3

    def test_generation_resumes_from_checkpoint(self, tmp_path):
        """Generation should resume from existing checkpoint, not regenerate."""

        class ResumableTask(EvalRunner):
            def task_name(self):
                return "resumable"

            def load_samples(self):
                return [
                    EvalSample(prompt="q1", target="a1"),
                    EvalSample(prompt="q2", target="a2"),
                    EvalSample(prompt="q3", target="a3"),
                ]

            def score(self, output, sample):
                return {"ok": 1.0}

        runner, mock_wrapper = self._make_task_with_mock_model(ResumableTask, tmp_path)
        mock_wrapper.generate.return_value = "generated_a3"

        # Pre-populate checkpoint with 2 of 3 samples
        gen_path = runner._generations_path()
        gen_path.parent.mkdir(parents=True, exist_ok=True)
        for i, out in enumerate(["cached_a1", "cached_a2"]):
            record = {
                "sample_id": i,
                "prompt": f"q{i+1}",
                "output": out,
                "target": f"a{i+1}",
                "metadata": {},
            }
            with open(gen_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        results = runner.run()

        assert len(results) == 3
        # First two should use cached outputs
        assert results[0].output == "cached_a1"
        assert results[1].output == "cached_a2"
        # Third should be newly generated
        assert results[2].output == "generated_a3"
        # Wrapper should only have been called once
        assert mock_wrapper.generate.call_count == 1

    def test_limit_restricts_sample_count(self, tmp_path):
        """--limit should restrict how many samples are processed."""

        class LimitTask(EvalRunner):
            def task_name(self):
                return "limit_test"

            def load_samples(self):
                return [EvalSample(prompt=f"q{i}", target=f"a{i}") for i in range(10)]

            def score(self, output, sample):
                return {"ok": 1.0}

        runner = LimitTask(
            model="test/model",
            heads="h.json",
            output_dir=str(tmp_path),
            limit=3,
        )

        mock_wrapper = MagicMock()
        mock_wrapper.supports_batch = False  # see comment in _make_task_with_mock_model
        mock_wrapper.generate.return_value = "output"
        mock_tokenizer = MagicMock(spec=[])

        def fake_init_model():
            runner._wrapper = mock_wrapper
            runner._tokenizer = mock_tokenizer

        runner._init_model = fake_init_model

        results = runner.run()

        assert len(results) == 3
        assert mock_wrapper.generate.call_count == 3


# ---------------------------------------------------------------------------
# Ablation mode
# ---------------------------------------------------------------------------
class TestAblationMode:
    def _make_runner(self, **overrides):
        defaults = dict(model="test/model", heads="heads.json")
        defaults.update(overrides)
        return EvalRunner(**defaults)

    def test_runner_accepts_ablation_decoding(self):
        """Runner should accept --decoding ablation with a heads file."""
        runner = self._make_runner(decoding="ablation")
        assert runner._ablation_kwargs["decoding"] == "ablation"

    def test_runner_ablation_requires_heads(self):
        """Ablation mode without heads should raise."""
        with pytest.raises(ValueError, match="--heads is required"):
            EvalRunner(model="test", heads=None, decoding="ablation")

    def test_ablation_variant_with_niah_heads(self):
        """Ablation variant should be 'ablation_niah'."""
        runner = self._make_runner(heads="retrieval_heads/Model.json", decoding="ablation")
        assert runner._run_variant() == "ablation_niah"

    def test_ablation_variant_with_nolima_heads(self):
        """Ablation variant with nolima heads."""
        runner = self._make_runner(heads="retrieval_heads/Model_nolima.json", decoding="ablation")
        assert runner._run_variant() == "ablation_nolima"

    def test_ablation_defaults_to_top_50_heads(self):
        """File-based ablation should use the top 50 ranked heads by default."""
        runner = self._make_runner(heads="retrieval_heads/Model_nolima.json", decoding="ablation")
        assert runner.experiment_key.num_heads == 50

    def test_random_heads_label(self):
        """Random heads should produce label 'random_s{seed}_n{count}'."""
        runner = self._make_runner(heads="random", decoding="ablation", num_heads=20, random_seed=42)
        assert runner._heads_label() == "random_s42_n20"

    def test_random_heads_variant(self):
        """Ablation with random heads should have variant 'ablation_random_s42_n20'."""
        runner = self._make_runner(heads="random", decoding="ablation", num_heads=20, random_seed=42)
        assert runner._run_variant() == "ablation_random_s42_n20"

    def test_random_heads_default_seed(self):
        """Random heads without explicit seed uses default 42."""
        runner = self._make_runner(heads="random", decoding="ablation", num_heads=10)
        assert runner._heads_label() == "random_s42_n10"

    def test_random_heads_default_count(self):
        """Random heads without explicit count uses default 50."""
        runner = self._make_runner(heads="random", decoding="ablation")
        assert runner._heads_label() == "random_s42_n50"


class TestSamplingSeed:
    """Tests for sampling_seed propagation through runner → ExperimentKey."""

    def _make_runner(self, **overrides):
        defaults = dict(model="test/model", heads="heads.json")
        defaults.update(overrides)
        return EvalRunner(**defaults)

    def test_experiment_key_includes_sampling_seed(self):
        runner = self._make_runner(decoding="greedy", heads=None, sampling_seed=1)
        assert runner.experiment_key.sampling_seed == 1
        assert runner.experiment_key.variant == "greedy_s1"

    def test_experiment_key_no_sampling_seed(self):
        runner = self._make_runner(decoding="greedy", heads=None)
        assert runner.experiment_key.sampling_seed is None
        assert runner.experiment_key.variant == "greedy"

    def test_ablation_with_sampling_seed(self):
        runner = self._make_runner(
            heads="retrieval_heads/Model_nolima.json",
            decoding="ablation",
            sampling_seed=2,
        )
        assert runner.experiment_key.variant == "ablation_nolima_s2"

    def test_sampling_seed_in_config_sidecar(self):
        runner = self._make_runner(decoding="greedy", heads=None, sampling_seed=3)
        # The config dict is built inside run(), but we can verify the kwarg is stored
        assert runner._ablation_kwargs.get("sampling_seed") == 3


# ---------------------------------------------------------------------------
# _extract_layer_head_counts (HF config layout shim)
# ---------------------------------------------------------------------------
class TestExtractLayerHeadCounts:
    def _make_config(self, **attrs):
        """Build a minimal mock that mimics a HuggingFace PretrainedConfig:
        attribute access for known keys, AttributeError for unknowns, and a
        to_dict() method used in the assertion message."""
        cfg = MagicMock(spec=[*attrs.keys(), "to_dict"])
        for k, v in attrs.items():
            setattr(cfg, k, v)
        cfg.to_dict.return_value = attrs
        return cfg

    def test_plain_causal_lm_top_level_attrs(self):
        """Llama / Qwen-style configs expose dims at the top level."""
        config = self._make_config(num_hidden_layers=32, num_attention_heads=32)
        assert _extract_layer_head_counts(config, "Llama-3-8B") == (32, 32)

    def test_gemma3_nested_under_text_config(self):
        """Gemma3ForConditionalGeneration nests text-model dims under text_config."""
        text_cfg = self._make_config(num_hidden_layers=48, num_attention_heads=16)
        # Gemma3Config: top-level lacks num_hidden_layers; lives in text_config
        config = MagicMock(spec=["text_config", "to_dict"])
        config.text_config = text_cfg
        config.to_dict.return_value = {"text_config": {}}
        assert _extract_layer_head_counts(config, "gemma-3-12b-it") == (48, 16)

    def test_text_config_takes_priority_over_top_level(self):
        """When both exist, text_config (text-model dims) wins over top-level
        (which on multimodal configs would be the wrapper config's dims)."""
        text_cfg = self._make_config(num_hidden_layers=48, num_attention_heads=16)
        config = self._make_config(num_hidden_layers=999, num_attention_heads=999)
        config.text_config = text_cfg
        assert _extract_layer_head_counts(config, "wrapped") == (48, 16)

    def test_missing_dims_raises_with_helpful_message(self):
        config = self._make_config(some_other_field=1)
        with pytest.raises(AssertionError, match="Could not determine"):
            _extract_layer_head_counts(config, "weird-model")
