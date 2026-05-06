"""Tests for eval task implementations (no GPU needed).

Tests scoring logic, prompt formatting, task metadata, and the
MedRAG/ACI-Bench helper functions directly.
"""

from __future__ import annotations

import pytest

from locos_eval.evals.runner import EvalSample

# ---------------------------------------------------------------------------
# NQ-Swap
# ---------------------------------------------------------------------------


class TestNQSwapScore:
    """Test NQSwapEval.score() logic."""

    def _make_task(self):
        from locos_eval.evals.tasks.nq_swap_task import NQSwapEval

        return NQSwapEval(model="test/model", heads="h.json", hf_repo="fake")

    def test_task_name(self):
        task = self._make_task()
        assert task.task_name() == "nq_swap"

    def test_system_message_is_none(self):
        task = self._make_task()
        assert task.system_message() is None

    def test_score_sub_match(self):
        """Output matching the substituted answer → sub_em=1, org_em=0."""
        task = self._make_task()
        sample = EvalSample(
            prompt="ignored",
            target="Paris",
            metadata={"sub_answer": "Paris", "org_answer": "London"},
        )
        scores = task.score("The answer is Paris.", sample)
        assert scores["sub_em"] == 1.0
        assert scores["org_em"] == 0.0

    def test_score_org_match(self):
        """Output matching the original answer → sub_em=0, org_em=1."""
        task = self._make_task()
        sample = EvalSample(
            prompt="ignored",
            target="Paris",
            metadata={"sub_answer": "Paris", "org_answer": "London"},
        )
        scores = task.score("I think it's London.", sample)
        assert scores["sub_em"] == 0.0
        assert scores["org_em"] == 1.0

    def test_score_neither_match(self):
        """Output matching neither answer → both 0."""
        task = self._make_task()
        sample = EvalSample(
            prompt="ignored",
            target="Paris",
            metadata={"sub_answer": "Paris", "org_answer": "London"},
        )
        scores = task.score("I don't know.", sample)
        assert scores["sub_em"] == 0.0
        assert scores["org_em"] == 0.0

    def test_score_both_match(self):
        """Output containing both answers → both 1."""
        task = self._make_task()
        sample = EvalSample(
            prompt="ignored",
            target="Paris",
            metadata={"sub_answer": "Paris", "org_answer": "London"},
        )
        scores = task.score("Paris and London are both cities.", sample)
        assert scores["sub_em"] == 1.0
        assert scores["org_em"] == 1.0


# ---------------------------------------------------------------------------
# MedRAG
# ---------------------------------------------------------------------------


class TestMedRAGScore:
    """Test MedRAGEval.score() and prompt formatting."""

    def _make_task(self, **kwargs):
        from locos_eval.evals.tasks.medrag_task import MedRAGEval

        defaults = dict(model="test/model", heads="h.json", hf_repo="fake")
        defaults.update(kwargs)
        return MedRAGEval(**defaults)

    def test_task_name_default(self):
        task = self._make_task()
        assert task.task_name() == "medrag_top5"

    def test_task_name_with_dataset(self):
        task = self._make_task(dataset_name="medqa")
        assert task.task_name() == "medrag_medqa_top5"

    def test_task_name_with_topk(self):
        task = self._make_task(dataset_name="medqa", top_k=10)
        assert task.task_name() == "medrag_medqa_top10"

    def test_system_message_loaded_from_yaml(self):
        task = self._make_task()
        msg = task.system_message()
        assert msg is not None
        assert "medical" in msg.lower()

    def test_score_correct_answer_tag(self):
        """Correct answer letter in <answer> tag → accuracy=1."""
        task = self._make_task()
        sample = EvalSample(prompt="ignored", target="B")
        scores = task.score("The answer is B because... <answer>B</answer>", sample)
        assert scores["accuracy"] == 1.0

    def test_score_wrong_answer(self):
        """Wrong answer letter → accuracy=0."""
        task = self._make_task()
        sample = EvalSample(prompt="ignored", target="B")
        scores = task.score("<answer>A</answer>", sample)
        assert scores["accuracy"] == 0.0

    def test_score_lowercase_target(self):
        """Target letter should be uppercased before comparison."""
        task = self._make_task()
        sample = EvalSample(prompt="ignored", target="b")
        scores = task.score("<answer>B</answer>", sample)
        assert scores["accuracy"] == 1.0

    def test_score_no_answer_extracted(self):
        """No recognizable answer → accuracy=0."""
        task = self._make_task()
        sample = EvalSample(prompt="ignored", target="C")
        scores = task.score("I'm not sure about the answer.", sample)
        assert scores["accuracy"] == 0.0

    def test_top_k_assertion(self):
        """top_k must be positive."""
        with pytest.raises(AssertionError, match="top_k must be positive"):
            self._make_task(top_k=0)


class TestMedRAGFormatPrompt:
    """Test the MedRAG prompt formatting function."""

    def test_format_prompt_with_passages(self):
        from locos_eval.evals.tasks.medrag_task import _format_prompt, _load_prompts

        prompts = _load_prompts()
        result = _format_prompt(
            question="What causes flu?",
            options={"A": "Virus", "B": "Bacteria", "C": "Fungus"},
            retrieved_passages=[
                {"content": "Influenza is caused by viruses."},
                {"content": "Bacteria cause different infections."},
            ],
            top_k=2,
            prompts=prompts,
        )
        assert "What causes flu?" in result
        assert "A. Virus" in result
        assert "B. Bacteria" in result
        assert "C. Fungus" in result
        assert "Influenza is caused by viruses." in result

    def test_format_prompt_respects_top_k(self):
        """Only top_k passages should be included."""
        from locos_eval.evals.tasks.medrag_task import _format_prompt, _load_prompts

        prompts = _load_prompts()
        result = _format_prompt(
            question="Q?",
            options={"A": "a"},
            retrieved_passages=[
                {"content": "passage 1"},
                {"content": "passage 2"},
                {"content": "passage 3"},
            ],
            top_k=1,
            prompts=prompts,
        )
        assert "passage 1" in result
        assert "passage 2" not in result


# ---------------------------------------------------------------------------
# XSum
# ---------------------------------------------------------------------------


class TestXSumScore:
    """Test XSumEval metadata and scoring dependencies."""

    def _make_task(self):
        from locos_eval.evals.tasks.xsum_task import XSumEval

        return XSumEval(model="test/model", heads="h.json", hf_repo="fake")

    def test_task_name(self):
        assert self._make_task().task_name() == "xsum_faithfulness"

    def test_system_message(self):
        msg = self._make_task().system_message()
        assert msg is not None
        assert "summarize" in msg.lower()


# ---------------------------------------------------------------------------
# ACI-Bench
# ---------------------------------------------------------------------------


class TestACIBench:
    """Test ACI-Bench prompt formatting, scoring logic, and judge parsing."""

    def _make_task(self, **kwargs):
        from locos_eval.evals.tasks.aci_bench_task import ACIBenchEval

        defaults = dict(
            model="test/model",
            heads="h.json",
            hf_repo="fake",
            judge_model="none",
        )
        defaults.update(kwargs)
        return ACIBenchEval(**defaults)

    def test_task_name(self):
        assert self._make_task().task_name() == "aci_bench"

    def test_system_message_from_yaml(self):
        msg = self._make_task().system_message()
        assert msg is not None
        assert "clinical" in msg.lower()

    def test_score_without_judge(self):
        """With judge_model='none', judge scores should be -1 and text scores present."""
        import types
        from unittest.mock import MagicMock, patch

        import torch

        task = self._make_task(judge_model="none")
        sample = EvalSample(
            prompt="ignored",
            target="Reference note.",
            metadata={"dialogue": "D: Hello P: Hi"},
        )

        # Mock bert_score.score which is lazily imported inside bertscore_f1
        mock_bert_score_fn = MagicMock(return_value=(torch.tensor([0.8]), torch.tensor([0.8]), torch.tensor([0.85])))
        mock_bert_score_module = types.ModuleType("bert_score")
        mock_bert_score_module.score = mock_bert_score_fn

        with patch.dict("sys.modules", {"bert_score": mock_bert_score_module}):
            scores = task.score("Generated note.", sample)

        assert "rouge_l" in scores
        assert isinstance(scores["rouge_l"], float)
        assert "bertscore" in scores
        assert isinstance(scores["bertscore"], float)
        assert scores["judge_completeness"] == -1.0
        assert scores["judge_accuracy"] == -1.0
        assert scores["judge_relevance"] == -1.0
        assert scores["judge_normalized"] == -1.0

    def test_n_shot_validation(self):
        """n_shot must be non-negative."""
        with pytest.raises(AssertionError, match="n_shot must be non-negative"):
            self._make_task(n_shot=-1)


class TestACIBenchFormatPrompt:
    """Test ACI-Bench prompt formatting helper."""

    def test_zero_shot(self):
        from locos_eval.evals.tasks.aci_bench_task import _format_prompt, _load_prompts

        prompts = _load_prompts()
        result = _format_prompt(
            test_dialogue="Doctor: How are you?\nPatient: Not great.",
            train_examples=[],
            prompts=prompts,
        )
        assert "How are you?" in result
        assert "Not great" in result
        # No example prefix for zero-shot
        assert "Example 1" not in result

    def test_few_shot(self):
        from locos_eval.evals.tasks.aci_bench_task import _format_prompt, _load_prompts

        prompts = _load_prompts()
        train_examples = [
            {"inputs": "D: Symptoms?\nP: Headache.", "target": "Assessment: headache."},
        ]
        result = _format_prompt(
            test_dialogue="D: How are you?\nP: Fine.",
            train_examples=train_examples,
            prompts=prompts,
        )
        assert "Example 1" in result
        assert "Headache" in result
        assert "How are you?" in result


# ---------------------------------------------------------------------------
# LongBench-v2
# ---------------------------------------------------------------------------


class TestLongBenchV2Score:
    """Test LongBenchV2Eval.score() and task metadata."""

    def _make_task(self, **kwargs):
        from locos_eval.evals.tasks.longbench_v2_task import LongBenchV2Eval

        defaults = dict(model="test/model", heads="h.json", hf_repo="fake")
        defaults.update(kwargs)
        return LongBenchV2Eval(**defaults)

    def test_task_name_default(self):
        task = self._make_task()
        assert task.task_name() == "longbench_v2_short"

    def test_task_name_medium(self):
        task = self._make_task(length="medium")
        assert task.task_name() == "longbench_v2_medium"

    def test_task_name_all(self):
        task = self._make_task(length="all")
        assert task.task_name() == "longbench_v2"

    def test_system_message_loaded_from_yaml(self):
        task = self._make_task()
        msg = task.system_message()
        assert msg is not None
        assert "<answer>" in msg
        assert "step by step" in msg.lower()

    def test_score_correct_answer_tag(self):
        """Correct answer in <answer> tag → accuracy=1, compensated=1."""
        task = self._make_task()
        sample = EvalSample(prompt="ignored", target="C")
        scores = task.score("The text says... <answer>C</answer>", sample)
        assert scores["accuracy"] == 1.0
        assert scores["accuracy_compensated"] == 1.0

    def test_score_wrong_answer(self):
        """Wrong answer letter → accuracy=0, compensated=0."""
        task = self._make_task()
        sample = EvalSample(prompt="ignored", target="C")
        scores = task.score("<answer>A</answer>", sample)
        assert scores["accuracy"] == 0.0
        assert scores["accuracy_compensated"] == 0.0

    def test_score_no_answer_extracted(self):
        """No parseable answer → accuracy=0, compensated=0.25 (random chance)."""
        task = self._make_task()
        sample = EvalSample(prompt="ignored", target="B")
        # Use text with no standalone A-J letters to trigger None extraction
        scores = task.score("not sure what to say 123", sample)
        assert scores["accuracy"] == 0.0
        assert scores["accuracy_compensated"] == 0.25

    def test_score_lowercase_target(self):
        """Target letter should be uppercased before comparison."""
        task = self._make_task()
        sample = EvalSample(prompt="ignored", target="d")
        scores = task.score("<answer>D</answer>", sample)
        assert scores["accuracy"] == 1.0

    def test_invalid_length_rejected(self):
        """Invalid length filter raises AssertionError."""
        with pytest.raises(AssertionError, match="length must be one of"):
            self._make_task(length="tiny")


class TestLongBenchV2FormatPrompt:
    """Test the LongBench-v2 prompt formatting function."""

    def test_format_prompt_includes_all_fields(self):
        from locos_eval.evals.tasks.longbench_v2_task import (
            _format_prompt,
            _load_prompts,
        )

        prompts = _load_prompts()
        result = _format_prompt(
            context="This is a long document about climate change.",
            question="What is the document about?",
            choice_A="Climate change",
            choice_B="Cooking recipes",
            choice_C="Space exploration",
            choice_D="Ancient history",
            prompts=prompts,
        )
        assert "climate change" in result.lower()
        assert "(A) Climate change" in result
        assert "(B) Cooking recipes" in result
        assert "(C) Space exploration" in result
        assert "(D) Ancient history" in result
        assert "What is the document about?" in result
        assert "<text>" in result
        assert "</text>" in result


class TestLongBenchV2MeasureOverhead:
    """Test exact per-sample overhead measurement."""

    def test_overhead_includes_question_and_choices(self):
        from unittest.mock import MagicMock

        from locos_eval.evals.tasks.longbench_v2_task import (
            _load_prompts,
            _measure_non_context_tokens,
        )

        prompts = _load_prompts()
        tokenizer = MagicMock()
        # Simulate: apply_chat_template returns a string, encode counts tokens
        tokenizer.apply_chat_template.return_value = "formatted prompt with question and choices"
        tokenizer.encode.return_value = list(range(42))

        row = {
            "question": "What is X?",
            "choice_A": "A1",
            "choice_B": "B1",
            "choice_C": "C1",
            "choice_D": "D1",
        }
        result = _measure_non_context_tokens(row, prompts, "system msg", tokenizer)
        assert result == 42
        # Should have called apply_chat_template with system + user messages
        call_args = tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # Context should be empty in the measurement prompt
        assert "What is X?" in messages[1]["content"]

    def test_overhead_varies_with_question_length(self):
        """Longer questions should produce higher overhead."""
        from unittest.mock import MagicMock

        from locos_eval.evals.tasks.longbench_v2_task import (
            _load_prompts,
            _measure_non_context_tokens,
        )

        prompts = _load_prompts()

        # Tokenizer that returns len(text) // 4 as token count (rough char→token)
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = lambda msgs, **kw: msgs[-1]["content"]
        tokenizer.encode.side_effect = lambda text, **kw: list(range(len(text) // 4))

        short_row = {
            "question": "Why?",
            "choice_A": "A",
            "choice_B": "B",
            "choice_C": "C",
            "choice_D": "D",
        }
        long_row = {
            "question": "What is the primary reason for the observed phenomenon in the third paragraph?",
            "choice_A": "Thermodynamic equilibrium",
            "choice_B": "Quantum entanglement effects",
            "choice_C": "Gravitational wave interference",
            "choice_D": "Electromagnetic field fluctuations",
        }

        short_overhead = _measure_non_context_tokens(short_row, prompts, None, tokenizer)
        long_overhead = _measure_non_context_tokens(long_row, prompts, None, tokenizer)
        assert long_overhead > short_overhead


class TestLongBenchV2TruncateContext:
    """Test the context truncation helper."""

    def test_no_truncation_when_short(self):
        from unittest.mock import MagicMock

        from locos_eval.evals.tasks.longbench_v2_task import _truncate_context

        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(100))
        tokenizer.decode.return_value = "decoded"

        result, was_truncated = _truncate_context("short text", tokenizer, 200)
        assert result == "short text"
        assert was_truncated is False
        tokenizer.decode.assert_not_called()

    def test_truncation_first_half_last_half(self):
        from unittest.mock import MagicMock

        from locos_eval.evals.tasks.longbench_v2_task import _truncate_context

        tokenizer = MagicMock()
        # 1000 tokens, budget of 100
        tokenizer.encode.return_value = list(range(1000))
        tokenizer.decode.return_value = "truncated text"

        result, was_truncated = _truncate_context("long text", tokenizer, 100)
        assert result == "truncated text"
        assert was_truncated is True

        # Should decode first 50 + last 50 tokens
        decode_call_args = tokenizer.decode.call_args[0][0]
        assert decode_call_args == list(range(50)) + list(range(950, 1000))

    def test_truncation_odd_budget_uses_all_tokens(self):
        """Odd max_ctx_tokens should not waste a token."""
        from unittest.mock import MagicMock

        from locos_eval.evals.tasks.longbench_v2_task import _truncate_context

        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(1000))
        tokenizer.decode.return_value = "truncated"

        _truncate_context("long text", tokenizer, 101)
        decode_call_args = tokenizer.decode.call_args[0][0]
        # first_half=50, second_half=51 → 101 tokens total
        assert len(decode_call_args) == 101
        assert decode_call_args == list(range(50)) + list(range(949, 1000))

    def test_truncation_exact_boundary(self):
        """Context exactly at limit → no truncation."""
        from unittest.mock import MagicMock

        from locos_eval.evals.tasks.longbench_v2_task import _truncate_context

        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(100))

        result, was_truncated = _truncate_context("exact text", tokenizer, 100)
        assert result == "exact text"
        assert was_truncated is False


class TestACIBenchJudgeParsing:
    """Test the _run_judge score parsing logic."""

    def _make_task_with_judge(self):
        from locos_eval.evals.tasks.aci_bench_task import ACIBenchEval

        return ACIBenchEval(
            model="test/model",
            heads="h.json",
            hf_repo="fake",
            judge_model="test-model",
        )

    def test_parse_dict_scores(self):
        """Judge returning {axis: {score: N, explanation: ...}} format."""
        from unittest.mock import patch

        task = self._make_task_with_judge()
        sample = EvalSample(
            prompt="p",
            target="t",
            metadata={"dialogue": "d"},
        )

        judge_response = {
            "completeness": {"score": 4, "explanation": "good"},
            "accuracy": {"score": 5, "explanation": "perfect"},
            "relevance": {"score": 3, "explanation": "ok"},
        }

        with patch("locos_eval.evals.scorers.call_llm_judge", return_value=judge_response):
            scores = task._run_judge("output", sample)

        assert scores["judge_completeness"] == 4.0
        assert scores["judge_accuracy"] == 5.0
        assert scores["judge_relevance"] == 3.0
        # Normalized: (mean(4,5,3) - 1) / 4 = (4 - 1) / 4 = 0.75
        assert scores["judge_normalized"] == pytest.approx(0.75, abs=0.01)

    def test_parse_bare_int_scores(self):
        """Judge returning {axis: N} format (bare integers)."""
        from unittest.mock import patch

        task = self._make_task_with_judge()
        sample = EvalSample(
            prompt="p",
            target="t",
            metadata={"dialogue": "d"},
        )

        judge_response = {
            "completeness": 5,
            "accuracy": 5,
            "relevance": 5,
        }

        with patch("locos_eval.evals.scorers.call_llm_judge", return_value=judge_response):
            scores = task._run_judge("output", sample)

        assert scores["judge_completeness"] == 5.0
        assert scores["judge_accuracy"] == 5.0
        assert scores["judge_relevance"] == 5.0
        # Normalized: (5 - 1) / 4 = 1.0
        assert scores["judge_normalized"] == pytest.approx(1.0, abs=0.01)

    def test_parse_failed_judge(self):
        """Judge returning empty dict → all scores -1."""
        from unittest.mock import patch

        task = self._make_task_with_judge()
        sample = EvalSample(
            prompt="p",
            target="t",
            metadata={"dialogue": "d"},
        )

        with patch("locos_eval.evals.scorers.call_llm_judge", return_value={}):
            scores = task._run_judge("output", sample)

        assert scores["judge_completeness"] == -1.0
        assert scores["judge_accuracy"] == -1.0
        assert scores["judge_relevance"] == -1.0
        assert scores["judge_normalized"] == -1.0

    def test_parse_out_of_range_score(self):
        """Scores outside [1,5] should default to -1."""
        from unittest.mock import patch

        task = self._make_task_with_judge()
        sample = EvalSample(
            prompt="p",
            target="t",
            metadata={"dialogue": "d"},
        )

        judge_response = {
            "completeness": {"score": 10, "explanation": "too high"},
            "accuracy": {"score": 0, "explanation": "too low"},
            "relevance": {"score": 3, "explanation": "valid"},
        }

        with patch("locos_eval.evals.scorers.call_llm_judge", return_value=judge_response):
            scores = task._run_judge("output", sample)

        assert scores["judge_completeness"] == -1.0
        assert scores["judge_accuracy"] == -1.0
        assert scores["judge_relevance"] == 3.0
        # Only relevance is valid: (3 - 1) / 4 = 0.5
        assert scores["judge_normalized"] == pytest.approx(0.5, abs=0.01)
