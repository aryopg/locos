"""Tests for standalone scoring functions (no GPU needed)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from locos_eval.evals.scorers import (
    call_llm_judge,
    extract_answer_letter,
    factkb_score,
    factkb_score_batch,
    normalize_answer,
    rouge_l_score,
    subspan_match,
)

# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------


class TestRougeL:
    """Test rouge_l_score function."""

    def test_identical_strings(self) -> None:
        score = rouge_l_score("The cat sat on the mat.", "The cat sat on the mat.")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_empty_prediction(self) -> None:
        score = rouge_l_score("", "The cat sat on the mat.")
        assert score == 0.0

    def test_partial_match(self) -> None:
        score = rouge_l_score("The cat sat.", "The cat sat on the mat.")
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# extract_answer_letter
# ---------------------------------------------------------------------------


class TestExtractAnswerLetter:
    """Test answer letter extraction with priority ordering."""

    def test_answer_tag(self) -> None:
        assert extract_answer_letter("<answer>B</answer>") == "B"

    def test_answer_is_pattern(self) -> None:
        assert extract_answer_letter("The answer is C") == "C"

    def test_option_pattern(self) -> None:
        assert extract_answer_letter("option A") == "A"

    def test_fallback_first_letter(self) -> None:
        assert extract_answer_letter("A is correct") == "A"

    def test_no_letter(self) -> None:
        assert extract_answer_letter("no answer 123") is None

    def test_lowercase(self) -> None:
        assert extract_answer_letter("<answer>b</answer>") == "B"


# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------


class TestNormalizeAnswer:
    """Test answer normalisation."""

    def test_removes_articles(self) -> None:
        assert normalize_answer("The quick fox") == "quick fox"

    def test_removes_punctuation(self) -> None:
        assert normalize_answer("hello, world!") == "hello world"

    def test_collapses_whitespace(self) -> None:
        # "  a  b  " → lowercase "  a  b  " → remove article "a" → "     b  " → collapse → "b"
        assert normalize_answer("  a  b  ") == "b"


# ---------------------------------------------------------------------------
# subspan_match
# ---------------------------------------------------------------------------


class TestSubspanMatch:
    """Test subspan matching."""

    def test_match(self) -> None:
        assert subspan_match("Paris is great", "Paris") is True

    def test_no_match(self) -> None:
        assert subspan_match("London is great", "Paris") is False

    def test_empty_reference(self) -> None:
        assert subspan_match("Paris is great", "") is False

    def test_case_insensitive(self) -> None:
        assert subspan_match("PARIS is great", "paris") is True


# ---------------------------------------------------------------------------
# call_llm_judge
# ---------------------------------------------------------------------------


class TestCallLlmJudge:
    """Test LLM judge with mocked Anthropic client."""

    def _mock_response(self, text: str) -> MagicMock:
        """Build a mock Anthropic response containing *text*."""
        block = MagicMock()
        block.text = text
        response = MagicMock()
        response.content = [block]
        return response

    @patch("anthropic.Anthropic")
    def test_parses_json_response(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._mock_response('{"score": 0.8, "reason": "good"}')

        result = call_llm_judge("sys", "user")
        assert result == {"score": 0.8, "reason": "good"}
        mock_client.messages.create.assert_called_once()

    @patch("anthropic.Anthropic")
    def test_retries_on_failure(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            RuntimeError("transient"),
            self._mock_response('{"score": 1.0}'),
        ]

        result = call_llm_judge("sys", "user", max_retries=3)
        assert result == {"score": 1.0}
        assert mock_client.messages.create.call_count == 2

    @patch("time.sleep")
    @patch("anthropic.Anthropic")
    def test_returns_empty_on_all_failures(self, mock_cls: MagicMock, mock_sleep: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.side_effect = RuntimeError("always fails")

        result = call_llm_judge("sys", "user", max_retries=3)
        assert result == {}
        assert mock_client.messages.create.call_count == 3


# ---------------------------------------------------------------------------
# FactKB
# ---------------------------------------------------------------------------


class TestFactKB:
    """Test FactKB scorer with mocked transformers model."""

    def _setup_mocks(self, mock_model_cls, mock_tok_cls, score_value: float = 0.75):
        """Set up mock tokenizer and model returning the given score."""
        import torch

        mock_tokenizer = MagicMock()
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        # Tokenizer returns a dict-like that supports .to() and **unpacking
        mock_tokens = MagicMock()
        mock_tokens.to.return_value = mock_tokens
        mock_tokenizer.return_value = mock_tokens

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        # logits that produce the desired softmax score for class 1
        # softmax([0, x]) where sigmoid(x) ≈ score_value → x = log(p/(1-p))
        import math

        logit = math.log(score_value / (1 - score_value))
        logits = torch.tensor([[0.0, logit]])
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.return_value = mock_output

        return mock_tokenizer, mock_model

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForSequenceClassification")
    def test_returns_float_score(self, mock_model_cls, mock_tok_cls) -> None:
        self._setup_mocks(mock_model_cls, mock_tok_cls, score_value=0.75)
        score = factkb_score("summary text", "source article text")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(0.75, abs=0.05)

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForSequenceClassification")
    def test_passes_prediction_source_pair(self, mock_model_cls, mock_tok_cls) -> None:
        mock_tokenizer, _ = self._setup_mocks(mock_model_cls, mock_tok_cls)
        factkb_score("the summary", "the article")
        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args
        assert call_args[0][0] == [["the summary", "the article"]]

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForSequenceClassification")
    def test_batch_returns_list(self, mock_model_cls, mock_tok_cls) -> None:
        import torch

        mock_tokenizer = MagicMock()
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        mock_tokens = MagicMock()
        mock_tokens.to.return_value = mock_tokens
        mock_tokenizer.return_value = mock_tokens

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        logits = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 0.5]])
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.return_value = mock_output

        scores = factkb_score_batch(
            ["s1", "s2", "s3"],
            ["a1", "a2", "a3"],
        )
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        assert all(0.0 <= s <= 1.0 for s in scores)
        # Verify actual values against known softmax class-1 probabilities:
        # softmax([0, x])[1] = e^x / (1 + e^x)
        import math

        expected = [
            math.exp(1) / (1 + math.exp(1)),  # logits [0, 1] → ~0.731
            math.exp(2) / (1 + math.exp(2)),  # logits [0, 2] → ~0.881
            math.exp(0.5) / (1 + math.exp(0.5)),  # logits [0, 0.5] → ~0.622
        ]
        for score, exp in zip(scores, expected):
            assert score == pytest.approx(exp, abs=0.01), f"Expected {exp:.3f}, got {score:.3f}"
        # Scores should be ordered: s2 > s1 > s3 (matching logit magnitudes)
        assert scores[1] > scores[0] > scores[2]
