"""Unit tests for the first-token logit-difference CRI metric (cri.py).

No GPU required. Covers:
- scalar-output invariant for the logit-difference metric
- counterfactual-token choice (argmax-excluding-gold) on the baseline
- end-to-end patched-minus-corrupt delta behaves as expected through
  the ``(patched - corrupt).mean().item()`` chain in the CRI loop
- metric dispatch accepts / rejects the right names + requires the
  counterfactual id for the logit-difference path
"""

from __future__ import annotations

import pytest
import torch

from locos.detectors.cri import (
    _extract_metric,
    choose_counterfactual_token_id,
    extract_first_token_logit_diff,
)


def _mini_logits(batch: int = 1, seq: int = 6, vocab: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    logits = torch.arange(batch * seq * vocab, dtype=torch.float32).view(batch, seq, vocab)
    input_ids = torch.tensor([[2, 3, 5, 7, 1, 9]])
    return logits, input_ids


def test_choose_counterfactual_excludes_gold_and_picks_argmax():
    # Logits at position 2 (which predicts token at pos 3 = gold=7) are [20..29].
    # The largest logit overall is 29 at vocab index 9. Since gold=7, argmax
    # excluding gold should be vocab index 9.
    logits, input_ids = _mini_logits()
    counterfactual = choose_counterfactual_token_id(logits, input_ids, answer_start=3)
    assert int(counterfactual.item()) == 9


def test_choose_counterfactual_handles_gold_being_argmax():
    # Logits at position `answer_start - 1` predict the token at answer_start.
    # So for answer_start=2, we probe row logits[0, 1, :].
    logits = torch.zeros(1, 3, 5)
    logits[0, 1, 2] = 100.0  # gold gets 100
    logits[0, 1, 4] = 5.0  # runner-up
    logits[0, 1, 3] = 2.0
    input_ids = torch.tensor([[0, 0, 2]])  # answer_start=2 → gold = input_ids[0, 2] = 2
    counterfactual = choose_counterfactual_token_id(logits, input_ids, answer_start=2)
    # Gold is masked out, so the runner-up (vocab id 4, logit 5) wins.
    assert int(counterfactual.item()) == 4


def test_extract_first_token_logit_diff_returns_scalar_zero_dim():
    logits, input_ids = _mini_logits()
    # answer_start=3 → predict logits at position 2, gold = input_ids[0,3]=7.
    # Pick counterfactual = vocab id 9 (see above).
    # Expected = logits[0,2,7] - logits[0,2,9] = 27 - 29 = -2.
    got = extract_first_token_logit_diff(logits, input_ids, answer_start=3, counterfactual_token_id=9)
    assert got.ndim == 0, f"Expected 0-d tensor, got shape {got.shape}"
    assert got.item() == pytest.approx(-2.0)


def test_patched_minus_corrupt_isolates_head_effect_under_scale_shift():
    """The whole point of logit-diff: uniform logit shifts must cancel.

    Construct a 'patched' logits tensor that shifts every logit at the answer
    position up by +100. Under raw logit, CRI would read +100 for every head.
    Under logit-difference, CRI must read 0 because both gold and
    counterfactual shifted by the same amount.
    """
    logits_corrupt, input_ids = _mini_logits()
    logits_patched = logits_corrupt.clone()
    logits_patched[0, 2, :] += 100.0  # uniform shift at the relevant position
    counterfactual = choose_counterfactual_token_id(logits_corrupt, input_ids, answer_start=3)

    corrupt_metric = _extract_metric(
        "first_token_logit_diff",
        logits_corrupt,
        input_ids,
        answer_start=3,
        answer_end=5,
        counterfactual_token_id=counterfactual,
    )
    patched_metric = _extract_metric(
        "first_token_logit_diff",
        logits_patched,
        input_ids,
        answer_start=3,
        answer_end=5,
        counterfactual_token_id=counterfactual,
    )
    cri = (patched_metric - corrupt_metric).mean().item()
    assert cri == pytest.approx(0.0), "Uniform logit shift should vanish under logit-difference"


def test_patched_minus_corrupt_detects_selective_gold_boost():
    """If the patched run boosts only the gold logit, logit-diff increases."""
    logits_corrupt, input_ids = _mini_logits()
    logits_patched = logits_corrupt.clone()
    logits_patched[0, 2, 7] += 1.5  # gold is vocab id 7, boost just that
    counterfactual = choose_counterfactual_token_id(logits_corrupt, input_ids, answer_start=3)

    corrupt_metric = _extract_metric(
        "first_token_logit_diff",
        logits_corrupt,
        input_ids,
        3,
        5,
        counterfactual_token_id=counterfactual,
    )
    patched_metric = _extract_metric(
        "first_token_logit_diff",
        logits_patched,
        input_ids,
        3,
        5,
        counterfactual_token_id=counterfactual,
    )
    cri = (patched_metric - corrupt_metric).mean().item()
    assert cri == pytest.approx(1.5)


def test_extract_metric_dispatch_matches_scalar_vs_vector_shape():
    logits, input_ids = _mini_logits()
    scalar = _extract_metric("first_token_logit_diff", logits, input_ids, 3, 5, counterfactual_token_id=9)
    vector = _extract_metric("answer_logprob", logits, input_ids, 3, 5)
    assert scalar.ndim == 0
    # answer_logprob returns per-token logprobs for positions [answer_start, answer_end) = 2 tokens
    assert vector.shape == (2,)


def test_extract_metric_rejects_unknown_name():
    logits, input_ids = _mini_logits()
    with pytest.raises(ValueError, match="not_a_metric"):
        _extract_metric("not_a_metric", logits, input_ids, 3, 5)


def test_extract_metric_requires_counterfactual_for_logit_diff():
    logits, input_ids = _mini_logits()
    with pytest.raises(AssertionError, match="counterfactual_token_id"):
        _extract_metric("first_token_logit_diff", logits, input_ids, 3, 5)


def test_answer_start_must_be_at_least_one():
    logits, input_ids = _mini_logits()
    with pytest.raises(AssertionError):
        extract_first_token_logit_diff(logits, input_ids, answer_start=0, counterfactual_token_id=9)
