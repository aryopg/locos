"""Tests for locos.utils.needle_utils.build_tracked_prompt.

The helper is the shared replacement for decode/re-encode round-trips in
behavioral / contrastive / logit_contrib / cri detectors. It must track the
needle position exactly through every composition step, regardless of
tokenizer quirks.
"""

import re

import pytest
import torch

from locos.utils.needle_utils import (
    build_period_token_positions,
    build_tracked_prompt,
)


class _BPEishTokenizer:
    """A tokenizer that mimics BPE leading-space boundary effects.

    - Words with a leading space tokenize to a different ID than the same
      word in isolation (e.g. " There" vs "There").
    - Whitespace runs tokenize as a single space token.
    - Punctuation is its own token.
    - No chat template.
    """

    bos_token_id = 0
    chat_template = None

    def __init__(self):
        self._vocab: dict[str, int] = {"<bos>": 0}
        self._inv: dict[int, str] = {0: "<bos>"}
        self._next = 1

    def _tok(self, piece: str) -> int:
        if piece not in self._vocab:
            self._vocab[piece] = self._next
            self._inv[self._next] = piece
            self._next += 1
        return self._vocab[piece]

    def encode(self, text, add_special_tokens=False):
        pieces = re.findall(r" ?[A-Za-z0-9]+|[^A-Za-z0-9\s]| +", text)
        ids = [self._tok(p) for p in pieces]
        if add_special_tokens:
            ids = [self.bos_token_id, *ids]
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(self._inv.get(t, "?") for t in token_ids)


class _ChatTokenizer(_BPEishTokenizer):
    """BPE-ish tokenizer with a simple chat template."""

    chat_template = "<|im_start|>{% for m in messages %}{{m.content}}<|im_end|>{% endfor %}<|assistant|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert not tokenize, "test only exercises tokenize=False"
        body = "".join(m["content"] for m in messages)
        suffix = "<|assistant|>" if add_generation_prompt else ""
        return f"<|im_start|>{body}<|im_end|>{suffix}"


@pytest.fixture
def haystack_tokens_and_periods():
    tok = _BPEishTokenizer()
    haystack = ("Some filler sentence about nothing. " * 80).strip()
    haystack_tokens = tok.encode(haystack)
    period_id = tok.encode(".")[0]
    period_positions = build_period_token_positions(haystack_tokens, {period_id})
    return tok, haystack_tokens, period_positions


def test_tracked_prompt_niah_path(haystack_tokens_and_periods):
    tok, haystack_tokens, period_positions = haystack_tokens_and_periods
    needle = "There was a vegan guest, named Gary."
    needle_tokens = tok.encode(needle)
    question_tokens = tok.encode("Who is the vegan guest?")

    input_ids, ns, ne, a_s, a_e = build_tracked_prompt(
        haystack_tokens=haystack_tokens,
        needle_tokens=needle_tokens,
        context_length=600,
        depth_percent=25,
        period_positions=period_positions,
        tokenizer=tok,
        prompt_template=None,
        question_tokens=question_tokens,
    )
    assert a_s is None and a_e is None
    # Tracked position contains the needle verbatim.
    assert input_ids[0, ns:ne].tolist() == needle_tokens


def test_tracked_prompt_template_path(haystack_tokens_and_periods):
    """NoLiMa-style template with {haystack} still tracks needle position."""
    tok, haystack_tokens, period_positions = haystack_tokens_and_periods
    needle = "There was a vegan guest, named Gary."
    needle_tokens = tok.encode(needle)
    template = "Instruction: find the guest.\n\n{haystack}\n\nAnswer:"

    input_ids, ns, ne, _, _ = build_tracked_prompt(
        haystack_tokens=haystack_tokens,
        needle_tokens=needle_tokens,
        context_length=600,
        depth_percent=50,
        period_positions=period_positions,
        tokenizer=tok,
        prompt_template=template,
    )
    assert input_ids[0, ns:ne].tolist() == needle_tokens


def test_tracked_prompt_with_bos(haystack_tokens_and_periods):
    tok, haystack_tokens, period_positions = haystack_tokens_and_periods
    needle_tokens = tok.encode("Gary is vegan.")
    question_tokens = tok.encode("Q?")

    input_ids, ns, ne, _, _ = build_tracked_prompt(
        haystack_tokens=haystack_tokens,
        needle_tokens=needle_tokens,
        context_length=500,
        depth_percent=50,
        period_positions=period_positions,
        tokenizer=tok,
        question_tokens=question_tokens,
        add_bos=True,
    )
    assert input_ids[0, 0].item() == tok.bos_token_id
    assert input_ids[0, ns:ne].tolist() == needle_tokens


def test_tracked_prompt_with_chat_template():
    tok = _ChatTokenizer()
    haystack = ("Some filler sentence about nothing. " * 80).strip()
    haystack_tokens = tok.encode(haystack)
    period_id = tok.encode(".")[0]
    period_positions = build_period_token_positions(haystack_tokens, {period_id})

    needle_tokens = tok.encode("Gary is vegan.")
    question_tokens = tok.encode("Q?")

    input_ids, ns, ne, _, _ = build_tracked_prompt(
        haystack_tokens=haystack_tokens,
        needle_tokens=needle_tokens,
        context_length=500,
        depth_percent=50,
        period_positions=period_positions,
        tokenizer=tok,
        question_tokens=question_tokens,
        use_chat_template=True,
    )
    # Chat-template prefix/suffix tokens surround the content; needle still
    # tracked verbatim at the reported position.
    assert input_ids[0, ns:ne].tolist() == needle_tokens
    # add_bos is ignored when use_chat_template=True.


def test_tracked_prompt_with_answer_tokens(haystack_tokens_and_periods):
    """Answer span is returned for teacher-forced setups (CRI)."""
    tok, haystack_tokens, period_positions = haystack_tokens_and_periods
    needle_tokens = tok.encode("Gary is vegan.")
    question_tokens = tok.encode("Q?")
    answer_tokens = tok.encode("Gary")

    input_ids, ns, ne, a_s, a_e = build_tracked_prompt(
        haystack_tokens=haystack_tokens,
        needle_tokens=needle_tokens,
        context_length=500,
        depth_percent=50,
        period_positions=period_positions,
        tokenizer=tok,
        question_tokens=question_tokens,
        answer_tokens=answer_tokens,
    )
    assert a_s is not None and a_e is not None
    assert input_ids[0, a_s:a_e].tolist() == answer_tokens
    assert input_ids[0, ns:ne].tolist() == needle_tokens
    # Answer must not overlap needle span.
    assert a_s >= ne or a_e <= ns


def test_tracked_prompt_with_prompt_suffix(haystack_tokens_and_periods):
    tok, haystack_tokens, period_positions = haystack_tokens_and_periods
    needle_tokens = tok.encode("Gary.")
    question_tokens = tok.encode("Q?")
    suffix_ids = [42, 43, 44]

    input_ids, ns, ne, _, _ = build_tracked_prompt(
        haystack_tokens=haystack_tokens,
        needle_tokens=needle_tokens,
        context_length=500,
        depth_percent=50,
        period_positions=period_positions,
        tokenizer=tok,
        question_tokens=question_tokens,
        prompt_suffix_ids=suffix_ids,
    )
    assert input_ids[0, -3:].tolist() == suffix_ids
    assert input_ids[0, ns:ne].tolist() == needle_tokens


def test_tracked_prompt_asserts_missing_question():
    tok = _BPEishTokenizer()
    with pytest.raises(AssertionError, match="question_tokens required"):
        build_tracked_prompt(
            haystack_tokens=tok.encode("some text"),
            needle_tokens=tok.encode("needle"),
            context_length=100,
            depth_percent=50,
            period_positions=None,
            tokenizer=tok,
            prompt_template=None,
            question_tokens=None,
        )


def test_tracked_prompt_needle_at_depth_0(haystack_tokens_and_periods):
    """depth_percent=0 keeps needle at the start; position still tracked."""
    tok, haystack_tokens, period_positions = haystack_tokens_and_periods
    needle_tokens = tok.encode("Gary.")
    question_tokens = tok.encode("Q?")

    input_ids, ns, ne, _, _ = build_tracked_prompt(
        haystack_tokens=haystack_tokens,
        needle_tokens=needle_tokens,
        context_length=500,
        depth_percent=0,
        period_positions=period_positions,
        tokenizer=tok,
        question_tokens=question_tokens,
    )
    assert input_ids[0, ns:ne].tolist() == needle_tokens


def test_tracked_prompt_needle_at_depth_100(haystack_tokens_and_periods):
    """depth_percent=100 places needle at end of context; position tracked."""
    tok, haystack_tokens, period_positions = haystack_tokens_and_periods
    needle_tokens = tok.encode("Gary.")
    question_tokens = tok.encode("Q?")

    input_ids, ns, ne, _, _ = build_tracked_prompt(
        haystack_tokens=haystack_tokens,
        needle_tokens=needle_tokens,
        context_length=500,
        depth_percent=100,
        period_positions=period_positions,
        tokenizer=tok,
        question_tokens=question_tokens,
    )
    assert input_ids[0, ns:ne].tolist() == needle_tokens


def test_tracked_prompt_full_stack():
    """Combine template + chat template + BOS-free (BOS ignored with chat) + suffix + answer."""
    tok = _ChatTokenizer()
    haystack = ("Some filler sentence about nothing. " * 80).strip()
    haystack_tokens = tok.encode(haystack)
    period_id = tok.encode(".")[0]
    period_positions = build_period_token_positions(haystack_tokens, {period_id})

    template = "Instruction.\n\n{haystack}\n\nAnswer:"
    needle_tokens = tok.encode("Gary is vegan.")
    answer_tokens = tok.encode("Gary")
    suffix_ids = [91]

    input_ids, ns, ne, a_s, a_e = build_tracked_prompt(
        haystack_tokens=haystack_tokens,
        needle_tokens=needle_tokens,
        context_length=500,
        depth_percent=50,
        period_positions=period_positions,
        tokenizer=tok,
        prompt_template=template,
        use_chat_template=True,
        prompt_suffix_ids=suffix_ids,
        answer_tokens=answer_tokens,
    )
    assert input_ids[0, ns:ne].tolist() == needle_tokens
    assert input_ids[0, a_s:a_e].tolist() == answer_tokens
    # Suffix lands between chat end and answer.
    assert 91 in input_ids[0].tolist()


def test_tracked_prompt_single_tensor_batch_dim(haystack_tokens_and_periods):
    tok, haystack_tokens, period_positions = haystack_tokens_and_periods
    input_ids, _, _, _, _ = build_tracked_prompt(
        haystack_tokens=haystack_tokens,
        needle_tokens=tok.encode("Gary."),
        context_length=500,
        depth_percent=50,
        period_positions=period_positions,
        tokenizer=tok,
        question_tokens=tok.encode("Q?"),
    )
    assert isinstance(input_ids, torch.Tensor)
    assert input_ids.dim() == 2
    assert input_ids.shape[0] == 1
