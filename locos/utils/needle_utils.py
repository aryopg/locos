"""Needle-in-a-haystack construction and search utilities.

Handles loading haystack/needle data, inserting needles at specified
depth positions (aligned to sentence boundaries), and finding needle
spans in tokenized prompts. Faithful to the original implementation
from Wu et al. / nightdessert/Retrieval_Head.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_needles(haystack_dir: Path) -> list[dict]:
    """Load needle/question/answer triples from needles.jsonl."""
    needles_path = haystack_dir / "needles.jsonl"
    assert needles_path.exists(), (
        f"needles.jsonl not found at {needles_path}. " f"Run locos/download_haystack_data.py first."
    )
    lines = [json.loads(line) for line in needles_path.read_text().strip().splitlines()]
    assert len(lines) >= 1, "needles.jsonl is empty"
    for line in lines:
        assert (
            "needle" in line and "question" in line and "real_needle" in line
        ), f"Each line must have 'needle', 'question', 'real_needle' keys, got: {list(line.keys())}"
    return lines


def load_haystack_texts(haystack_dir: Path, max_tokens: int) -> list[str]:
    """Load per-needle haystack texts, matching the original's part1/part2/part3.

    The original repo uses separate haystack directories per needle:
      - part1/: Cooper (Last of the Mohicans) for needle 0
      - part2/: Cooper (different chapters) for needle 1
      - part3/: Stoker (Dracula) for needle 2

    Each needle is tested against a different literary corpus. This matters
    because different haystack content produces different retrieval difficulty,
    directly affecting the number of ROUGE-passing trials and thus the mean
    retrieval scores.

    Returns a list of haystack texts, one per needle (indexed by needle_idx).
    """
    texts = []
    for part_idx in range(1, 4):
        plain_path = haystack_dir / f"part{part_idx}" / "p1.plain.txt"
        raw_path = haystack_dir / f"part{part_idx}" / "p1.txt"

        if plain_path.exists():
            text = plain_path.read_text()
        elif raw_path.exists():
            data = json.loads(raw_path.read_text())
            text = "\n\n".join(r["row"]["text"] for r in data["rows"] if r["row"].get("text"))
        else:
            raise AssertionError(
                f"Haystack part{part_idx} not found at {haystack_dir}. " f"Run locos/download_haystack_data.py first."
            )

        # Repeat text until long enough for max context length
        while len(text.split()) < max_tokens:
            text += "\n\n" + text
        texts.append(text)

    return texts


# ---------------------------------------------------------------------------
# Period tokens (sentence boundary alignment)
# ---------------------------------------------------------------------------


def get_period_tokens(
    model_name: str,
    tokenizer,
    use_hardcoded: bool = True,
    force_llama2: bool = False,
) -> set[int]:
    """Get period token IDs for sentence boundary alignment.

    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
        tokenizer: Tokenizer instance for fallback
        use_hardcoded: If True, use hardcoded tokens for known models; else always compute
        force_llama2: DEBUG: Force Llama-2 tokens even on other models
    """
    if force_llama2:
        return {29889, 869}

    if use_hardcoded:
        model_lower = model_name.lower()
        if "llama-2" in model_lower:
            return {29889, 869}
        elif "llama-3" in model_lower:
            return {13}
        elif "mistral" in model_lower or "mixtral" in model_lower:
            return {842, 28723}

    # Fallback: dynamically compute
    return set(tokenizer.encode(".", add_special_tokens=False))


def build_period_token_positions(token_ids: list[int], period_tokens: set[int]) -> list[int]:
    """Return token indices that end a sentence."""
    return [idx for idx, token_id in enumerate(token_ids) if token_id in period_tokens]


# ---------------------------------------------------------------------------
# Needle insertion
# ---------------------------------------------------------------------------


def insert_needle_tokens(
    context_tokens: list[int],
    needle_tokens: list[int],
    context_length: int,
    depth_percent: float,
    period_positions: list[int] | None = None,
    context_buffer: int = 200,
) -> tuple[list[int], int, int]:
    """Insert pre-tokenized needle into context, preserving original alignment logic."""
    effective_length = context_length - context_buffer

    if len(context_tokens) + len(needle_tokens) > effective_length:
        context_tokens = context_tokens[: effective_length - len(needle_tokens)]

    if depth_percent == 100:
        final_tokens = context_tokens + needle_tokens
        needle_start = len(context_tokens)
        needle_end = needle_start + len(needle_tokens)
        return final_tokens, needle_start, needle_end

    insertion_point = int(len(context_tokens) * (depth_percent / 100))
    if period_positions:
        valid_count = bisect_right(period_positions, len(context_tokens) - 1)
        valid_period_positions = period_positions[:valid_count]
        boundary_idx = bisect_right(valid_period_positions, insertion_point - 1) - 1
        insertion_point = valid_period_positions[boundary_idx] + 1 if boundary_idx >= 0 else 0

    needle_start = insertion_point
    needle_end = insertion_point + len(needle_tokens)
    final_tokens = context_tokens[:insertion_point] + needle_tokens + context_tokens[insertion_point:]
    return final_tokens, needle_start, needle_end


def insert_needle(
    haystack_text: str,
    needle: str,
    context_length: int,
    depth_percent: float,
    tokenizer,
    context_buffer: int = 200,
    model_name: str = "",
    use_hardcoded_periods: bool = True,
    force_llama2_periods: bool = False,
) -> tuple[list[int], int, int]:
    """Insert needle into haystack at depth_percent position.

    Faithful to the original: works in token space, aligns insertion to
    sentence boundaries (period tokens), applies a context buffer.

    Args:
        haystack_text: Source text for context
        needle: Text to insert
        context_length: Target length in tokens (including needle)
        depth_percent: Position (0-100) to insert needle
        tokenizer: HuggingFace tokenizer
        context_buffer: Tokens reserved for question + answer
        model_name: Model identifier for hardcoded period tokens
        use_hardcoded_periods: Use known hardcoded periods for supported models
        force_llama2_periods: DEBUG: Force Llama-2 tokens regardless of model

    Returns:
        (token_ids, needle_start_idx, needle_end_idx) where indices are
        positions in token_ids.
    """
    needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
    context_tokens = tokenizer.encode(haystack_text, add_special_tokens=False)
    period_tokens = get_period_tokens(model_name, tokenizer, use_hardcoded_periods, force_llama2_periods)
    period_positions = build_period_token_positions(context_tokens, period_tokens)
    return insert_needle_tokens(
        context_tokens,
        needle_tokens,
        context_length=context_length,
        depth_percent=depth_percent,
        period_positions=period_positions,
        context_buffer=context_buffer,
    )


# ---------------------------------------------------------------------------
# Needle finding (in tokenized prompts)
# ---------------------------------------------------------------------------


def find_needle_idx(prompt_ids: torch.Tensor, needle: str, tokenizer) -> tuple[int, int]:
    """Find needle span in prompt using 90% token overlap heuristic.

    Faithful to the original's find_needle_idx method.
    """
    needle_ids = tokenizer(needle, add_special_tokens=False)["input_ids"]
    return find_needle_idx_from_tokens(prompt_ids, needle_ids)


def find_needle_idx_from_tokens(prompt_ids: torch.Tensor, needle_ids: list[int]) -> tuple[int, int]:
    """Find needle span in prompt from pre-tokenized needle IDs.

    .. warning::
        Uses a 90% set-overlap heuristic. Can silently fail under BPE
        boundary drift (e.g. " There" vs "There" after re-tokenization)
        and can match wrong-but-similar windows. Prefer
        :func:`build_tracked_prompt`, which tracks the needle position
        through token-space composition and needs no search heuristic.
        Retained for the answer-within-needle narrowing in the NIAH
        behavioral detector, where the search window is constrained.
    """
    span_len = len(needle_ids)
    needle_set = set(needle_ids)

    for i in range(len(prompt_ids) - span_len + 1):
        token_span = prompt_ids[i : i + span_len]
        span_set = set(token_span.tolist())
        overlap = len(span_set.intersection(needle_set)) / len(needle_set)
        if overlap > 0.9:
            return i, i + span_len

    return -1, -1


# ---------------------------------------------------------------------------
# Token-space prompt composition (position-tracked, no search heuristics)
# ---------------------------------------------------------------------------


_CHAT_SENTINEL = "\u0001DECORE_NEEDLE_SENTINEL\u0001"


def _split_chat_template(tokenizer, disable_thinking: bool = True) -> tuple[str, str] | None:
    """Return (prefix, suffix) strings that bracket user content in chat templates.

    We call ``apply_chat_template`` with a unique sentinel as the user
    content and split the resulting string on that sentinel. This gives us
    the literal text that wraps user input, which we can tokenize separately
    and concat around our token-space content. Returns ``None`` if the
    tokenizer has no chat template.
    """
    if not (hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template):
        return None
    messages = [{"role": "user", "content": _CHAT_SENTINEL}]
    kwargs: dict = dict(tokenize=False, add_generation_prompt=True)
    if disable_thinking and "enable_thinking" in tokenizer.chat_template:
        kwargs["enable_thinking"] = False
    formatted = tokenizer.apply_chat_template(messages, **kwargs)
    if _CHAT_SENTINEL not in formatted:
        return None
    prefix, suffix = formatted.split(_CHAT_SENTINEL, 1)
    return prefix, suffix


def build_tracked_prompt(
    *,
    haystack_tokens: list[int],
    needle_tokens: list[int],
    context_length: int,
    depth_percent: float,
    period_positions: list[int] | None,
    tokenizer,
    prompt_template: str | None = None,
    question_tokens: list[int] | None = None,
    answer_tokens: list[int] | None = None,
    use_chat_template: bool = False,
    disable_thinking: bool = True,
    prompt_suffix_ids: list[int] | None = None,
    add_bos: bool = False,
    context_buffer: int = 200,
) -> tuple[torch.Tensor, int, int, int | None, int | None]:
    """Compose a prompt entirely in token space, tracking the needle position.

    Replaces the decode/re-encode round-trip pattern used across detectors
    (behavioral, contrastive, logit_contrib, cri). The key invariant: each
    text piece (chat-template prefix/suffix, prompt-template prefix/suffix,
    question) is tokenized exactly once, and every concat step tracks the
    needle offset. This avoids BPE boundary drift and the 90% set-overlap
    heuristic in :func:`find_needle_idx_from_tokens`, which silently skips
    or false-matches on re-tokenized prompts.

    Pipeline (pieces present only when applicable):
        [BOS] [chat_prefix] [tmpl_prefix | ""] [context_with_needle]
              [tmpl_suffix | question] [chat_suffix] [prompt_suffix]
              [answer_tokens]

    Args:
        haystack_tokens: Pre-tokenized haystack (``add_special_tokens=False``).
        needle_tokens: Pre-tokenized needle.
        period_positions: Pre-computed period positions in ``haystack_tokens``
            for sentence-boundary depth alignment (see
            :func:`insert_needle_tokens`).
        prompt_template: NoLiMa-style template string containing
            ``{haystack}``. When None, ``question_tokens`` is appended after
            the context instead.
        question_tokens: NIAH question (used when ``prompt_template`` is None).
        answer_tokens: Gold answer tokens appended after the prompt (for
            teacher-forced CRI-style scoring). Returns the answer span.
        use_chat_template: Wrap the composed prompt in the tokenizer's chat
            template. Uses a sentinel split (not tokenize=True) so we can
            tokenize the chat wrapper pieces independently.
        add_bos: Prepend ``tokenizer.bos_token_id``. Only applies when
            ``use_chat_template=False`` (chat templates typically include
            BOS themselves).
        prompt_suffix_ids: Tokens appended after the chat template (e.g. the
            GPT-oss ``<|channel|>final<|message|>`` suffix).
        context_buffer: Tokens reserved beyond the needle for question+answer
            in :func:`insert_needle_tokens`.

    Returns:
        ``(input_ids [1, L], needle_start, needle_end, answer_start, answer_end)``.
        ``answer_start``/``answer_end`` are None when ``answer_tokens`` is None.
        Needle positions always refer to the final token sequence.
    """
    # 1. Insert needle into haystack at sentence-aligned depth.
    context_tokens, needle_start_in_ctx, needle_end_in_ctx = insert_needle_tokens(
        haystack_tokens,
        needle_tokens,
        context_length=context_length,
        depth_percent=depth_percent,
        period_positions=period_positions,
        context_buffer=context_buffer,
    )
    needle_len = needle_end_in_ctx - needle_start_in_ctx

    # 2. Compose the "content" region: template-wrapped context, or
    # context + question.
    if prompt_template is not None and "{haystack}" in prompt_template:
        # FIXME(aryo): template_prefix/suffix are tokenized independently of
        # the haystack, so BPE may tokenize the concat differently than
        # encoding the whole string as one. This is a minor fidelity loss
        # confined to two token boundaries at the {haystack} split — far
        # from needle/answer positions. Essential for position tracking.
        tmpl_prefix_str, tmpl_suffix_str = prompt_template.split("{haystack}", 1)
        tmpl_prefix_tokens = tokenizer.encode(tmpl_prefix_str, add_special_tokens=False)
        tmpl_suffix_tokens = tokenizer.encode(tmpl_suffix_str, add_special_tokens=False)
        content_tokens = tmpl_prefix_tokens + context_tokens + tmpl_suffix_tokens
        needle_start_in_content = len(tmpl_prefix_tokens) + needle_start_in_ctx
    else:
        assert question_tokens is not None, "question_tokens required when prompt_template is None"
        content_tokens = context_tokens + question_tokens
        needle_start_in_content = needle_start_in_ctx

    # 3. Optionally wrap in chat template (token-space splice).
    if use_chat_template:
        split = _split_chat_template(tokenizer, disable_thinking=disable_thinking)
        if split is None:
            # Caller asked for a chat template but the tokenizer has none;
            # fall back to raw content. Matches pre-existing detector behaviour
            # (they wrapped in try/except returning None).
            wrapped_tokens = content_tokens
            needle_start_wrapped = needle_start_in_content
        else:
            chat_prefix_str, chat_suffix_str = split
            chat_prefix_tokens = tokenizer.encode(chat_prefix_str, add_special_tokens=False)
            chat_suffix_tokens = tokenizer.encode(chat_suffix_str, add_special_tokens=False)
            wrapped_tokens = chat_prefix_tokens + content_tokens + chat_suffix_tokens
            needle_start_wrapped = len(chat_prefix_tokens) + needle_start_in_content
    else:
        wrapped_tokens = content_tokens
        needle_start_wrapped = needle_start_in_content

    # 4. Prepend BOS if requested and not already provided by the chat template.
    bos_offset = 0
    if add_bos and not use_chat_template:
        assert tokenizer.bos_token_id is not None, "add_bos=True but tokenizer has no bos_token_id"
        wrapped_tokens = [tokenizer.bos_token_id, *wrapped_tokens]
        bos_offset = 1

    # 5. Append prompt suffix (e.g. GPT-oss channel hint).
    if prompt_suffix_ids:
        wrapped_tokens = wrapped_tokens + list(prompt_suffix_ids)

    # 6. Append teacher-forced answer.
    answer_start: int | None
    answer_end: int | None
    if answer_tokens is not None:
        assert len(answer_tokens) >= 1, "answer_tokens must be non-empty"
        answer_start = len(wrapped_tokens)
        wrapped_tokens = wrapped_tokens + list(answer_tokens)
        answer_end = len(wrapped_tokens)
    else:
        answer_start = None
        answer_end = None

    needle_start = needle_start_wrapped + bos_offset
    needle_end = needle_start + needle_len

    # Sanity: the tracked position really contains the needle tokens verbatim.
    # Catches any off-by-one in the composition above.
    assert wrapped_tokens[needle_start:needle_end] == list(needle_tokens), (
        f"Needle position tracking failed: expected {needle_tokens[:6]}..., "
        f"got {wrapped_tokens[needle_start : needle_start + 6]}..."
    )

    input_ids = torch.tensor([wrapped_tokens])
    return input_ids, needle_start, needle_end, answer_start, answer_end
