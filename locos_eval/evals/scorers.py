"""Standalone scoring functions that do not depend on Inspect AI.

Each function is a pure function: string(s) in, float/bool/dict out.
Heavy libraries (rouge_score, bert_score, anthropic) are lazily imported
inside the functions that need them.
"""

from __future__ import annotations

import json
import re
import string
import time

# ---------------------------------------------------------------------------
# Text-overlap scorers
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def rouge_l_score(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F-measure between *prediction* and *reference*.

    Uses the Google ``rouge_score`` library with stemming enabled.

    Returns:
        Float in [0, 1].  0.0 when *prediction* is empty.
    """
    from rouge_score.rouge_scorer import RougeScorer  # lazy import

    scorer = RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure


def bertscore_f1(prediction: str, reference: str) -> float:
    """Compute BERTScore F1 for a single prediction/reference pair.

    Uses the ``bert_score`` library with English defaults.

    Returns:
        Float in [0, 1].
    """
    from bert_score import score as _bert_score  # lazy import

    _p, _r, f1 = _bert_score([prediction], [reference], lang="en", verbose=False)
    return f1.item()


def bertscore_f1_batch(predictions: list[str], references: list[str]) -> list[float]:
    """Compute BERTScore F1 for a batch of prediction/reference pairs.

    Loads the model once and scores all pairs in a single call.

    Returns:
        List of floats in [0, 1], one per pair.
    """
    assert len(predictions) == len(references), "predictions and references must have the same length"
    from bert_score import score as _bert_score  # lazy import

    _p, _r, f1 = _bert_score(predictions, references, lang="en", verbose=False)
    return f1.tolist()


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-Ja-j])\s*</answer>")
_ANSWER_IS_RE = re.compile(r"(?:answer|option)\s+(?:is\s+)?([A-Ja-j])\b", re.IGNORECASE)
_STANDALONE_LETTER_RE = re.compile(r"\b([A-Ja-j])\b")

# Free-form variant: matches whatever sits between <answer>...</answer>.
# Multi-line, non-greedy. Used by free-form QA tasks (BABILong, MuSiQue).
_ANSWER_TAG_FREEFORM_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer_text(text: str) -> str | None:
    """Extract free-form text between ``<answer>...</answer>`` tags.

    If multiple tags are present, returns the *last* one (matches the
    "think step by step, then answer" pattern where the model may rehearse
    candidate answers before committing to a final tag).

    Returns:
        The trimmed inner text, or ``None`` if no closing tag is found.
    """
    matches = _ANSWER_TAG_FREEFORM_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_answer_letter(text: str) -> str | None:
    """Extract a multiple-choice answer letter (A-J) from free-form *text*.

    Priority order:
        1. ``<answer>X</answer>`` tag
        2. "answer/option [is] X" pattern
        3. First standalone letter A-J

    Returns:
        Uppercase letter ``"A"``-``"J"``, or ``None`` if no letter found.
    """
    # Priority 1: explicit tag
    m = _ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).upper()

    # Priority 2: natural-language "answer is X" / "option X"
    m = _ANSWER_IS_RE.search(text)
    if m:
        return m.group(1).upper()

    # Priority 3: first standalone letter
    m = _STANDALONE_LETTER_RE.search(text)
    if m:
        return m.group(1).upper()

    return None


# ---------------------------------------------------------------------------
# Answer normalisation & subspan matching
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    """Normalize an answer string for comparison.

    Steps: lowercase, strip articles (a/an/the), remove punctuation,
    collapse whitespace, strip outer whitespace.

    Returns:
        Cleaned string.
    """
    text = text.lower()
    text = _ARTICLES_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def subspan_match(prediction: str, reference: str) -> bool:
    """Check whether *reference* appears as a subspan of *prediction*.

    Both strings are normalized before comparison.

    Returns:
        ``True`` if the normalized reference is a non-empty substring of the
        normalized prediction, ``False`` otherwise.
    """
    norm_ref = normalize_answer(reference)
    if not norm_ref:
        return False
    return norm_ref in normalize_answer(prediction)


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


def call_llm_judge(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-haiku-4-5-20251001",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> dict:
    """Call an Anthropic model and parse the JSON response.

    Uses the Anthropic SDK directly.  Retries on transient failures.
    Strips markdown code blocks before parsing.

    Args:
        system_prompt: System message content.
        user_prompt: User message content.
        model: Anthropic model identifier.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        max_retries: Number of attempts before giving up.

    Returns:
        Parsed JSON dict from the model response, or ``{}`` on failure.
    """
    import anthropic  # lazy import

    client = anthropic.Anthropic()

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text

            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
            raw = re.sub(r"\n?```\s*$", "", raw.strip())

            return json.loads(raw)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))

    # All retries exhausted
    return {}
