"""Unit tests for locos/detect_retrieval_heads.py.

No GPU required -- all model interactions are mocked.
"""

import json
from collections import defaultdict
from unittest.mock import MagicMock

import pytest
import torch

from locos.detectors.behavioral import (
    detect_single_trial,
    load_checkpoint,
    save_checkpoint,
)
from locos.utils.needle_utils import (
    build_period_token_positions,
    find_needle_idx,
    find_needle_idx_from_tokens,
    insert_needle,
    insert_needle_tokens,
    load_needles,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tokenizer():
    """A minimal tokenizer mock that maps characters to token IDs."""
    tok = MagicMock()

    # Simple encoding: each character is a token (ASCII value)
    def encode_fn(text, add_special_tokens=True, return_tensors=None):
        ids = [ord(c) for c in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return ids

    tok.encode = encode_fn
    tok.decode = lambda ids, **kw: "".join(chr(i) for i in ids if 32 <= i < 127)
    tok.convert_ids_to_tokens = lambda x: chr(x) if 32 <= x < 127 else f"<{x}>"
    tok.bos_token_id = None
    tok.eos_token_id = 0
    tok.side_effect = lambda text, add_special_tokens=False: {"input_ids": [ord(c) for c in text]}
    return tok


@pytest.fixture
def needles_dir(tmp_path):
    """Create a temporary haystack directory with test data."""
    needles = [
        {
            "needle": "The secret code is alpha-bravo-charlie.",
            "question": "What is the secret code?",
            "real_needle": "alpha-bravo-charlie",
        },
    ]
    (tmp_path / "needles.jsonl").write_text("\n".join(json.dumps(n) for n in needles))
    # Small haystack text
    (tmp_path / "PaulGrahamEssays.txt").write_text("This is some filler text. " * 500)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: load_needles
# ---------------------------------------------------------------------------


def test_load_needles(needles_dir):
    needles = load_needles(needles_dir)
    assert len(needles) == 1
    assert needles[0]["needle"] == "The secret code is alpha-bravo-charlie."
    assert needles[0]["real_needle"] == "alpha-bravo-charlie"


def test_load_needles_missing(tmp_path):
    with pytest.raises(AssertionError, match=r"needles\.jsonl not found"):
        load_needles(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Tests: insert_needle
# ---------------------------------------------------------------------------


def test_insert_needle_positions(mock_tokenizer):
    """Needle tokens appear at the reported positions."""
    haystack = "A" * 500 + "." + "B" * 500  # Has a period in the middle
    needle = "XYZ"

    tokens, start, end = insert_needle(
        haystack,
        needle,
        context_length=800,
        depth_percent=50,
        tokenizer=mock_tokenizer,
        context_buffer=0,
    )

    needle_ids = mock_tokenizer.encode(needle, add_special_tokens=False)
    assert tokens[start:end] == needle_ids
    assert end - start == len(needle_ids)


def test_insert_needle_depth_100(mock_tokenizer):
    """depth_percent=100 puts needle at the end."""
    haystack = "A" * 200 + "." + "B" * 200
    needle = "XY"

    tokens, start, end = insert_needle(
        haystack,
        needle,
        context_length=500,
        depth_percent=100,
        tokenizer=mock_tokenizer,
        context_buffer=0,
    )

    needle_ids = mock_tokenizer.encode(needle, add_special_tokens=False)
    assert tokens[start:end] == needle_ids
    # Needle should be at the very end
    assert end == len(tokens)


def test_insert_needle_tokens_matches_string_path(mock_tokenizer):
    """Token-level insertion matches the original string-based helper."""
    haystack = "A" * 500 + "." + "B" * 500 + "." + "C" * 500
    needle = "XYZ"
    context_length = 900
    depth_percent = 50

    expected_tokens, expected_start, expected_end = insert_needle(
        haystack,
        needle,
        context_length=context_length,
        depth_percent=depth_percent,
        tokenizer=mock_tokenizer,
        context_buffer=0,
    )

    haystack_tokens = mock_tokenizer.encode(haystack, add_special_tokens=False)
    needle_tokens = mock_tokenizer.encode(needle, add_special_tokens=False)
    period_positions = build_period_token_positions(haystack_tokens, {ord(".")})

    actual_tokens, actual_start, actual_end = insert_needle_tokens(
        haystack_tokens,
        needle_tokens,
        context_length=context_length,
        depth_percent=depth_percent,
        period_positions=period_positions,
        context_buffer=0,
    )

    assert actual_tokens == expected_tokens
    assert (actual_start, actual_end) == (expected_start, expected_end)


# ---------------------------------------------------------------------------
# Tests: find_needle_idx
# ---------------------------------------------------------------------------


def test_find_needle_idx_exact(mock_tokenizer):
    """find_needle_idx locates needle with exact match."""
    needle = "XYZ"
    prompt = "AAAAAXYZBBBB"
    prompt_ids = torch.tensor([ord(c) for c in prompt])

    start, end = find_needle_idx(prompt_ids, needle, mock_tokenizer)
    assert start == 5
    assert end == 8


def test_find_needle_idx_not_found(mock_tokenizer):
    """Returns (-1, -1) when needle is absent."""
    prompt_ids = torch.tensor([ord(c) for c in "AAAAAABBBB"])
    start, end = find_needle_idx(prompt_ids, "XYZ", mock_tokenizer)
    assert start == -1
    assert end == -1


def test_find_needle_idx_from_tokens_matches_string_helper(mock_tokenizer):
    """Pre-tokenized matching behaves like the string helper."""
    needle = "XYZ"
    prompt = "AAAAAXYZBBBB"
    prompt_ids = torch.tensor([ord(c) for c in prompt])
    needle_ids = mock_tokenizer.encode(needle, add_special_tokens=False)

    assert find_needle_idx_from_tokens(prompt_ids, needle_ids) == (5, 8)
    assert find_needle_idx(prompt_ids, needle, mock_tokenizer) == (5, 8)


# ---------------------------------------------------------------------------
# Tests: scoring formula
# ---------------------------------------------------------------------------


def test_scoring_formula():
    """Verify the 1/(needle_end - needle_start) accumulation."""
    # Create a mock model that returns controlled attention weights
    num_layers = 2
    num_heads = 2
    needle_start = 5
    needle_end = 10  # needle_len = 5
    seq_len = 20

    # Build mock input_ids where token at position 7 has id=42
    input_ids = torch.arange(seq_len).unsqueeze(0)  # (1, 20)
    # The generated token will be 7 (matching position 7 in input)
    # Position 7 is within needle span [5, 10)

    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.convert_ids_to_tokens = lambda x: "a"  # Non-newline
    tokenizer.eos_token_id = -1
    tokenizer.encode = lambda *a, **kw: [10]  # newline check

    # Mock model outputs for 1 decode step, then stop via EOS
    def make_outputs(step):
        # Attention: layer 0, head 0 attends to position 7 (in needle)
        # Attention: layer 0, head 1 attends to position 2 (outside needle)
        # All other heads attend to position 0
        attn_0 = torch.zeros(1, num_heads, 1, seq_len + step)
        attn_0[0, 0, 0, 7] = 1.0  # head 0 -> pos 7 (in needle)
        attn_0[0, 1, 0, 2] = 1.0  # head 1 -> pos 2 (outside needle)

        attn_1 = torch.zeros(1, num_heads, 1, seq_len + step)
        attn_1[0, 0, 0, 7] = 1.0  # head 0 -> pos 7 (in needle)
        attn_1[0, 1, 0, 7] = 1.0  # head 1 -> pos 7 (in needle)

        # Logits: argmax returns token ID that matches input_ids[0, 7] = 7
        logits = torch.zeros(1, 1, seq_len)
        logits[0, 0, 7] = 10.0  # argmax -> 7, matches input_ids[0, 7]

        out = MagicMock()
        out.attentions = (attn_0, attn_1)
        out.logits = logits
        out.past_key_values = MagicMock()
        return out

    # First call: prefill (no attentions), second call: decode step
    prefill_out = MagicMock()
    prefill_out.past_key_values = MagicMock()

    call_count = [0]

    def model_call(**kwargs):
        if not kwargs.get("output_attentions", False):
            return prefill_out
        result = make_outputs(call_count[0])
        call_count[0] += 1
        return result

    model.side_effect = model_call
    # get_input_device uses next(model.parameters()).device
    dummy_param = torch.nn.Parameter(torch.empty(0))
    model.parameters = lambda: iter([dummy_param])

    # Token 7 is a normal token (scored); the loop stops via max_decode_steps.
    tokenizer.convert_ids_to_tokens = lambda x: "a"

    scores, _ = detect_single_trial(
        model,
        tokenizer,
        input_ids,
        needle_start,
        needle_end,
        num_layers,
        num_heads,
        prefill_attn_impl="eager",
        max_decode_steps=1,
    )

    expected_score = 1.0 / (needle_end - needle_start)  # 1/5 = 0.2

    # Layer 0, head 0: attended to pos 7, generated token 7 == input[7] = 7 ✓
    assert abs(scores[0][0] - expected_score) < 1e-6

    # Layer 0, head 1: attended to pos 2, outside needle → 0
    assert scores[0][1] == 0.0

    # Layer 1, head 0: attended to pos 7, generated token 7 == input[7] ✓
    assert abs(scores[1][0] - expected_score) < 1e-6

    # Layer 1, head 1: attended to pos 7, generated token 7 == input[7] ✓
    assert abs(scores[1][1] - expected_score) < 1e-6


# ---------------------------------------------------------------------------
# Tests: checkpoint / resume
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path):
    """Checkpoint save/load preserves data."""
    head_counter = defaultdict(list)
    head_counter["0-0"] = [0.5, 0.3]
    head_counter["1-2"] = [0.9]
    completed = [[1000, 50, 0], [2000, 50, 1]]

    path = tmp_path / "test.checkpoint.json"
    save_checkpoint(head_counter, completed, path)

    loaded_counter, loaded_completed = load_checkpoint(path)
    assert loaded_counter["0-0"] == [0.5, 0.3]
    assert loaded_counter["1-2"] == [0.9]
    assert loaded_completed == completed


def test_load_checkpoint_missing(tmp_path):
    """Missing checkpoint returns empty defaults."""
    counter, completed = load_checkpoint(tmp_path / "nonexistent.json")
    assert len(counter) == 0
    assert len(completed) == 0


# ---------------------------------------------------------------------------
# Tests: output format compatibility
# ---------------------------------------------------------------------------


def test_output_format_loadable(tmp_path):
    """Output JSON is loadable by locos_eval's load_retrieval_heads()."""
    from locos_eval.retrieval_heads import load_retrieval_heads

    # Create a mock output file
    result = {
        "0-0": [0.0, 0.0],
        "0-1": [0.5, 0.3],
        "1-0": [0.9, 0.8],
        "1-1": [0.1, 0.0],
    }
    output_path = tmp_path / "test_heads.json"
    output_path.write_text(json.dumps(result))

    heads = load_retrieval_heads(str(output_path), num_heads=2)
    assert len(heads) == 2
    # Top head should be (1, 0) with mean 0.85
    assert heads[0] == (1, 0)
    # Second should be (0, 1) with mean 0.4
    assert heads[1] == (0, 1)
