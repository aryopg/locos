"""Unit tests for locos/detect_cri.py.

No GPU required -- all model interactions are mocked.
"""

import json
from collections import defaultdict
from unittest.mock import MagicMock

import pytest
import torch

from locos.detectors.cri import (
    CRIHookManager,
    build_cri_prompt_pair,
    extract_answer_logprobs,
    load_checkpoint,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# Tests: extract_answer_logprobs
# ---------------------------------------------------------------------------


def test_extract_answer_logprobs_basic():
    """Log-probs are extracted at the correct positions."""
    vocab_size = 10
    seq_len = 8
    # Logits: position t predicts token t+1
    logits = torch.zeros(1, seq_len, vocab_size)
    # At position 4, the model predicts token 7 with high logit
    logits[0, 4, 7] = 10.0
    # At position 5, the model predicts token 3 with high logit
    logits[0, 5, 3] = 10.0

    # Input IDs: tokens at positions 5 and 6 are the answer
    input_ids = torch.tensor([[0, 0, 0, 0, 0, 7, 3, 0]])

    answer_start = 5  # token 7
    answer_end = 7  # token 3 (exclusive end)

    log_probs = extract_answer_logprobs(logits, input_ids, answer_start, answer_end)
    assert log_probs.shape == (2,)

    # First answer token (7) predicted from position 4 — should have high logprob
    assert log_probs[0].item() > -1.0
    # Second answer token (3) predicted from position 5 — should have high logprob
    assert log_probs[1].item() > -1.0


def test_extract_answer_logprobs_single_token():
    """Works with single-token answers."""
    logits = torch.zeros(1, 5, 10)
    logits[0, 2, 5] = 10.0
    input_ids = torch.tensor([[0, 0, 0, 5, 0]])

    log_probs = extract_answer_logprobs(logits, input_ids, 3, 4)
    assert log_probs.shape == (1,)
    assert log_probs[0].item() > -1.0


def test_extract_answer_logprobs_asserts_start():
    """answer_start must be >= 1."""
    with pytest.raises(AssertionError):
        extract_answer_logprobs(torch.zeros(1, 5, 10), torch.zeros(1, 5, dtype=torch.long), 0, 1)


# ---------------------------------------------------------------------------
# Tests: CRIHookManager
# ---------------------------------------------------------------------------


def _make_mock_model(num_layers=2, num_heads=4, head_dim=8):
    """Create a mock model with the expected structure."""
    model = MagicMock()
    model.model = MagicMock()

    layers = []
    for _ in range(num_layers):
        layer = MagicMock()
        attn = MagicMock()
        attn.o_proj = torch.nn.Linear(num_heads * head_dim, num_heads * head_dim, bias=False)
        layer.self_attn = attn
        layers.append(layer)

    model.model.layers = layers
    return model


def test_hook_manager_init():
    """HookManager finds attention layers correctly."""
    model = _make_mock_model(num_layers=3, num_heads=4, head_dim=8)
    manager = CRIHookManager(model, num_layers=3, num_heads=4, head_dim=8)
    assert len(manager._attn_layers) == 3


def test_hook_manager_wrong_num_layers():
    """Raises if model has different number of layers."""
    model = _make_mock_model(num_layers=2)
    with pytest.raises(AssertionError, match="Expected 3"):
        CRIHookManager(model, num_layers=3, num_heads=4, head_dim=8)


def test_hook_manager_install_remove():
    """Hooks can be installed and removed."""
    model = _make_mock_model(num_layers=2, num_heads=4, head_dim=8)
    manager = CRIHookManager(model, num_layers=2, num_heads=4, head_dim=8)

    manager.install_hooks()
    assert len(manager.handles) == 2

    manager.remove_hooks()
    assert len(manager.handles) == 0


def test_hook_manager_capture_mode():
    """In capture mode, the hook stores activations."""
    num_heads = 4
    head_dim = 8
    model = _make_mock_model(num_layers=1, num_heads=num_heads, head_dim=head_dim)
    manager = CRIHookManager(model, num_layers=1, num_heads=num_heads, head_dim=head_dim)
    manager.install_hooks()

    # Simulate calling the o_proj pre-hook
    manager.set_capture_mode()
    hidden = torch.randn(1, 10, num_heads * head_dim)

    # Call the hook directly
    hook_fn = manager._make_hook(0)
    result = hook_fn(None, (hidden,))

    assert result is None  # Capture doesn't modify input
    assert 0 in manager.captured
    assert manager.captured[0].shape == (1, 10, num_heads, head_dim)

    manager.remove_hooks()


def test_hook_manager_patch_mode():
    """In patch mode, the hook replaces one head's activation."""
    num_heads = 4
    head_dim = 8
    model = _make_mock_model(num_layers=1, num_heads=num_heads, head_dim=head_dim)
    manager = CRIHookManager(model, num_layers=1, num_heads=num_heads, head_dim=head_dim)
    manager.install_hooks()

    # First capture
    manager.set_capture_mode()
    clean_hidden = torch.ones(1, 10, num_heads * head_dim) * 5.0
    hook_fn = manager._make_hook(0)
    hook_fn(None, (clean_hidden,))

    # Now patch head 2 at layer 0
    manager.set_patch_mode(layer_idx=0, head_idx=2)

    corrupt_hidden = torch.zeros(1, 10, num_heads * head_dim)
    result = hook_fn(None, (corrupt_hidden,))

    assert result is not None
    patched = result[0]
    patched_3d = patched.view(1, 10, num_heads, head_dim)

    # Head 2 should be overwritten with the clean activation (5.0)
    assert torch.allclose(patched_3d[:, :, 2, :], torch.ones(1, 10, head_dim) * 5.0)
    # Other heads should remain 0 (from corrupt_hidden)
    assert torch.allclose(patched_3d[:, :, 0, :], torch.zeros(1, 10, head_dim))
    assert torch.allclose(patched_3d[:, :, 1, :], torch.zeros(1, 10, head_dim))
    assert torch.allclose(patched_3d[:, :, 3, :], torch.zeros(1, 10, head_dim))

    manager.remove_hooks()


def test_hook_manager_off_mode():
    """In off mode, hooks don't modify input."""
    num_heads = 4
    head_dim = 8
    model = _make_mock_model(num_layers=1, num_heads=num_heads, head_dim=head_dim)
    manager = CRIHookManager(model, num_layers=1, num_heads=num_heads, head_dim=head_dim)

    manager.set_off()
    hook_fn = manager._make_hook(0)
    hidden = torch.randn(1, 10, num_heads * head_dim)
    result = hook_fn(None, (hidden,))

    assert result is None  # No modification


# ---------------------------------------------------------------------------
# Tests: Checkpoint / resume
# ---------------------------------------------------------------------------


def test_cri_checkpoint_roundtrip(tmp_path):
    """CRI checkpoint save/load preserves data."""
    scores = defaultdict(list)
    scores["0-0"] = [0.1, 0.2]
    scores["1-3"] = [0.5]
    completed = ["trial_1", "trial_2"]

    path = tmp_path / "cri.checkpoint.json"
    save_checkpoint(scores, completed, path)

    loaded_scores, loaded_completed = load_checkpoint(path)
    assert loaded_scores["0-0"] == [0.1, 0.2]
    assert loaded_scores["1-3"] == [0.5]
    assert loaded_completed == ["trial_1", "trial_2"]


def test_cri_load_checkpoint_missing(tmp_path):
    """Missing checkpoint returns empty defaults."""
    scores, completed = load_checkpoint(tmp_path / "nonexistent.json")
    assert len(scores) == 0
    assert len(completed) == 0


# ---------------------------------------------------------------------------
# Tests: Output format compatibility
# ---------------------------------------------------------------------------


def test_cri_output_loadable(tmp_path):
    """CRI envelope output is loadable by load_retrieval_heads()."""
    from locos_eval.retrieval_heads import load_retrieval_heads

    result = {
        "meta": {
            "method": "cri",
            "dataset": "nolima",
        },
        "scores": {
            "0-0": [0.01, 0.02],
            "0-1": [0.5, 0.6],
            "1-0": [0.9, 0.8],
            "1-1": [0.1, 0.0],
        },
    }
    output_path = tmp_path / "test_cri.json"
    output_path.write_text(json.dumps(result))

    heads = load_retrieval_heads(str(output_path), num_heads=2)
    assert len(heads) == 2
    # Top head should be (1, 0) with mean 0.85
    assert heads[0] == (1, 0)
    # Second should be (0, 1) with mean 0.55
    assert heads[1] == (0, 1)


# ---------------------------------------------------------------------------
# Tests: build_cri_prompt_pair — token-space splice shape invariant
# ---------------------------------------------------------------------------


class _CharTokenizer:
    """Character-level tokenizer: each char → its ASCII codepoint.

    Whitespace and punctuation survive the round-trip, so this is a more
    faithful substitute for an HF tokenizer than a MagicMock for exercising
    ``build_cri_prompt_pair`` end-to-end.
    """

    bos_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(i) for i in token_ids if 0 <= i < 0x110000)


@pytest.fixture
def char_tokenizer():
    return _CharTokenizer()


def _haystack_with_period(n_chars: int = 800) -> str:
    # Periods act as sentence boundaries for insert_needle's depth alignment.
    return ("A" * 40 + ". ") * (n_chars // 42 + 1)


def test_build_cri_prompt_pair_remove_shape(char_tokenizer):
    """Remove corruption: clean and corrupt tensors have identical shape."""
    clean_ids, corrupt_ids, ans_start, ans_end = build_cri_prompt_pair(
        haystack_text=_haystack_with_period(),
        needle_text="XYZ-SECRET-PAYLOAD",
        question="What is the secret?",
        answer_text="XYZ",
        context_length=600,
        depth_percent=50,
        tokenizer=char_tokenizer,
        corruption="remove",
    )
    assert clean_ids.shape == corrupt_ids.shape
    assert ans_end > ans_start >= 1
    # Clean and corrupt must differ somewhere (the needle span) — otherwise
    # we accidentally produced identical inputs.
    assert not torch.equal(clean_ids, corrupt_ids)


def test_build_cri_prompt_pair_remove_filler_in_span(char_tokenizer):
    """Remove corruption replaces the needle span with filler (period) tokens."""
    clean_ids, corrupt_ids, _, _ = build_cri_prompt_pair(
        haystack_text=_haystack_with_period(),
        needle_text="QWERTY-NEEDLE",
        question="Q?",
        answer_text="A",
        context_length=500,
        depth_percent=25,
        tokenizer=char_tokenizer,
        corruption="remove",
    )
    # Positions where clean and corrupt differ must all be filler (period = 46)
    diff_mask = clean_ids[0] != corrupt_ids[0]
    assert diff_mask.any(), "expected needle span to differ"
    period_id = ord(".")
    assert torch.all(corrupt_ids[0][diff_mask] == period_id)


def test_build_cri_prompt_pair_scramble_shape(char_tokenizer):
    """Scramble corruption preserves shape even when scrambled tokenizes differently."""
    # Scrambled needle is *longer* in char count than original — would have
    # drifted under the text-round-trip approach, but splice clamps to span.
    clean_ids, corrupt_ids, _, _ = build_cri_prompt_pair(
        haystack_text=_haystack_with_period(),
        needle_text="Megan",
        question="Q?",
        answer_text="A",
        context_length=500,
        depth_percent=50,
        tokenizer=char_tokenizer,
        corruption="scramble",
        corrupted_needle_text="Montgomery",  # 10 chars vs 5
    )
    assert clean_ids.shape == corrupt_ids.shape


def test_build_cri_prompt_pair_scramble_shorter_pads(char_tokenizer):
    """Scrambled needle shorter than original gets right-padded with filler."""
    clean_ids, corrupt_ids, _, _ = build_cri_prompt_pair(
        haystack_text=_haystack_with_period(),
        needle_text="Montgomery",
        question="Q?",
        answer_text="A",
        context_length=500,
        depth_percent=50,
        tokenizer=char_tokenizer,
        corruption="scramble",
        corrupted_needle_text="Al",  # 2 chars vs 10
    )
    assert clean_ids.shape == corrupt_ids.shape


def test_build_cri_prompt_pair_with_template(char_tokenizer):
    """Shape invariant holds with NoLiMa-style prompt_template wrapping."""
    template = "Instruction: find the needle.\n\n{haystack}\n\nAnswer:"
    clean_ids, corrupt_ids, ans_start, ans_end = build_cri_prompt_pair(
        haystack_text=_haystack_with_period(),
        needle_text="RARETOKEN-NEEDLE",
        question="Q?",  # ignored when template present
        answer_text="A",
        context_length=500,
        depth_percent=50,
        tokenizer=char_tokenizer,
        corruption="remove",
        prompt_template=template,
    )
    assert clean_ids.shape == corrupt_ids.shape
    # Answer span remains valid
    assert ans_end > ans_start >= 1
    assert ans_end <= clean_ids.shape[1]


def test_build_cri_prompt_pair_asserts_needle_required(char_tokenizer):
    """needle_text is required (clean needle, not None)."""
    with pytest.raises(AssertionError, match="needle_text is required"):
        build_cri_prompt_pair(
            haystack_text=_haystack_with_period(),
            needle_text=None,
            question="Q?",
            answer_text="A",
            context_length=500,
            depth_percent=50,
            tokenizer=char_tokenizer,
            corruption="remove",
        )


def test_build_cri_prompt_pair_asserts_scramble_needs_corrupted(char_tokenizer):
    """scramble corruption requires corrupted_needle_text."""
    with pytest.raises(AssertionError, match="corrupted_needle_text required"):
        build_cri_prompt_pair(
            haystack_text=_haystack_with_period(),
            needle_text="N",
            question="Q?",
            answer_text="A",
            context_length=500,
            depth_percent=50,
            tokenizer=char_tokenizer,
            corruption="scramble",
            corrupted_needle_text=None,
        )


class _BoundarySensitiveTokenizer:
    """Tokenizer that encodes a word differently depending on leading space.

    Mimics BPE behaviour where " There" and "There" map to different IDs.
    Used to verify that ``build_cri_prompt_pair`` does NOT rely on a
    re-tokenisation heuristic (which would fail here).
    """

    bos_token_id = None

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._inv: dict[int, str] = {}
        self._next = 1000

    def _tok(self, piece: str) -> int:
        if piece not in self._vocab:
            self._vocab[piece] = self._next
            self._inv[self._next] = piece
            self._next += 1
        return self._vocab[piece]

    def encode(self, text, add_special_tokens=False):
        import re

        # Word-level with leading space preserved as part of the token, mimicking BPE.
        pieces = re.findall(r" ?[A-Za-z0-9]+|[^A-Za-z0-9\s]| +", text)
        return [self._tok(p) for p in pieces]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(self._inv.get(t, "?") for t in token_ids)


def test_build_cri_prompt_pair_boundary_sensitive_tokenizer():
    """Needle tracked correctly even when BPE would re-tokenize needle differently.

    Regression: under a leading-space-sensitive tokenizer, encoding the needle
    in isolation yields different IDs than the needle tokens produced by
    insert_needle. Token-space composition must track the position directly
    rather than searching for the needle in the final sequence.
    """
    tok = _BoundarySensitiveTokenizer()
    haystack = ("Some filler sentence about nothing. " * 80).strip()
    needle = "There was a vegan guest, named Gary."

    # Sanity check: needle-in-isolation and needle-with-leading-space tokenise
    # to different ID sequences under this tokenizer.
    iso = tok.encode(needle)
    with_space = tok.encode(" " + needle)
    assert iso != with_space, "tokenizer should be boundary-sensitive for this test"

    clean_ids, corrupt_ids, ans_start, ans_end = build_cri_prompt_pair(
        haystack_text=haystack,
        needle_text=needle,
        question="Who is the vegan guest?",
        answer_text="Gary",
        context_length=600,
        depth_percent=25,
        tokenizer=tok,
        corruption="remove",
    )
    assert clean_ids.shape == corrupt_ids.shape
    assert ans_end > ans_start >= 1
    # Exactly one contiguous span must differ (the needle).
    diff_positions = (clean_ids[0] != corrupt_ids[0]).nonzero(as_tuple=True)[0]
    assert len(diff_positions) > 0
    assert (diff_positions[-1] - diff_positions[0] + 1).item() == len(diff_positions)


def test_build_cri_prompt_pair_asserts_unknown_corruption(char_tokenizer):
    with pytest.raises(AssertionError, match="Unknown corruption"):
        build_cri_prompt_pair(
            haystack_text=_haystack_with_period(),
            needle_text="N",
            question="Q?",
            answer_text="A",
            context_length=500,
            depth_percent=50,
            tokenizer=char_tokenizer,
            corruption="bogus",
        )


def test_flat_format_still_works(tmp_path):
    """Original flat format still works after envelope support."""
    from locos_eval.retrieval_heads import load_retrieval_heads

    result = {
        "0-0": [0.0, 0.0],
        "0-1": [0.5, 0.3],
        "1-0": [0.9, 0.8],
        "1-1": [0.1, 0.0],
    }
    output_path = tmp_path / "test_flat.json"
    output_path.write_text(json.dumps(result))

    heads = load_retrieval_heads(str(output_path), num_heads=2)
    assert len(heads) == 2
    assert heads[0] == (1, 0)
