#!/usr/bin/env python3
"""Detect retrieval heads via Causal Retrieval Importance (CRI).

CRI uses activation patching to causally identify which attention heads
carry retrieval-relevant information. Unlike Wu et al.'s behavioral
retrieval score, CRI measures whether restoring a head's clean activation
into a corrupted run actually recovers retrieval performance.

Algorithm per example:
  1. Clean forward:  model processes input with needle → high P(answer)
  2. Corrupted forward: model processes input without needle → low P(answer)
  3. Per-head patching: for each head h, run corrupted forward but replace
     head h's pre-o_proj activation with the clean value → measure recovery

  CRI(h) = E[ log P(answer | corrupted + patch(h)) - log P(answer | corrupted) ]

Uses standard HuggingFace transformers (not vLLM). Teacher-forced evaluation
(gold answer appended to input) for efficient single-pass logprob extraction.

Supports both NIAH and NoLiMa probing datasets via the shared datasets module.

Usage:
    # Quick test
    python -m locos.detectors.cri \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --dataset nolima --num-examples 5

    # Full CRI detection
    python -m locos.detectors.cri \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --dataset nolima --num-examples 200

    # With scramble corruption
    python -m locos.detectors.cri \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --dataset nolima --corruption scramble

Requires: GPU and the project runtime dependencies.
"""

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure repo root is on sys.path so `locos.*` imports work
# regardless of the working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from locos.utils.common import (
    extract_model_config,
    format_duration,
)
from locos.utils.common import load_checkpoint as _load_checkpoint_generic
from locos.utils.common import save_checkpoint as _save_checkpoint_generic
from locos.utils.model_utils import get_decoder_layers

console = Console()


# ---------------------------------------------------------------------------
# Activation capture & patching via o_proj hooks
# ---------------------------------------------------------------------------


class CRIHookManager:
    """Manages forward hooks on attention o_proj layers for CRI.

    Uses register_forward_pre_hook on each attention layer's o_proj to:
    - Capture mode: store the pre-o_proj activation per head
    - Patch mode: replace one head's activation with a stored clean value

    The input to o_proj has shape [batch, seq_len, num_heads * head_dim].
    We reshape to [batch, seq_len, num_heads, head_dim] for per-head access.
    """

    def __init__(self, model, num_layers: int, num_heads: int, head_dim: int):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.handles: list = []

        # State
        self.mode = "off"  # "capture", "patch", or "off"
        self.captured: dict[int, torch.Tensor] = {}  # layer_idx -> [batch, seq, heads, dim]
        self.patch_layer: int = -1
        self.patch_head: int = -1
        self.patch_activation: torch.Tensor | None = None

        # Find attention layers and register hooks
        self._attn_layers = self._find_attention_layers(model)
        assert (
            len(self._attn_layers) == num_layers
        ), f"Expected {num_layers} attention layers, found {len(self._attn_layers)}"

    def _find_attention_layers(self, model) -> list:
        """Find all attention modules in the model."""
        decoder_layers = get_decoder_layers(model)
        layers = []
        for layer in decoder_layers:
            assert hasattr(layer, "self_attn"), f"Decoder layer {type(layer).__name__} has no self_attn attribute"
            layers.append(layer.self_attn)
        return layers

    def install_hooks(self) -> None:
        """Register pre-hooks on all o_proj modules."""
        self.remove_hooks()
        for layer_idx, attn in enumerate(self._attn_layers):
            assert hasattr(attn, "o_proj"), f"Attention layer {layer_idx} has no o_proj attribute"
            handle = attn.o_proj.register_forward_pre_hook(self._make_hook(layer_idx))
            self.handles.append(handle)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _make_hook(self, layer_idx: int):
        """Create a pre-hook for the o_proj at the given layer.

        The hook receives (module, input) where input is a tuple.
        For nn.Linear, input[0] has shape [batch, seq_len, hidden_dim]
        where hidden_dim = num_heads * head_dim.

        Returns modified input tuple if patching, otherwise None (no modification).
        """

        def hook_fn(module, args):
            if self.mode == "off":
                return None

            hidden = args[0]  # [batch, seq, num_heads * head_dim]
            bsz, seq_len, _ = hidden.shape

            if self.mode == "capture":
                # Store per-head activations for this layer
                hidden_3d = hidden.view(bsz, seq_len, self.num_heads, self.head_dim)
                self.captured[layer_idx] = hidden_3d.detach().clone()
                return None  # Don't modify

            if self.mode == "patch" and layer_idx == self.patch_layer:
                # Replace one head's activation with the clean version.
                hidden_3d = hidden.view(bsz, seq_len, self.num_heads, self.head_dim).clone()
                assert self.patch_activation is not None
                clean_act = self.patch_activation  # [batch, clean_seq, heads, dim]
                # Activation patching requires exact positional alignment.
                # `build_cri_prompt` guarantees clean/corrupt sequences match
                # in length ("remove" corruption uses filler tokens sized to
                # the original needle; "scramble" preserves length trivially).
                # Fail loudly if the invariant ever breaks — a silent prefix-
                # only patch would give a CRI that misses the answer position
                # and reads ~zero for every head.
                assert clean_act.shape[1] == seq_len, (
                    f"Clean/corrupt sequence length mismatch in CRI hook at "
                    f"layer {layer_idx}: clean_seq={clean_act.shape[1]}, "
                    f"corrupt_seq={seq_len}. Check build_cri_prompt filler "
                    f"token padding."
                )
                hidden_3d[:, :, self.patch_head, :] = clean_act[:, :, self.patch_head, :]
                return (hidden_3d.reshape(bsz, seq_len, -1),)

            return None

        return hook_fn

    def set_capture_mode(self) -> None:
        """Switch to capture mode: store activations during forward pass."""
        self.mode = "capture"
        self.captured.clear()

    def set_patch_mode(self, layer_idx: int, head_idx: int) -> None:
        """Switch to patch mode: replace one head's activation during forward."""
        self.mode = "patch"
        self.patch_layer = layer_idx
        self.patch_head = head_idx
        assert layer_idx in self.captured, f"Layer {layer_idx} not captured. Run a capture pass first."
        self.patch_activation = self.captured[layer_idx]

    def set_off(self) -> None:
        """Disable hooks (pass-through)."""
        self.mode = "off"

    def clear_captured(self) -> None:
        """Free captured activation memory."""
        self.captured.clear()
        self.patch_activation = None


# ---------------------------------------------------------------------------
# CRI computation
# ---------------------------------------------------------------------------


def extract_answer_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    answer_start: int,
    answer_end: int,
) -> torch.Tensor:
    """Extract log-probabilities of the gold answer tokens.

    In teacher-forced evaluation, the input contains the answer tokens.
    The logits at position t predict token t+1. So for answer tokens at
    positions [answer_start, answer_end), we extract logprobs from
    logits at positions [answer_start - 1, answer_end - 1).

    Args:
        logits: [batch, seq_len, vocab_size]
        input_ids: [batch, seq_len]
        answer_start: Start position of answer tokens in input_ids.
        answer_end: End position (exclusive) of answer tokens.

    Returns:
        Tensor of per-token log-probabilities, shape [answer_end - answer_start].
    """
    assert answer_start >= 1, "answer_start must be >= 1 (need preceding logit)"
    assert answer_end > answer_start

    # Logits at positions [answer_start-1, answer_end-1) predict answer tokens
    pred_logits = logits[0, answer_start - 1 : answer_end - 1]  # [num_answer_tokens, vocab]
    target_ids = input_ids[0, answer_start:answer_end]  # [num_answer_tokens]

    log_probs = F.log_softmax(pred_logits, dim=-1)
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    return token_log_probs


def extract_first_token_logit_diff(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    answer_start: int,
    counterfactual_token_id: int | torch.Tensor,
) -> torch.Tensor:
    """Extract the IOI-style logit difference for the gold first answer token.

    Community standard for activation-patching metrics (Wang et al. 2023 IOI,
    Meng et al. 2022 ROME): report the difference between the gold logit and
    a chosen counterfactual-token logit so uniform logit-scale shifts cancel::

        logit_diff = logits[0, t-1, y_correct] - logits[0, t-1, y_counterfactual]

    The caller owns the counterfactual; in the CRI loop we derive it from
    the corrupt-baseline argmax (excluding y_correct), i.e. "what the model
    wrongly prefers without the needle". Returns a 0-d tensor so
    ``(patched - corrupt).mean()`` is a no-op and the downstream loop is
    shape-uniform with the ``answer_logprob`` path.
    """
    assert answer_start >= 1, "answer_start must be >= 1 (need preceding logit)"
    gold_first_token = input_ids[0, answer_start]
    gold_logit = logits[0, answer_start - 1, gold_first_token]
    cf_logit = logits[0, answer_start - 1, counterfactual_token_id]
    return gold_logit - cf_logit


def choose_counterfactual_token_id(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    answer_start: int,
) -> torch.Tensor:
    """Pick the counterfactual token for the logit-difference CRI metric.

    Returns the argmax of the baseline logits at ``answer_start - 1``,
    excluding the gold first answer token. This is the "what does the model
    incorrectly prefer without the needle" token — the natural counterfactual
    for retrieval patching, and the reason logit-difference is robust to
    uniform logit-scale shifts.
    """
    assert answer_start >= 1, "answer_start must be >= 1 (need preceding logit)"
    row = logits[0, answer_start - 1].clone()
    gold = int(input_ids[0, answer_start].item())
    row[gold] = float("-inf")
    return torch.argmax(row)


def _extract_metric(
    metric: str,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    answer_start: int,
    answer_end: int,
    counterfactual_token_id: int | torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch between the two CRI metrics on a single forward's logits."""
    if metric == "first_token_logit_diff":
        assert counterfactual_token_id is not None, "counterfactual_token_id is required for first_token_logit_diff"
        return extract_first_token_logit_diff(logits, input_ids, answer_start, counterfactual_token_id)
    if metric == "answer_logprob":
        return extract_answer_logprobs(logits, input_ids, answer_start, answer_end)
    raise ValueError(f"Unknown metric: {metric}")


def compute_cri_for_example(
    model,
    tokenizer,
    hook_manager: CRIHookManager,
    clean_ids: torch.Tensor,
    corrupt_ids: torch.Tensor,
    answer_start_clean: int,
    answer_end_clean: int,
    answer_start_corrupt: int,
    answer_end_corrupt: int,
    num_layers: int,
    num_heads: int,
    layer_by_layer: bool = False,
    metric: str = "first_token_logit_diff",
) -> dict[str, float]:
    """Compute CRI scores for all heads on one example.

    Args:
        clean_ids: Input IDs with needle present, [1, seq_len_clean].
        corrupt_ids: Input IDs with needle removed/scrambled, [1, seq_len_corrupt].
        answer_start_clean, answer_end_clean: Answer token positions in clean input.
        answer_start_corrupt, answer_end_corrupt: Answer token positions in corrupt input.
        layer_by_layer: If True, process one layer at a time to save memory.
        metric: Scoring metric. ``first_token_logit_diff`` (default) is the
            IOI-style logit-difference between gold and a counterfactual token
            chosen as the corrupt-baseline argmax (excluding gold).
            ``answer_logprob`` is the legacy mean answer-span logprob delta.

    Returns:
        Dict mapping "layer-head" keys to CRI scores (float).
    """
    # Activation patching requires exact positional alignment between clean
    # and corrupt passes. build_cri_prompt pads the "remove" corruption with
    # filler tokens matching the original needle token count precisely so this
    # holds. Fail loudly if it does not — silent prefix-only patching would
    # give a CRI that measures only part of the sequence and appear close to
    # zero for every head whose effect is at the answer position.
    assert clean_ids.shape == corrupt_ids.shape, (
        f"Clean and corrupt sequences must have identical shape for activation "
        f"patching alignment. Got clean={tuple(clean_ids.shape)}, "
        f"corrupt={tuple(corrupt_ids.shape)}."
    )
    assert answer_start_clean == answer_start_corrupt, (
        f"answer_start must match between clean and corrupt prompts. "
        f"clean={answer_start_clean}, corrupt={answer_start_corrupt}."
    )
    assert answer_end_clean == answer_end_corrupt, (
        f"answer_end must match between clean and corrupt prompts. "
        f"clean={answer_end_clean}, corrupt={answer_end_corrupt}."
    )

    device = next(model.parameters()).device

    clean_ids = clean_ids.to(device)
    corrupt_ids = corrupt_ids.to(device)

    with torch.inference_mode():
        # 1. Clean forward pass: capture all head activations. We do not need
        # the clean metric itself — CRI measures the *recovery* of the patched
        # corrupt run toward the clean distribution, not the clean-vs-corrupt
        # delta directly. Dropping the extraction saves a softmax per example.
        hook_manager.set_capture_mode()
        clean_out = model(clean_ids, output_attentions=False, return_dict=True)
        del clean_out

        # 2. Corrupted forward pass: baseline metric without needle.
        # For the logit-difference metric we also need to pick a counterfactual
        # token from the corrupt-baseline distribution (the strongest non-gold
        # competitor at the answer position).
        hook_manager.set_off()
        corrupt_out = model(corrupt_ids, output_attentions=False, return_dict=True)
        if metric == "first_token_logit_diff":
            counterfactual_token_id = choose_counterfactual_token_id(
                corrupt_out.logits, corrupt_ids, answer_start_corrupt
            )
        else:
            counterfactual_token_id = None
        corrupt_metric = _extract_metric(
            metric,
            corrupt_out.logits,
            corrupt_ids,
            answer_start_corrupt,
            answer_end_corrupt,
            counterfactual_token_id=counterfactual_token_id,
        )
        del corrupt_out

        # 3. Per-head patching — same counterfactual token is reused across
        # all patch passes so the logit-difference is measured against a fixed
        # reference and scale shifts from patching genuinely cancel.
        cri_scores: dict[str, float] = {}

        if layer_by_layer:
            # Process one layer at a time, free memory after each
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    hook_manager.set_patch_mode(layer_idx, head_idx)
                    patched_out = model(corrupt_ids, output_attentions=False, return_dict=True)
                    patched_metric = _extract_metric(
                        metric,
                        patched_out.logits,
                        corrupt_ids,
                        answer_start_corrupt,
                        answer_end_corrupt,
                        counterfactual_token_id=counterfactual_token_id,
                    )
                    del patched_out

                    # .mean() is a no-op on 0-d (first_token_logit_diff) tensors
                    # and averages over answer tokens for answer_logprob.
                    cri = (patched_metric - corrupt_metric).mean().item()
                    cri_scores[f"{layer_idx}-{head_idx}"] = cri

                # Free this layer's captured activation after processing all its heads
                if layer_idx in hook_manager.captured:
                    del hook_manager.captured[layer_idx]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        else:
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    hook_manager.set_patch_mode(layer_idx, head_idx)
                    patched_out = model(corrupt_ids, output_attentions=False, return_dict=True)
                    patched_metric = _extract_metric(
                        metric,
                        patched_out.logits,
                        corrupt_ids,
                        answer_start_corrupt,
                        answer_end_corrupt,
                        counterfactual_token_id=counterfactual_token_id,
                    )
                    del patched_out

                    cri = (patched_metric - corrupt_metric).mean().item()
                    cri_scores[f"{layer_idx}-{head_idx}"] = cri

    hook_manager.set_off()
    hook_manager.clear_captured()

    return cri_scores


# ---------------------------------------------------------------------------
# Prompt construction for CRI
# ---------------------------------------------------------------------------


def _get_filler_token_id(tokenizer) -> int:
    """Get a semantically neutral filler token ID (period).

    Used to pad the corrupted input to the same length as the clean input
    when needle is removed, preserving positional alignment for activation
    patching.
    """
    period_ids = tokenizer.encode(".", add_special_tokens=False)
    return period_ids[-1]  # Last token in case tokenizer prepends space


def build_cri_prompt(
    haystack_text: str,
    needle_text: str | None,
    question: str,
    answer_text: str,
    context_length: int,
    depth_percent: float,
    tokenizer,
    model_name: str = "",
    prompt_template: str | None = None,
    original_needle_text: str | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Build a teacher-forced prompt with gold answer appended.

    Args:
        needle_text: The needle to insert. None means insert filler tokens
            (corrupted version for CRI "remove" corruption).
        prompt_template: Full prompt template with {haystack} placeholder.
            If provided, the haystack (with needle) is filled into this template.
            If None, uses simple concatenation (context + question).
        original_needle_text: The original needle (before corruption). Required
            when needle_text is None, to know how many filler tokens to insert.

    Returns:
        (input_ids [1, seq_len], answer_start, answer_end) where
        answer_start/end are token positions of the gold answer.
    """
    from locos.utils.needle_utils import insert_needle

    if needle_text is not None:
        # Clean or scrambled: insert needle into haystack normally
        context_tokens, _, _ = insert_needle(
            haystack_text,
            needle_text,
            context_length,
            depth_percent,
            tokenizer,
            model_name=model_name,
        )
    else:
        # "Remove" corruption: insert filler tokens of the same length as
        # the original needle to preserve positional alignment. This ensures
        # clean and corrupted sequences have identical length, so activation
        # patching aligns positions exactly.
        assert (
            original_needle_text is not None
        ), "original_needle_text required when needle_text is None (remove corruption)"
        original_needle_tokens = tokenizer.encode(original_needle_text, add_special_tokens=False)
        filler_id = _get_filler_token_id(tokenizer)
        filler_text = ". " * (len(original_needle_tokens) // 2 + 1)
        # Truncate filler to exactly match original needle token count
        filler_tokens = tokenizer.encode(filler_text, add_special_tokens=False)
        filler_tokens = filler_tokens[: len(original_needle_tokens)]
        # Pad with the filler token if needed
        while len(filler_tokens) < len(original_needle_tokens):
            filler_tokens.append(filler_id)
        filler_as_text = tokenizer.decode(filler_tokens)
        context_tokens, _, _ = insert_needle(
            haystack_text,
            filler_as_text,
            context_length,
            depth_percent,
            tokenizer,
            model_name=model_name,
        )

    if prompt_template is not None and "{haystack}" in prompt_template:
        # NoLiMa: decode context back to text, fill into template, re-tokenize.
        # This preserves the intended prompt framing (instructions, etc.).
        haystack_with_needle = tokenizer.decode(context_tokens, skip_special_tokens=True)
        full_prompt = prompt_template.replace("{haystack}", haystack_with_needle)
        prefix_tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
    else:
        # NIAH: simple concatenation (context + question)
        question_tokens = tokenizer.encode(question, add_special_tokens=False)
        prefix_tokens = context_tokens + question_tokens

    # Append gold answer (teacher forcing)
    answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
    assert len(answer_tokens) >= 1, f"Answer '{answer_text}' encodes to empty tokens"

    full_tokens = prefix_tokens + answer_tokens

    # Add BOS if tokenizer expects it
    bos_offset = 0
    if tokenizer.bos_token_id is not None:
        test_encode = tokenizer.encode("test", add_special_tokens=True)
        if test_encode[0] == tokenizer.bos_token_id:
            full_tokens = [tokenizer.bos_token_id, *full_tokens]
            bos_offset = 1

    answer_start = len(prefix_tokens) + bos_offset
    answer_end = answer_start + len(answer_tokens)

    input_ids = torch.tensor([full_tokens])
    return input_ids, answer_start, answer_end


def build_cri_prompt_pair(
    haystack_text: str,
    needle_text: str,
    question: str,
    answer_text: str,
    context_length: int,
    depth_percent: float,
    tokenizer,
    corruption: str,
    corrupted_needle_text: str | None = None,
    model_name: str = "",
    prompt_template: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build a (clean, corrupt) prompt pair with guaranteed identical shape.

    Token-space composition approach (cf. Wang 2023 IOI, Meng 2022 ROME): we
    assemble the prompt by concatenating independently-tokenized pieces
    (template prefix, haystack+needle, template suffix, answer) so that the
    needle's position in the final sequence is *tracked exactly*, never
    searched for. This is strictly more robust than the previous
    decode/re-encode round-trip (which dropped ~50% of examples via shape
    drift) or a post-hoc find_needle_idx heuristic (which could miss the
    needle when BPE re-tokenizes with different leading-space boundaries).

    The corrupt prompt is produced by splicing a length-matched token span
    into the clean sequence at the needle position; prefix and suffix remain
    byte-identical, so activation-patching alignment is guaranteed.

    Args:
        needle_text: The original (clean) needle. Must not be None.
        corruption: ``"remove"`` overwrites the needle span with filler period
            tokens; ``"scramble"`` overwrites with the (padded/truncated)
            ``corrupted_needle_text`` tokens.
        corrupted_needle_text: Required for scramble corruption.

    Returns:
        (clean_ids [1, L], corrupt_ids [1, L], answer_start, answer_end).
    """
    from locos.utils.needle_utils import (
        build_period_token_positions,
        build_tracked_prompt,
        get_period_tokens,
    )

    assert needle_text is not None, "needle_text is required (the original clean needle)"
    assert corruption in ("remove", "scramble"), f"Unknown corruption: {corruption}"
    if corruption == "scramble":
        assert corrupted_needle_text is not None, "corrupted_needle_text required for scramble"

    # Pre-tokenize pieces and compute sentence-boundary positions for the
    # haystack (used by insert_needle_tokens via build_tracked_prompt).
    haystack_tokens = tokenizer.encode(haystack_text, add_special_tokens=False)
    needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    answer_tokens_list = tokenizer.encode(answer_text, add_special_tokens=False)
    assert len(answer_tokens_list) >= 1, f"Answer '{answer_text}' encodes to empty tokens"
    period_tokens = get_period_tokens(model_name, tokenizer)
    period_positions = build_period_token_positions(haystack_tokens, period_tokens)

    # Detect whether the tokenizer prepends BOS on add_special_tokens=True, to
    # match the old build_cri_prompt behaviour.
    add_bos = False
    if tokenizer.bos_token_id is not None:
        test_encode = tokenizer.encode("test", add_special_tokens=True)
        if test_encode and test_encode[0] == tokenizer.bos_token_id:
            add_bos = True

    clean_ids, needle_start, needle_end, answer_start, answer_end = build_tracked_prompt(
        haystack_tokens=haystack_tokens,
        needle_tokens=needle_tokens,
        context_length=context_length,
        depth_percent=depth_percent,
        period_positions=period_positions,
        tokenizer=tokenizer,
        prompt_template=prompt_template if prompt_template and "{haystack}" in prompt_template else None,
        question_tokens=question_tokens,
        answer_tokens=answer_tokens_list,
        add_bos=add_bos,
    )
    assert answer_start is not None and answer_end is not None
    needle_len = needle_end - needle_start

    # Build the corrupt span, clamped to the clean needle span length.
    if corruption == "remove":
        filler_id = _get_filler_token_id(tokenizer)
        corrupt_span: list[int] = [filler_id] * needle_len
    else:  # scramble
        scrambled_tokens = tokenizer.encode(corrupted_needle_text, add_special_tokens=False)
        if len(scrambled_tokens) >= needle_len:
            corrupt_span = scrambled_tokens[:needle_len]
        else:
            # NOTE: scrambled needle can tokenize shorter than the clean
            # needle (e.g. shorter alternate character name). Right-pad with
            # filler periods so lengths align. Reduces "corruption strength"
            # marginally but preserves the activation-patching invariant.
            # Padding preserves the activation-patching length invariant.
            filler_id = _get_filler_token_id(tokenizer)
            corrupt_span = scrambled_tokens + [filler_id] * (needle_len - len(scrambled_tokens))

    assert len(corrupt_span) == needle_len, "corrupt_span length invariant violated"

    corrupt_list = clean_ids[0].tolist()
    corrupt_list[needle_start:needle_end] = corrupt_span
    corrupt_ids = torch.tensor([corrupt_list])

    assert clean_ids.shape == corrupt_ids.shape, (
        f"Shape invariant violated after splice: clean={tuple(clean_ids.shape)}, "
        f"corrupt={tuple(corrupt_ids.shape)}."
    )

    return clean_ids, corrupt_ids, answer_start, answer_end


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------


def save_checkpoint(head_scores, completed_ids, checkpoint_path):
    """Save CRI progress for resuming (wraps generic checkpoint)."""
    _save_checkpoint_generic(
        {"head_scores": dict(head_scores), "completed_ids": completed_ids},
        checkpoint_path,
    )


def load_checkpoint(checkpoint_path):
    """Load CRI checkpoint (wraps generic checkpoint)."""
    data = _load_checkpoint_generic(checkpoint_path)
    if data is None:
        return defaultdict(list), []
    head_scores = defaultdict(list)
    for k, v in data["head_scores"].items():
        head_scores[k] = v
    return head_scores, data["completed_ids"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Detect retrieval heads via Causal Retrieval Importance (CRI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON path (default: retrieval_heads/<model>_cri.json)"
    )
    parser.add_argument("--dataset", type=str, default="nolima", choices=["niah", "nolima"], help="Probing dataset")
    parser.add_argument(
        "--haystack-dir", type=Path, default=Path("data/haystack_for_detect"), help="NIAH data directory"
    )
    parser.add_argument("--nolima-dir", type=Path, default=Path("data/nolima"), help="NoLiMa data directory")
    parser.add_argument(
        "--question-type",
        type=str,
        default="onehop",
        choices=["onehop", "twohop", "twohop2"],
        help="NoLiMa question type",
    )
    parser.add_argument("--nolima-variant", type=str, default="needle_set", choices=["needle_set", "needle_set_hard"])
    parser.add_argument(
        "--corruption",
        type=str,
        default="remove",
        choices=["remove", "scramble"],
        help="Corruption strategy: remove needle or scramble it",
    )
    parser.add_argument("--num-examples", type=int, default=200, help="Number of examples to use for CRI computation")
    parser.add_argument("--context-length", type=int, default=4000, help="Context length in tokens for CRI examples")
    parser.add_argument("--num-depths", type=int, default=5, help="Number of depth positions to sample")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--device-map", type=str, default="auto", choices=["auto", "balanced", "balanced_low_0", "sequential"]
    )
    parser.add_argument(
        "--layer-by-layer", action="store_true", help="Process one layer at a time (saves memory, slower)"
    )
    parser.add_argument(
        "--save-per-token", action="store_true", help="Save per-token CRI values to a separate JSONL file"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric",
        type=str,
        default="first_token_logit_diff",
        choices=["first_token_logit_diff", "answer_logprob"],
        help=(
            "Per-example causal metric. 'first_token_logit_diff' (default) is the "
            "IOI-style logit difference between the gold first answer token and a "
            "counterfactual (the strongest non-gold token in the corrupt baseline) — "
            "robust to uniform logit-scale shifts, matches community practice for "
            "activation-patching metrics (Wang 2023, Meng 2022). "
            "'answer_logprob' is the legacy mean log-prob change across the answer "
            "span; kept as a robustness variant."
        ),
    )
    args = parser.parse_args()

    # Resolve output path. Suffix the file by metric so the two variants can
    # coexist on disk without overwriting each other.
    # NOTE: for backward compatibility with the existing
    # answer_logprob CRI JSON on HF (aryopg/locos-results), the
    # answer_logprob variant still writes to the historical `{model}_cri.json`
    # path. The new first_token_logit_diff variant gets a
    # `_first_token_logit_diff` suffix. If we want to retroactively rename
    # the legacy file, flip this to tag both explicitly.
    model_short_name = args.model.split("/")[-1]
    if args.output is None:
        suffix = "" if args.metric == "answer_logprob" else "_first_token_logit_diff"
        output_path = Path("retrieval_heads") / f"{model_short_name}_cri{suffix}.json"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    console.rule("[bold]CRI: Causal Retrieval Importance Detection[/bold]")

    # Build dataset
    from locos.utils.datasets import build_niah_dataset, build_nolima_dataset

    depth_percents = np.round(np.linspace(0, 100, num=args.num_depths, endpoint=True)).astype(int).tolist()
    context_lengths = [args.context_length]

    if args.dataset == "niah":
        all_trials = build_niah_dataset(
            args.haystack_dir,
            context_lengths,
            depth_percents,
        )
    elif args.dataset == "nolima":
        all_trials = build_nolima_dataset(
            args.nolima_dir,
            context_lengths,
            depth_percents,
            question_type=args.question_type,
            variant=args.nolima_variant,
            corruption=args.corruption,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Stratified sample: balanced coverage across entries, tests, depths
    from locos.utils.datasets import stratified_sample

    trials = stratified_sample(all_trials, args.num_examples, seed=args.seed)

    config_table = Table(title="Configuration", show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="bold")
    config_table.add_column("Value")
    config_table.add_row("Model", args.model)
    config_table.add_row("Dataset", args.dataset)
    config_table.add_row("Corruption", args.corruption)
    config_table.add_row("Metric", args.metric)
    config_table.add_row("Examples", f"{len(trials)} (from {len(all_trials)} available)")
    config_table.add_row("Context length", str(args.context_length))
    config_table.add_row("Depth positions", str(args.num_depths))
    config_table.add_row("Layer-by-layer", str(args.layer_by_layer))
    config_table.add_row("Output", str(output_path))
    console.print(config_table)

    # Load model
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    console.print(f"\nLoading model ({args.dtype}) ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    ).eval()

    # Extract model dimensions
    num_layers, num_heads = extract_model_config(model)

    config = model.config
    text_config = getattr(config, "text_config", config)
    head_dim = getattr(text_config, "head_dim", None) or getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(text_config, "hidden_size", None) or getattr(config, "hidden_size", None)
        if hidden_size and num_heads:
            head_dim = hidden_size // num_heads
    assert head_dim is not None, "Could not determine head_dim"

    total_heads = num_layers * num_heads
    # NOTE: For GQA models, num_heads is the number of Q-heads, not KV-heads.
    # CRI patches at Q-head granularity (the attention output has num_heads
    # dimensions before o_proj), which is correct. But the interpretation is
    # that we measure each Q-head's contribution, not each KV-head group's.
    # This may miss heads that contribute primarily through shared KV.

    model_table = Table(title="Model", show_header=False, box=None, padding=(0, 2))
    model_table.add_column("Key", style="bold")
    model_table.add_column("Value")
    model_table.add_row("Architecture", f"{num_layers} layers × {num_heads} heads × {head_dim} dim")
    model_table.add_row("Total heads", str(total_heads))
    model_table.add_row("Passes per example", f"{total_heads + 2} (1 clean + 1 corrupt + {total_heads} patches)")
    est_hours = (total_heads + 2) * len(trials) * 0.05 / 3600  # rough: 50ms per pass
    model_table.add_row("Estimated time", f"~{est_hours:.1f} hours (rough)")
    console.print(model_table)
    console.print("[green]Model loaded[/green]\n")

    # Install hooks
    hook_manager = CRIHookManager(model, num_layers, num_heads, head_dim)
    hook_manager.install_hooks()
    console.print(f"Installed CRI hooks on {num_layers} attention layers\n")

    # Resume or fresh start
    if args.resume:
        head_scores, completed_ids = load_checkpoint(checkpoint_path)
        console.print(f"Resumed: {len(completed_ids)} examples already completed")
    else:
        head_scores = defaultdict(list)
        completed_ids = []

    completed_set = set(completed_ids)
    remaining_trials = [t for t in trials if t.trial_id not in completed_set]
    console.print(f"Examples: {len(trials)} total, {len(remaining_trials)} remaining\n")

    if not remaining_trials:
        console.print("[yellow]All examples already completed.[/yellow]")
    else:
        loop_start_time = time.time()
        processed = 0
        oom_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[status]}[/dim]"),
            console=console,
        ) as progress:
            ptask = progress.add_task(
                "Computing CRI",
                total=len(remaining_trials),
                status="",
            )

            for trial in remaining_trials:
                elapsed_s = time.time() - loop_start_time
                if processed > 0:
                    avg_s = elapsed_s / processed
                    eta_s = (len(remaining_trials) - processed) * avg_s
                    timing = f"elapsed={format_duration(elapsed_s)} eta={format_duration(eta_s)}"
                else:
                    timing = "elapsed=00:00 eta=estimating..."

                progress.update(
                    ptask,
                    status=f"{trial.trial_id[:40]} | {timing}",
                )

                try:
                    # Build (clean, corrupt) as a shape-aligned pair via
                    # token-space splice (IOI/ROME-style) — guarantees identical
                    # length without decode/re-encode round-trip drift.
                    clean_ids, corrupt_ids, ans_start_clean, ans_end_clean = build_cri_prompt_pair(
                        trial.haystack_text,
                        trial.needle_text,
                        trial.question,
                        trial.answer_text,
                        trial.context_length,
                        trial.depth_percent,
                        tokenizer,
                        corruption=args.corruption,
                        corrupted_needle_text=trial.corrupted_needle,
                        model_name=args.model,
                        prompt_template=trial.prompt_template,
                    )
                    ans_start_corrupt = ans_start_clean
                    ans_end_corrupt = ans_end_clean

                    # Compute CRI for all heads
                    cri = compute_cri_for_example(
                        model,
                        tokenizer,
                        hook_manager,
                        clean_ids,
                        corrupt_ids,
                        ans_start_clean,
                        ans_end_clean,
                        ans_start_corrupt,
                        ans_end_corrupt,
                        num_layers,
                        num_heads,
                        layer_by_layer=args.layer_by_layer,
                        metric=args.metric,
                    )

                    # Accumulate scores
                    for key, score in cri.items():
                        head_scores[key].append(score)

                except torch.cuda.OutOfMemoryError:
                    console.print(f"[red]OOM on {trial.trial_id}. Skipping.[/red]")
                    oom_count += 1
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    console.print(f"[red]Error on {trial.trial_id}: {e}. Skipping.[/red]")

                completed_ids.append(trial.trial_id)
                processed += 1
                progress.advance(ptask)

                # Checkpoint every 5 examples (CRI is expensive per example)
                if len(completed_ids) % 5 == 0:
                    save_checkpoint(head_scores, completed_ids, checkpoint_path)

                # Memory cleanup between examples
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        total_elapsed = time.time() - loop_start_time
        console.print(f"\n[green]Completed {processed} examples[/green]")
        if oom_count > 0:
            console.print(f"[yellow]OOM skips: {oom_count}[/yellow]")
        console.print(f"Total time: {format_duration(total_elapsed)}")

    # Remove hooks
    hook_manager.remove_hooks()

    # Save results with envelope format
    result = {
        "meta": {
            "method": "cri",
            "metric": args.metric,
            "dataset": args.dataset,
            "corruption": args.corruption,
            "num_examples": len(completed_ids),
            "context_length": args.context_length,
            "num_depths": args.num_depths,
            "model": args.model,
            "question_type": args.question_type if args.dataset == "nolima" else None,
            "layer_by_layer": args.layer_by_layer,
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "scores": {},
    }

    # Ensure all layer-head combos are present
    for layer in range(num_layers):
        for head in range(num_heads):
            key = f"{layer}-{head}"
            result["scores"][key] = head_scores.get(key, [])

    output_path.write_text(json.dumps(result, indent=2))
    console.print(f"\n[bold green]Saved CRI scores to {output_path}[/bold green]")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Print top CRI heads
    scored = [(key, float(np.mean(scores)) if scores else 0.0) for key, scores in result["scores"].items()]
    scored.sort(key=lambda x: x[1], reverse=True)

    table = Table(title="Top 20 CRI Heads")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Layer-Head")
    table.add_column("Mean CRI", justify="right")
    table.add_column("Examples", justify="right")

    for i, (key, mean_cri) in enumerate(scored[:20]):
        table.add_row(
            str(i + 1),
            key,
            f"{mean_cri:.4f}",
            str(len(result["scores"][key])),
        )
    console.print(table)

    # Validate output is loadable
    sys.path.insert(0, ".")
    from locos_eval.retrieval_heads import load_retrieval_heads

    heads = load_retrieval_heads(str(output_path), num_heads=10)
    console.print(f"\nValidation: load_retrieval_heads() returned {len(heads)} heads")
    console.print(f"Top 10: {heads}")


if __name__ == "__main__":
    main()
