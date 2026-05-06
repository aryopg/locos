"""Model interaction helpers for retrieval head detection.

Utilities for querying and manipulating HuggingFace model state:
device detection, attention implementation switching, and tokenizer
introspection.
"""

from __future__ import annotations

import re

import torch


def get_decoder_layers(model) -> list:
    """Find the decoder (transformer block) layers for any HuggingFace causal LM.

    Handles different model families:
    - Standard: model.model.layers (Llama, Qwen, Mistral, Gemma2, ...)
    - Composite/VLM: model.model.language_model.model.layers (Gemma3, ...)
    - GPT-2 style: model.transformer.h

    Returns:
        List of decoder layer modules (each containing self_attn, etc.).

    Raises:
        RuntimeError: If no decoder layers can be found.
    """
    # Standard HuggingFace layout: model.model.layers
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return list(inner.layers)

    # Composite / VLM layout (e.g. Gemma3ForConditionalGeneration):
    # model.model.language_model.layers          (Gemma3: language_model is TextModel directly)
    # model.model.language_model.model.layers    (other VLMs: language_model wraps a CausalLM)
    # model.language_model.model.layers          (language_model on top-level)
    for root in [inner, model]:
        if root is None:
            continue
        lang = getattr(root, "language_model", None)
        if lang is not None:
            # Direct layers on language_model (e.g. Gemma3TextModel)
            if hasattr(lang, "layers"):
                return list(lang.layers)
            # Wrapped: language_model.model.layers (e.g. SomeCausalLM.model)
            lang_inner = getattr(lang, "model", None)
            if lang_inner is not None and hasattr(lang_inner, "layers"):
                return list(lang_inner.layers)

    # GPT-2 style: model.transformer.h
    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "h"):
        return list(transformer.h)

    raise RuntimeError(
        "Could not find decoder layers. Tried model.model.layers, "
        "model.model.language_model.model.layers, model.transformer.h. "
        f"Top-level type: {type(model).__name__}, "
        f"inner type: {type(inner).__name__ if inner is not None else 'N/A'}"
    )


def get_input_device(model) -> torch.device:
    """Get the device for input tensors (first parameter's device)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def get_model_attn_impl(model) -> str | None:
    """Best-effort read of the model attention implementation setting."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    return getattr(cfg, "_attn_implementation", getattr(cfg, "attn_implementation", None))


def set_model_attn_impl(model, impl: str) -> bool:
    """Best-effort set of model attention implementation.

    Returns True if at least one relevant config field was updated.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return False

    updated = False
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = impl
        updated = True
    if hasattr(cfg, "attn_implementation"):
        cfg.attn_implementation = impl
        updated = True
    return updated


def get_stop_token_ids(tokenizer, model=None) -> set[int]:
    """Collect all stop token IDs from tokenizer and model config.

    Uses only authoritative sources — ``tokenizer.eos_token_id`` and
    ``model.generation_config.eos_token_id`` (which can be a list and
    is where model authors declare all their stop tokens, e.g. Gemma
    sets ``[1, 107]`` for both ``<eos>`` and ``<end_of_turn>``).
    """
    stop_ids: set[int] = set()

    # Tokenizer's primary EOS
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    # Model's generation_config.eos_token_id (authoritative, can be a list)
    if model is not None:
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None:
            eos = getattr(gen_cfg, "eos_token_id", None)
            if isinstance(eos, list):
                stop_ids.update(eos)
            elif eos is not None:
                stop_ids.add(eos)

    return stop_ids


def format_prompt_with_chat_template(
    tokenizer,
    prompt_text: str,
    *,
    disable_thinking: bool = True,
) -> str | None:
    """Apply the tokenizer's chat template, optionally disabling thinking mode.

    Some models (GPT-oss, Qwen3, …) require chat-formatted prompts to produce
    sensible output.  This wraps *prompt_text* in a single-turn user message
    and adds the generation prompt.  For reasoning models whose chat template
    supports ``enable_thinking``, thinking is disabled by default so the model
    gives a direct answer (important for ROUGE gating in detection).

    Returns the formatted prompt string, or ``None`` if the tokenizer has no
    chat template (caller should fall back to raw tokens).
    """
    if not (hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template):
        return None

    messages = [{"role": "user", "content": prompt_text}]
    chat_kwargs: dict = dict(tokenize=False, add_generation_prompt=True)

    if disable_thinking and "enable_thinking" in tokenizer.chat_template:
        chat_kwargs["enable_thinking"] = False

    return tokenizer.apply_chat_template(messages, **chat_kwargs)


def detect_thinking_tokens(tokenizer) -> tuple[int | None, int | None]:
    """Detect thinking start/end token IDs for reasoning models.

    Reasoning models (Qwen3, GPT-oss, etc.) wrap chain-of-thought output
    in special marker tokens (e.g. ``<think>…</think>``).  This function
    checks the tokenizer vocabulary for known marker patterns.

    Returns:
        ``(start_id, end_id)`` if both markers are found in the vocab,
        ``(None, None)`` otherwise.
    """
    vocab = tokenizer.get_vocab()

    candidates_start = ["<think>", "<|thinking|>"]
    candidates_end = ["</think>", "<|/thinking|>"]

    start_id = None
    for marker in candidates_start:
        if marker in vocab:
            start_id = vocab[marker]
            break

    end_id = None
    for marker in candidates_end:
        if marker in vocab:
            end_id = vocab[marker]
            break

    # Both markers must be present for detection to work.
    if start_id is not None and end_id is not None:
        return start_id, end_id
    return None, None


# Pattern covers <think>…</think> (Qwen3-style) and
# <|thinking|>…<|/thinking|> (alternative reasoning format).
_THINKING_RE = re.compile(
    r"<think>.*?</think>|<\|thinking\|>.*?<\|/thinking\|>",
    flags=re.DOTALL,
)


def strip_thinking_content(text: str) -> str:
    """Remove common thinking/reasoning blocks from generated text.

    Safety net for models that emit thinking tokens despite being instructed
    not to.  Returns stripped text (may be empty if the entire output was a
    thinking block).
    """
    return _THINKING_RE.sub("", text).strip()


def tokenizer_adds_bos(tokenizer) -> bool:
    """Return whether add_special_tokens prepends BOS for this tokenizer."""
    if tokenizer.bos_token_id is None:
        return False
    test_encode = tokenizer.encode("test", add_special_tokens=True)
    return bool(test_encode) and test_encode[0] == tokenizer.bos_token_id
