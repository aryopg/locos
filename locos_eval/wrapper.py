"""Ablation and greedy wrapper API for vLLM LLM instances.

Usage::

    from vllm import LLM
    from locos_eval import ablation

    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

    # Greedy baseline
    gen = ablation(llm, decoding="greedy")
    result = gen.generate("The capital of France is")

    # Ablation: zero retrieval heads throughout
    gen = ablation(llm, heads="retrieval_heads/Meta-Llama-3-8B-Instruct.json", decoding="ablation")
    result = gen.generate("The capital of France is")

    # Context manager for scoped patch/unpatch
    with ablation(llm, heads="retrieval_heads/...", decoding="ablation") as gen:
        result = gen.generate("The capital of France is")
    # attention unpatched on exit
"""

import os

import torch

from locos_eval.ablation import (
    calibrate_mean_q_activations,
    patch_model_for_ablation,
    select_replacements_for_masked_heads,
    unpatch_model_for_ablation,
)
from locos_eval.retrieval_heads import group_heads_by_layer, load_retrieval_heads
from locos_eval.rpc_ops import (
    rpc_calibrate_and_install_mean_ablation,
    rpc_install_ablation_hooks,
    rpc_uninstall_ablation_hooks,
)

ABLATION_MODES = ("zero", "mean")
DECODING_MODES = ("greedy", "ablation")

# ablation requires direct access to the model's nn.Module for manual forward
# passes. vLLM v0.18+ defaults to multiprocess mode, which puts the model in
# a separate process and prevents apply_model() from working with lambdas.
# This must be set before vLLM creates the engine.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


class GreedyWrapper:
    """Thin wrapper around vLLM's native generation for greedy baselines.

    Uses vLLM's optimised generation pipeline (paged attention, batching)
    instead of a manual autoregressive loop.  Works with any model
    architecture vLLM supports — no attention patching required.
    """

    # Accepts a list of prompts in a single ``generate`` call. The eval runner
    # checks this flag to decide whether to fan out per-sample or hand the
    # whole remaining batch to vLLM at once.
    supports_batch: bool = True

    def __init__(self, llm) -> None:
        self._llm = llm
        self._tokenizer = llm.get_tokenizer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self) -> None:
        """No-op — nothing to unpatch."""

    def generate(
        self,
        prompts: str | list[str],
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> str | list[str]:
        """Generate text using vLLM's native generation."""
        from vllm import SamplingParams

        single = isinstance(prompts, str)
        if single:
            prompts = [prompts]

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            top_k=top_k,
        )
        vllm_outputs = self._llm.generate(prompts, params, use_tqdm=False)
        results = [out.outputs[0].text for out in vllm_outputs]

        return results[0] if single else results


class AblationWrapper:
    """Single-GPU zero/mean-ablation via vLLM's native generation pipeline.

    Patches each supported attention layer's ``forward`` so that, immediately
    after the qkv split, the queries of the configured retrieval heads are
    overridden — either zeroed (``mode="zero"``) or replaced with a precomputed
    mean q activation (``mode="mean"``). The KV cache, scheduler, paged
    attention, and continuous batching all stay native — we only override the
    Q tensor, not the attention compute path.

    For ``mode="mean"`` the caller must supply either ``calibration_prompts``
    (in which case the wrapper runs an in-place calibration via
    :func:`calibrate_mean_q_activations`) or pre-computed ``mean_activations``.
    """

    supports_batch: bool = True

    def __init__(
        self,
        llm,
        heads: list[tuple[int, int]],
        *,
        mode: str = "zero",
        calibration_prompts: list[str] | None = None,
        mean_activations: dict[int, torch.Tensor] | None = None,
    ) -> None:
        assert isinstance(heads, list), f"heads must be a list, got {type(heads)}"
        if len(heads) == 0:
            raise ValueError("heads must be non-empty for ablation mode")
        if mode not in ABLATION_MODES:
            raise ValueError(f"Unknown ablation mode: {mode!r}. Choose from {ABLATION_MODES}")

        self._llm = llm
        self._tokenizer = llm.get_tokenizer()
        self._mode = mode

        # Direct access to the underlying nn.Module (TP=1 path).
        model_list = llm.apply_model(lambda m: m)
        assert len(model_list) == 1, f"Expected 1 model from apply_model, got {len(model_list)}"
        [self._model] = model_list

        heads_per_layer = group_heads_by_layer(heads)

        replacements_per_layer = None
        if mode == "mean":
            if mean_activations is None and calibration_prompts is None:
                raise ValueError("mode='mean' requires either calibration_prompts or pre-computed mean_activations")
            if mean_activations is not None and calibration_prompts is not None:
                raise ValueError("Pass either calibration_prompts OR mean_activations, not both")
            if mean_activations is None:
                model_dtype = next(self._model.parameters()).dtype
                layers_to_capture = sorted(heads_per_layer.keys())
                mean_activations = calibrate_mean_q_activations(
                    llm,
                    prompts=calibration_prompts,
                    layers_to_capture=layers_to_capture,
                    target_dtype=model_dtype,
                )
            replacements_per_layer = select_replacements_for_masked_heads(mean_activations, heads_per_layer)

        patch_model_for_ablation(
            self._model,
            heads_per_layer,
            replacements_per_layer=replacements_per_layer,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self) -> None:
        """Restore the original attention forwards."""
        unpatch_model_for_ablation(self._model)

    def generate(
        self,
        prompts: str | list[str],
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> str | list[str]:
        """Generate text under retrieval-head ablation via vLLM's native pipeline."""
        from vllm import SamplingParams

        single = isinstance(prompts, str)
        if single:
            prompts = [prompts]

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            top_k=top_k,
        )
        vllm_outputs = self._llm.generate(prompts, params, use_tqdm=False)
        results = [out.outputs[0].text for out in vllm_outputs]

        return results[0] if single else results


class AblationRPCWrapper:
    """Multi-GPU (TP>1) zero/mean-ablation orchestrator.

    Same public API as :class:`AblationWrapper`, but installs the per-layer
    forward patches on each worker via ``llm.collective_rpc``. The remap
    from global → local-rank head indices happens inside the worker (see
    :mod:`locos_eval.rpc_ops`).

    Mean ablation under TP requires calibration to happen on the workers (each
    rank only sees its own head slice). The orchestrator drives a calibration
    ``llm.generate`` call between hook install and finalize so that capture
    runs on the same vLLM scheduler/paged-attention path as the eval. See
    :func:`rpc_calibrate_and_install_mean_ablation` for the worker-side
    finalize step.

    Generation always goes through ``llm.generate`` directly: vLLM's TP
    routing, paged attention, and continuous batching are all preserved.
    """

    supports_batch: bool = True

    def __init__(
        self,
        llm,
        heads: list[tuple[int, int]],
        *,
        mode: str = "zero",
        calibration_prompts: list[str] | None = None,
    ) -> None:
        assert isinstance(heads, list), f"heads must be a list, got {type(heads)}"
        if len(heads) == 0:
            raise ValueError("heads must be non-empty for ablation mode")
        if mode not in ABLATION_MODES:
            raise ValueError(f"Unknown ablation mode: {mode!r}. Choose from {ABLATION_MODES}")
        if mode == "mean" and not calibration_prompts:
            # NOTE: we don't currently support pre-computed mean_activations
            # under TP>1 because the means must be sharded by local-rank head
            # index, which the orchestrator can't know without first inspecting
            # the worker. Add a from-disk loader (with rank-aware sharding)
            # only if calibration becomes prohibitively slow.
            raise ValueError("mode='mean' under TP>1 requires calibration_prompts")

        self._llm = llm
        self._tokenizer = llm.get_tokenizer()
        self._mode = mode

        if mode == "zero":
            llm.collective_rpc(rpc_install_ablation_hooks, kwargs=dict(heads=heads))
        else:
            # 1) Install capture hooks on each worker (no patching yet).
            from locos_eval.rpc_ops import rpc_install_q_capture_hooks

            llm.collective_rpc(rpc_install_q_capture_hooks, kwargs=dict(heads=heads))
            # 2) Drive calibration through the normal vLLM path so workers see
            #    a real prefill (this is when most tokens get captured).
            from vllm import SamplingParams

            calib_params = SamplingParams(max_tokens=1, temperature=0.0)
            llm.generate(calibration_prompts, calib_params, use_tqdm=False)
            # 3) On each worker: finalize means, swap capture → ablation hooks.
            llm.collective_rpc(rpc_calibrate_and_install_mean_ablation, kwargs=dict(heads=heads))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self) -> None:
        """Unpatch attention on all workers."""
        self._llm.collective_rpc(rpc_uninstall_ablation_hooks)

    def generate(
        self,
        prompts: str | list[str],
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> str | list[str]:
        """Generate text under retrieval-head ablation via vLLM's native pipeline."""
        from vllm import SamplingParams

        single = isinstance(prompts, str)
        if single:
            prompts = [prompts]

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            top_k=top_k,
        )
        vllm_outputs = self._llm.generate(prompts, params, use_tqdm=False)
        results = [out.outputs[0].text for out in vllm_outputs]

        return results[0] if single else results


def _get_tp_size(llm) -> int:
    """Detect tensor_parallel_size from a vLLM LLM instance."""
    try:
        return llm.llm_engine.model_executor.parallel_config.tensor_parallel_size
    except AttributeError:
        return 1


def ablation(
    llm,
    heads: str | list[tuple[int, int]] | None = None,
    num_heads: int | None = None,
    score_threshold: float = 0.4,
    decoding: str = "greedy",
    ablation_mode: str = "zero",
    calibration_prompts: list[str] | None = None,
    mean_activations: dict[int, torch.Tensor] | None = None,
) -> GreedyWrapper | AblationWrapper | AblationRPCWrapper:
    """Wrap a vLLM LLM instance for greedy or ablation decoding.

    Selection table:
        - ``decoding="greedy"`` → :class:`GreedyWrapper` (vLLM native)
        - ``decoding="ablation"`` + TP=1 → :class:`AblationWrapper` (vLLM native + q-zero patch)
        - ``decoding="ablation"`` + TP>1 → :class:`AblationRPCWrapper` (vLLM native + RPC q-zero patch)

    The LLM must be created with ``enforce_eager=True`` to disable
    torch.compile, which requires vLLM's ForwardContext (not available
    when calling the model directly).

    Args:
        llm: A vllm.LLM instance (created with enforce_eager=True).
        heads: Path to retrieval heads JSON file, or pre-loaded list of (layer, head) tuples.
        num_heads: If set, use exactly this many top retrieval heads.
        score_threshold: Keep heads with mean score >= threshold (default 0.4).
            Ignored when ``num_heads`` is set.
        decoding: Decoding strategy — ``"greedy"`` or ``"ablation"``.
        ablation_mode: ``"zero"`` (default) zeroes retrieval-head queries;
            ``"mean"`` replaces them with calibrated mean activations.
        calibration_prompts: Prompts used to compute mean query activations
            (only required when ``ablation_mode="mean"``).
        mean_activations: Pre-computed mean activations (alternative to
            ``calibration_prompts`` for ``ablation_mode="mean"``, TP=1 only).

    Returns:
        A wrapper instance that can be used directly or as a context manager.
    """
    if decoding not in DECODING_MODES:
        raise ValueError(f"Unknown decoding mode: {decoding!r}. Supported: {DECODING_MODES}")

    # Greedy mode: use vLLM's native generation — no patching needed,
    # works with any model architecture, and is faster (batched + paged attn).
    if decoding == "greedy":
        return GreedyWrapper(llm)

    if heads is None:
        raise ValueError("heads must be provided for ablation decoding")
    elif isinstance(heads, str):
        heads = load_retrieval_heads(heads, num_heads=num_heads, score_threshold=score_threshold)

    tp_size = _get_tp_size(llm)

    if tp_size > 1:
        return AblationRPCWrapper(
            llm=llm,
            heads=heads,
            mode=ablation_mode,
            calibration_prompts=calibration_prompts,
        )
    return AblationWrapper(
        llm=llm,
        heads=heads,
        mode=ablation_mode,
        calibration_prompts=calibration_prompts,
        mean_activations=mean_activations,
    )
