"""Module-level RPC functions for multi-GPU ablation via vLLM collective_rpc.

Each function is dispatched to ALL worker processes via llm.collective_rpc().
Workers execute simultaneously so NCCL collectives work correctly.

The first argument ``self`` is the WorkerBase instance on each worker.
``self.get_model()`` returns the local nn.Module shard.

All functions must be cloudpickle-serializable (module-level, no closures
over unpicklable objects).
"""

import torch

from locos_eval.ablation import (
    finalize_q_capture_means,
    install_q_capture_hooks,
    patch_model_for_ablation,
    select_replacements_for_masked_heads,
    unpatch_model_for_ablation,
)
from locos_eval.attention import get_decoder_layers
from locos_eval.retrieval_heads import group_heads_by_layer


def _get_rank() -> int:
    """Return the current process rank, defaulting to 0 if not distributed."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _remap_heads_for_tp(
    heads: list[tuple[int, int]],
    num_heads_per_shard: int,
    tp_rank: int,
) -> list[tuple[int, int]]:
    """Remap global head indices to local shard indices for tensor parallelism.

    With TP, each rank owns a slice of the attention heads. TP rank ``r``
    owns global heads ``[r * num_heads_per_shard, (r+1) * num_heads_per_shard)``.
    This function filters to heads on this rank and converts to local indices.

    Args:
        heads: List of (layer, global_head_idx) tuples.
        num_heads_per_shard: Number of attention heads per TP shard.
        tp_rank: This worker's TP rank.

    Returns:
        List of (layer, local_head_idx) tuples for this shard only.
    """
    shard_start = tp_rank * num_heads_per_shard
    shard_end = shard_start + num_heads_per_shard
    local_heads = []
    for layer, head in heads:
        if shard_start <= head < shard_end:
            local_heads.append((layer, head - shard_start))
    return local_heads


# ---------------------------------------------------------------------------
# Native zero/mean-ablation hooks (TP>1)
#
# These patch each worker's attention modules to zero or replace retrieval-head
# queries while leaving vLLM's paged attention + scheduler untouched.
# Generation goes through ``llm.generate`` directly — no per-token RPC trips.
# ---------------------------------------------------------------------------


def rpc_install_ablation_hooks(self, *, heads):
    """Install zero-ablation forward patches on this worker's model shard.

    Remaps global ``(layer, head)`` indices to local-rank head indices, then
    calls :func:`locos_eval.ablation.patch_model_for_ablation`.

    Args:
        self: WorkerBase instance (injected by collective_rpc).
        heads: List of (layer_idx, global_head_idx) tuples.
    """
    assert isinstance(heads, list), f"Expected list of heads, got {type(heads)}"

    model = self.get_model()
    _, _, local_heads = _local_heads_for_worker(model, heads)
    heads_per_layer = group_heads_by_layer(local_heads)
    patch_model_for_ablation(model, heads_per_layer)

    # Stash so cleanup can find the same model instance.
    self._ablation_model = model


def rpc_uninstall_ablation_hooks(self):
    """Reverse :func:`rpc_install_ablation_hooks`. Idempotent.

    Also tears down any in-flight calibration capture state from
    :func:`rpc_install_q_capture_hooks` if calibration was started but never
    finalized.

    Args:
        self: WorkerBase instance.
    """
    if hasattr(self, "_ablation_model"):
        unpatch_model_for_ablation(self._ablation_model)
        del self._ablation_model
    # Defensive cleanup of orphaned capture hooks (e.g. exception during
    # mean-mode setup before finalize ran).
    if hasattr(self, "_ablation_capture_handles"):
        for h in self._ablation_capture_handles:
            h.remove()
        del self._ablation_capture_handles
    for attr in ("_ablation_capture_states", "_ablation_capture_heads", "_ablation_capture_model"):
        if hasattr(self, attr):
            delattr(self, attr)


def _local_heads_for_worker(model, heads):
    """Compute (num_heads_local, tp_rank, local_heads) for this worker."""
    first_attn = get_decoder_layers(model)[0].self_attn
    num_heads_local = first_attn.num_heads
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        tp_rank = get_tensor_model_parallel_rank()
    except (ImportError, AssertionError):
        tp_rank = _get_rank()
    local_heads = _remap_heads_for_tp(heads, num_heads_local, tp_rank)
    return num_heads_local, tp_rank, local_heads


def rpc_install_q_capture_hooks(self, *, heads):
    """Install qkv_proj capture hooks for layers that have masked heads.

    First step of the TP>1 mean-ablation flow:
        1. ``rpc_install_q_capture_hooks`` (this) — register hooks
        2. Orchestrator runs ``llm.generate(calibration_prompts, max_tokens=1)``
        3. ``rpc_calibrate_and_install_mean_ablation`` — finalize means + patch

    Stashed state on ``self``:
        ``_ablation_capture_handles`` — list of hook handles (for removal)
        ``_ablation_capture_states``  — ``layer_idx -> _QCaptureState``
        ``_ablation_capture_heads``   — local-rank head map for finalize
        ``_ablation_capture_model``   — model reference for finalize

    Args:
        self: WorkerBase instance.
        heads: List of (layer_idx, global_head_idx) tuples.
    """
    assert isinstance(heads, list), f"Expected list of heads, got {type(heads)}"
    assert not hasattr(self, "_ablation_capture_handles"), "rpc_install_q_capture_hooks called twice without finalize"

    model = self.get_model()
    _, _, local_heads = _local_heads_for_worker(model, heads)
    heads_per_layer = group_heads_by_layer(local_heads)

    # Only capture for layers that actually have masked heads on this rank.
    layers_to_capture = sorted(heads_per_layer.keys())
    handles, states = install_q_capture_hooks(model, layers_to_capture)

    self._ablation_capture_handles = handles
    self._ablation_capture_states = states
    self._ablation_capture_heads = heads_per_layer
    self._ablation_capture_model = model


def rpc_calibrate_and_install_mean_ablation(self, *, heads):
    """Finalize mean-q calibration and install ablation patches.

    Second step of the TP>1 mean-ablation flow. Reads accumulators populated
    by :func:`rpc_install_q_capture_hooks` (during the orchestrator's
    ``llm.generate`` calibration call), removes capture hooks, computes the
    per-(layer, head) mean tensor in the model dtype, slices it down to the
    local masked-head subset, and installs ablation patches.

    Args:
        self: WorkerBase instance.
        heads: Same global head list passed to ``rpc_install_q_capture_hooks``.
            Forwarded for parity / sanity check; the local remap is recovered
            from the stashed ``_ablation_capture_heads``.
    """
    del heads  # local remap was already computed at install time
    assert hasattr(
        self, "_ablation_capture_handles"
    ), "rpc_install_q_capture_hooks must be called before rpc_calibrate_and_install_mean_ablation"

    model = self._ablation_capture_model
    states = self._ablation_capture_states
    heads_per_layer = self._ablation_capture_heads

    # Remove capture hooks BEFORE installing ablation patches so they don't
    # both fire during subsequent eval forwards.
    for h in self._ablation_capture_handles:
        h.remove()

    model_dtype = next(model.parameters()).dtype
    means = finalize_q_capture_means(states, target_dtype=model_dtype)
    missing = [layer_idx for layer_idx in heads_per_layer if layer_idx not in means]
    if missing:
        raise RuntimeError(
            f"Mean calibration finished but layers {missing} received zero forwards on this rank. "
            f"This usually means calibration_prompts was empty or didn't reach the model."
        )

    replacements = select_replacements_for_masked_heads(means, heads_per_layer)
    patch_model_for_ablation(model, heads_per_layer, replacements_per_layer=replacements)

    self._ablation_model = model

    # Capture state no longer needed.
    del self._ablation_capture_handles
    del self._ablation_capture_states
    del self._ablation_capture_heads
    del self._ablation_capture_model
