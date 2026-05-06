import torch

from locos_eval.retrieval_heads import group_heads_by_layer


class AblationState:
    """Mutable singleton-style state shared between the generation loop and patched attention layers.

    Lifecycle:
        1. set_retrieval_heads() — once, at init
        2. reset_kv_caches() — at the start of each new prompt
        3. set_kv_capacity(prompt_len + max_tokens) — sizes the pre-allocated buffers
        4. active = True — enters generation loop
        5. masked_pass_active = True for entire generation (prefill + decode)
        6. Only _masked_kv is populated; _base_kv unused
        7. active = False — exits generation loop

    KV cache layout:
        When set_kv_capacity(N) is called before the first update, each layer's
        cache is a single pre-allocated [N, kv_heads, head_dim] buffer that grows
        logically by tracking _*_seq_len[layer]. Decode-time slice-copies replace
        torch.cat, eliminating the 2× peak memory spike at every concatenation.

        When set_kv_capacity is *not* called (capacity == 0), update falls back
        to torch.cat — preserving the old behavior for tests and validation
        scripts that don't know the upper bound ahead of time.
    """

    def __init__(self) -> None:
        self.active: bool = False
        self.masked_pass_active: bool = False
        self._heads_by_layer: dict[int, list[int]] = {}
        # 0 means "capacity not set, use torch.cat fallback".
        self._kv_capacity: int = 0
        # Per-layer KV buffers and their populated lengths.
        self._base_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._base_seq_len: dict[int, int] = {}
        self._masked_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._masked_seq_len: dict[int, int] = {}

    def set_retrieval_heads(self, heads: list[tuple[int, int]]) -> None:
        """Configure which heads to mask. Call once at initialization."""
        assert isinstance(heads, list), f"Expected list of heads, got {type(heads)}"
        self._heads_by_layer = group_heads_by_layer(heads)

    def masked_heads_for_layer(self, layer_idx: int) -> list[int]:
        return self._heads_by_layer.get(layer_idx, [])

    def set_kv_capacity(self, capacity: int) -> None:
        """Configure the pre-allocated KV cache size for the upcoming generation.

        Call this between ``reset_kv_caches()`` and the first attention forward.
        ``capacity`` must be an upper bound on the total number of tokens the
        prompt + decode loop will consume (typically ``prompt_len + max_tokens``).

        Set to 0 to disable pre-allocation and fall back to ``torch.cat`` (used
        by tests/validation scripts that don't know the upper bound).
        """
        assert capacity >= 0, f"capacity must be non-negative, got {capacity}"
        self._kv_capacity = capacity

    # --- Base KV cache ---

    def get_base_kv(self, layer: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if layer not in self._base_kv:
            return (None, None)
        seq_len = self._base_seq_len.get(layer, 0)
        if seq_len == 0:
            return (None, None)
        buf_k, buf_v = self._base_kv[layer]
        if self._kv_capacity > 0:
            return buf_k[:seq_len], buf_v[:seq_len]
        # cat path: buffer is already exactly seq_len long
        return buf_k, buf_v

    def update_base_kv(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        assert k.ndim == 3 and v.ndim == 3, f"Expected 3D k/v, got k={k.shape}, v={v.shape}"
        assert k.shape == v.shape, f"k/v shape mismatch: {k.shape} vs {v.shape}"
        n = k.shape[0]
        cur = self._base_seq_len.get(layer, 0)

        if self._kv_capacity > 0:
            assert (
                cur + n <= self._kv_capacity
            ), f"base KV overflow on layer {layer}: {cur} + {n} > capacity {self._kv_capacity}"
            if layer not in self._base_kv:
                self._base_kv[layer] = (
                    torch.empty((self._kv_capacity, k.shape[1], k.shape[2]), dtype=k.dtype, device=k.device),
                    torch.empty((self._kv_capacity, v.shape[1], v.shape[2]), dtype=v.dtype, device=v.device),
                )
            buf_k, buf_v = self._base_kv[layer]
            buf_k[cur : cur + n].copy_(k)
            buf_v[cur : cur + n].copy_(v)
        else:
            prev = self._base_kv.get(layer)
            if prev is not None:
                prev_k, prev_v = prev
                assert prev_k.shape[1:] == k.shape[1:], f"KV cache shape mismatch: {prev_k.shape[1:]} vs {k.shape[1:]}"
                self._base_kv[layer] = (torch.cat([prev_k, k], dim=0), torch.cat([prev_v, v], dim=0))
            else:
                self._base_kv[layer] = (k, v)

        self._base_seq_len[layer] = cur + n

    # --- Masked KV cache ---

    def get_masked_kv(self, layer: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if layer not in self._masked_kv:
            return (None, None)
        seq_len = self._masked_seq_len.get(layer, 0)
        if seq_len == 0:
            return (None, None)
        buf_k, buf_v = self._masked_kv[layer]
        if self._kv_capacity > 0:
            return buf_k[:seq_len], buf_v[:seq_len]
        return buf_k, buf_v

    def update_masked_kv(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        assert k.ndim == 3 and v.ndim == 3, f"Expected 3D k/v, got k={k.shape}, v={v.shape}"
        assert k.shape == v.shape, f"k/v shape mismatch: {k.shape} vs {v.shape}"
        n = k.shape[0]
        cur = self._masked_seq_len.get(layer, 0)

        if self._kv_capacity > 0:
            assert (
                cur + n <= self._kv_capacity
            ), f"masked KV overflow on layer {layer}: {cur} + {n} > capacity {self._kv_capacity}"
            if layer not in self._masked_kv:
                self._masked_kv[layer] = (
                    torch.empty((self._kv_capacity, k.shape[1], k.shape[2]), dtype=k.dtype, device=k.device),
                    torch.empty((self._kv_capacity, v.shape[1], v.shape[2]), dtype=v.dtype, device=v.device),
                )
            buf_k, buf_v = self._masked_kv[layer]
            buf_k[cur : cur + n].copy_(k)
            buf_v[cur : cur + n].copy_(v)
        else:
            prev = self._masked_kv.get(layer)
            if prev is not None:
                prev_k, prev_v = prev
                assert prev_k.shape[1:] == k.shape[1:], f"KV cache shape mismatch: {prev_k.shape[1:]} vs {k.shape[1:]}"
                self._masked_kv[layer] = (torch.cat([prev_k, k], dim=0), torch.cat([prev_v, v], dim=0))
            else:
                self._masked_kv[layer] = (k, v)

        self._masked_seq_len[layer] = cur + n

    def copy_base_to_masked_kv(self) -> None:
        """Deep-copy the base KV cache into the masked KV cache.

        Call after shared prefill so both passes start from the same state,
        matching the reference implementation's `copy.deepcopy(past_key_values)`.
        """
        assert len(self._base_kv) > 0, "Cannot copy: base KV cache is empty"
        self._masked_kv.clear()
        self._masked_seq_len.clear()
        for layer, (buf_k, buf_v) in self._base_kv.items():
            seq_len = self._base_seq_len.get(layer, 0)
            if seq_len == 0:
                continue
            if self._kv_capacity > 0:
                # Allocate a fresh full-size buffer; copy only the populated slice
                # (avoids cloning the unpopulated tail).
                m_k = torch.empty_like(buf_k)
                m_v = torch.empty_like(buf_v)
                m_k[:seq_len].copy_(buf_k[:seq_len])
                m_v[:seq_len].copy_(buf_v[:seq_len])
            else:
                # cat path: clone the populated buffer (which is exactly seq_len long).
                m_k = buf_k.clone()
                m_v = buf_v.clone()
            self._masked_kv[layer] = (m_k, m_v)
            self._masked_seq_len[layer] = seq_len

    def reset_kv_caches(self) -> None:
        """Clear both KV caches. Call at start of each new prompt."""
        self._base_kv.clear()
        self._base_seq_len.clear()
        self._masked_kv.clear()
        self._masked_seq_len.clear()
