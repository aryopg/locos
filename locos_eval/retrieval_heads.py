import json
import random
from collections import defaultdict


def load_retrieval_heads(
    json_path: str,
    num_heads: int | None = None,
    score_threshold: float = 0.4,
) -> list[tuple[int, int]]:
    """Load pre-computed retrieval head rankings from a DeCoRe JSON file.

    Heads are selected by *score threshold* (default) or *top-N count*.
    When ``num_heads`` is provided it takes precedence over the threshold.

    Args:
        json_path: Path to JSON file mapping ``"layer-head"`` keys to
            per-sample score lists.
        num_heads: If set, return exactly this many top-ranked heads
            (ignoring the threshold).
        score_threshold: Keep heads whose mean retrieval score is
            ``>= score_threshold``.  Ignored when ``num_heads`` is set.

    Returns:
        List of (layer_idx, head_idx) tuples sorted by mean score descending.
    """
    with open(json_path) as f:
        data = json.load(f)

    # Support both flat format {"layer-head": [...]} and envelope format
    # {"meta": {...}, "scores": {"layer-head": [...]}} used by CRI output.
    if "scores" in data and isinstance(data["scores"], dict):
        scores_data = data["scores"]
    else:
        scores_data = data

    scored = [(key, float(sum(scores) / len(scores))) for key, scores in scores_data.items() if len(scores) > 0]
    scored.sort(key=lambda x: x[1], reverse=True)

    if num_heads is not None:
        selected = scored[:num_heads]
    else:
        selected = [(key, s) for key, s in scored if s >= score_threshold]

    assert len(selected) > 0, (
        f"No retrieval heads selected (threshold={score_threshold}, "
        f"top score={scored[0][1]:.3f}). Lower the threshold or use num_heads."
    )

    result = []
    for key, _ in selected:
        layer_str, head_str = key.split("-")
        result.append((int(layer_str), int(head_str)))
    return result


def generate_random_heads(
    num_layers: int,
    num_heads: int,
    count: int,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Generate random (layer, head) tuples for ablation controls.

    Args:
        num_layers: Number of transformer layers in the model.
        num_heads: Number of attention heads per layer.
        count: Number of random heads to select.
        seed: Random seed for reproducibility.

    Returns:
        List of (layer_idx, head_idx) tuples, length min(count, num_layers * num_heads).
    """
    assert num_layers > 0, f"num_layers must be positive, got {num_layers}"
    assert num_heads > 0, f"num_heads must be positive, got {num_heads}"
    assert count > 0, f"count must be positive, got {count}"

    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    rng = random.Random(seed)
    k = min(count, len(all_heads))
    return rng.sample(all_heads, k)


def group_heads_by_layer(heads: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Convert [(layer, head), ...] to {layer: [head, ...]} for O(1) per-layer lookup."""
    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, head in heads:
        by_layer[layer].append(head)
    return dict(by_layer)
