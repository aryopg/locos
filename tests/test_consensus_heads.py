"""Unit tests for locos/analysis/consensus_heads.py.

Verifies the cross-method consensus extraction invariants:
- Intersection ⊆ each method's top-k.
- Emitted JSON uses the envelope format that nolima_ablation.py accepts.
- Empty intersection round-trips cleanly.
"""

from __future__ import annotations

import json

from locos.analysis import consensus_heads as ch
from locos.analysis.nolima_ablation import load_all_head_scores, select_heads


def _toy_scores(seed: int) -> dict[str, list[float]]:
    import random

    rng = random.Random(seed)
    return {f"{layer}-{head}": [rng.random()] for layer in range(3) for head in range(4)}


def test_top_k_keys_picks_highest_mean():
    scores = {"0-0": [1.0], "0-1": [0.5], "0-2": [0.9], "0-3": [0.1]}
    assert ch.top_k_keys(scores, 2) == {"0-0", "0-2"}


def test_intersection_is_subset_of_each_topk(tmp_path):
    a = _toy_scores(seed=1)
    b = _toy_scores(seed=2)
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(json.dumps({"scores": a}))
    b_path.write_text(json.dumps({"scores": b}))
    for k in (3, 6, 9):
        top_a = ch.top_k_keys(ch.load_scores(a_path), k)
        top_b = ch.top_k_keys(ch.load_scores(b_path), k)
        intersection = top_a & top_b
        assert intersection <= top_a
        assert intersection <= top_b


def test_emitted_json_is_readable_by_nolima_ablation(tmp_path):
    # Minimal fixture: two methods agree on one head.
    a = {"0-0": [1.0], "0-1": [0.9], "0-2": [0.1]}
    b = {"0-0": [0.95], "0-1": [0.05], "0-2": [0.85]}
    top_a = ch.top_k_keys(a, 2)  # {"0-0", "0-1"}
    top_b = ch.top_k_keys(b, 2)  # {"0-0", "0-2"}
    consensus = top_a & top_b  # {"0-0"}

    out_path = tmp_path / "consensus_k2.json"
    ch.write_head_set(consensus, out_path, meta={"set_kind": "consensus_intersection"})

    # nolima_ablation.load_all_head_scores should parse it and
    # select_heads("top-k", 1) must return exactly one layer-head tuple.
    all_scored = load_all_head_scores(out_path)
    assert len(all_scored) == 1
    heads = select_heads(all_scored, mode="top-k", value=1)
    assert heads == [(0, 0)]


def test_empty_intersection_round_trips(tmp_path):
    out_path = tmp_path / "empty.json"
    ch.write_head_set(set(), out_path, meta={"set_kind": "consensus_intersection"})
    data = json.loads(out_path.read_text())
    assert data["scores"] == {}
    assert data["meta"]["set_kind"] == "consensus_intersection"
    # An empty set emits an empty "scores" dict; load_all_head_scores should
    # return an empty list cleanly.
    assert load_all_head_scores(out_path) == []
