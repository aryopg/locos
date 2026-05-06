import json

import pytest

from locos_eval.retrieval_heads import generate_random_heads, group_heads_by_layer, load_retrieval_heads


def make_head_json(tmp_path, data):
    p = tmp_path / "TestModel.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_load_top_heads_sorted_by_mean_score(tmp_path):
    data = {
        "0-0": [0.1, 0.2, 0.3],  # mean=0.2
        "0-1": [0.9, 0.8, 0.7],  # mean=0.8  ← top
        "1-3": [0.4, 0.5, 0.6],  # mean=0.5  ← second
    }
    path = make_head_json(tmp_path, data)
    heads = load_retrieval_heads(path, num_heads=2)
    assert heads == [(0, 1), (1, 3)]


def test_load_all_heads_when_num_exceeds_available(tmp_path):
    data = {"2-5": [1.0], "3-7": [0.5]}
    path = make_head_json(tmp_path, data)
    heads = load_retrieval_heads(path, num_heads=100)
    # Must return both heads in descending score order
    assert heads == [(2, 5), (3, 7)]


def test_load_respects_num_heads_cutoff(tmp_path):
    data = {
        "0-0": [1.0],
        "0-1": [0.9],
        "1-2": [0.8],
        "2-3": [0.1],
    }
    path = make_head_json(tmp_path, data)
    heads = load_retrieval_heads(path, num_heads=2)
    assert len(heads) == 2
    # Must be the top-2 by score, not arbitrary
    assert heads == [(0, 0), (0, 1)]


def test_group_heads_by_layer_groups_correctly():
    heads = [(0, 1), (0, 3), (2, 5), (2, 7), (5, 0)]
    by_layer = group_heads_by_layer(heads)
    assert set(by_layer.keys()) == {0, 2, 5}
    assert by_layer[0] == [1, 3]
    assert by_layer[2] == [5, 7]
    assert by_layer[5] == [0]


def test_group_heads_by_layer_empty_input():
    assert group_heads_by_layer([]) == {}


def test_load_by_score_threshold(tmp_path):
    """Default mode: keep heads with mean score >= threshold."""
    data = {
        "0-0": [0.1, 0.2, 0.3],  # mean=0.2  ← below 0.4
        "0-1": [0.9, 0.8, 0.7],  # mean=0.8  ← above
        "1-3": [0.4, 0.5, 0.6],  # mean=0.5  ← above
        "2-0": [0.4, 0.4, 0.4],  # mean=0.4  ← exactly at threshold
    }
    path = make_head_json(tmp_path, data)
    heads = load_retrieval_heads(path, score_threshold=0.4)
    assert len(heads) == 3
    assert heads[0] == (0, 1)  # 0.8
    assert (2, 0) in heads  # 0.4 — included (>=)


def test_load_threshold_none_selected_raises(tmp_path):
    """If no heads meet the threshold, raise with a helpful message."""
    data = {"0-0": [0.1], "0-1": [0.2]}
    path = make_head_json(tmp_path, data)
    with pytest.raises(AssertionError, match="No retrieval heads selected"):
        load_retrieval_heads(path, score_threshold=0.9)


def test_load_num_heads_overrides_threshold(tmp_path):
    """When num_heads is set, threshold is ignored."""
    data = {
        "0-0": [0.1],  # mean=0.1 — below any reasonable threshold
        "0-1": [0.9],  # mean=0.9
    }
    path = make_head_json(tmp_path, data)
    heads = load_retrieval_heads(path, num_heads=2, score_threshold=0.5)
    assert len(heads) == 2  # both returned despite 0-0 being below 0.5


def test_load_handles_multidigit_layer_and_head(tmp_path):
    """Keys like "31-15" should parse correctly, not break on split."""
    data = {"31-15": [0.9], "2-0": [0.1]}
    path = make_head_json(tmp_path, data)
    heads = load_retrieval_heads(path, num_heads=2)
    assert heads[0] == (31, 15)
    assert heads[1] == (2, 0)


def test_load_envelope_format(tmp_path):
    """Envelope format with 'scores' key (used by CRI output) should work."""
    data = {
        "meta": {"method": "cri", "model": "test"},
        "scores": {
            "0-0": [0.9, 0.8],
            "1-2": [0.3, 0.4],
            "2-5": [0.6, 0.7],
        },
    }
    path = make_head_json(tmp_path, data)
    heads = load_retrieval_heads(path, num_heads=2)
    assert len(heads) == 2
    assert heads[0] == (0, 0)  # mean=0.85
    assert heads[1] == (2, 5)  # mean=0.65


def test_load_empty_scores_skipped(tmp_path):
    """Heads with empty score lists should be skipped (not crash on division)."""
    data = {
        "0-0": [0.9],
        "0-1": [],  # empty — should be skipped
        "1-0": [0.5],
    }
    path = make_head_json(tmp_path, data)
    heads = load_retrieval_heads(path, num_heads=10)
    assert len(heads) == 2
    assert (0, 1) not in heads


def test_generate_random_heads_count():
    """generate_random_heads returns exactly `count` heads."""
    heads = generate_random_heads(num_layers=32, num_heads=32, count=20, seed=42)
    assert len(heads) == 20


def test_generate_random_heads_deterministic():
    """Same seed produces same heads."""
    h1 = generate_random_heads(num_layers=32, num_heads=32, count=10, seed=123)
    h2 = generate_random_heads(num_layers=32, num_heads=32, count=10, seed=123)
    assert h1 == h2


def test_generate_random_heads_different_seeds():
    """Different seeds produce different heads."""
    h1 = generate_random_heads(num_layers=32, num_heads=32, count=10, seed=1)
    h2 = generate_random_heads(num_layers=32, num_heads=32, count=10, seed=2)
    assert h1 != h2


def test_generate_random_heads_valid_range():
    """All generated heads are within valid layer/head ranges."""
    heads = generate_random_heads(num_layers=8, num_heads=4, count=10, seed=42)
    for layer, head in heads:
        assert 0 <= layer < 8
        assert 0 <= head < 4


def test_generate_random_heads_no_duplicates():
    """No duplicate (layer, head) pairs."""
    heads = generate_random_heads(num_layers=32, num_heads=32, count=50, seed=42)
    assert len(heads) == len(set(heads))


def test_generate_random_heads_count_exceeds_total():
    """When count > num_layers * num_heads, return all possible heads."""
    heads = generate_random_heads(num_layers=2, num_heads=2, count=100, seed=42)
    assert len(heads) == 4  # 2*2 = 4 total possible


def test_generate_random_heads_rejects_zero_layers():
    """Zero layers should raise AssertionError."""
    with pytest.raises(AssertionError, match="num_layers must be positive"):
        generate_random_heads(num_layers=0, num_heads=4, count=5)


def test_generate_random_heads_rejects_negative_count():
    """Negative count should raise AssertionError."""
    with pytest.raises(AssertionError, match="count must be positive"):
        generate_random_heads(num_layers=4, num_heads=4, count=-1)
