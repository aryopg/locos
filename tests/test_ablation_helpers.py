"""Tests for head selection and cache key helpers in nolima_ablation.py."""

from typing import ClassVar

import pytest

rouge_score = pytest.importorskip("rouge_score", reason="rouge-score not installed (eval dep)")

from locos.analysis.nolima_ablation import run_key, select_heads


class TestSelectHeads:
    SCORED: ClassVar[list[tuple[str, float]]] = [
        ("0-1", 0.9),
        ("1-3", 0.7),
        ("2-0", 0.5),
        ("3-2", 0.3),
        ("4-4", 0.1),
    ]

    def test_topk_selects_highest(self):
        heads = select_heads(self.SCORED, "top-k", 2)
        assert heads == [(0, 1), (1, 3)]

    def test_bottomk_selects_lowest(self):
        heads = select_heads(self.SCORED, "bottom-k", 2)
        assert heads == [(3, 2), (4, 4)]

    def test_bottomk_k_equals_total(self):
        heads = select_heads(self.SCORED, "bottom-k", 5)
        assert len(heads) == 5

    def test_bottomk_single(self):
        heads = select_heads(self.SCORED, "bottom-k", 1)
        assert heads == [(4, 4)]

    def test_bottomk_zero_returns_empty(self):
        heads = select_heads(self.SCORED, "bottom-k", 0)
        assert heads == []

    def test_threshold_unchanged(self):
        heads = select_heads(self.SCORED, "threshold", 0.5)
        assert heads == [(0, 1), (1, 3), (2, 0)]


class TestRunKey:
    def test_topk_key(self):
        key = run_key("top-k", 20, "TestModel", None, "mean", False)
        assert key == "TestModel__topk_20_mean"

    def test_bottomk_key(self):
        key = run_key("bottom-k", 20, "TestModel", None, "mean", False)
        assert key == "TestModel__bottomk_20_mean"

    def test_bottomk_key_differs_from_topk(self):
        top = run_key("top-k", 20, "TestModel", None, "mean", False)
        bottom = run_key("bottom-k", 20, "TestModel", None, "mean", False)
        assert top != bottom

    def test_baseline_key(self):
        key = run_key("baseline", 0, "TestModel", None, "zero", False)
        assert key == "TestModel__baseline"

    def test_limit_included(self):
        key = run_key("bottom-k", 10, "TestModel", 50, "mean", False)
        assert "_limit50" in key
