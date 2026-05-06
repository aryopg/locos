"""Tests for locos/plotting/dissociation_score.py."""

from __future__ import annotations

import pytest

from locos.plotting import dissociation_score
from locos.plotting.dissociation_score import DSStats, compute_ds_stats, make_dual_axis_plot


def test_compute_ds_stats_bootstraps_derived_ds_from_retrieval_and_parametric_trials():
    stats = compute_ds_stats(
        nolima_runs={5: 0.5},
        nolima_baseline=1.0,
        parametric_runs={5: {"accuracy": 0.75}},
        parametric_baseline={"accuracy": 1.0},
        nolima_trial_values={5: [0.0, 0.0, 1.0, 1.0]},
        nolima_baseline_trials=[1.0, 1.0, 1.0, 1.0],
        parametric_trial_values={5: [1.0, 1.0, 1.0, 0.0]},
        parametric_baseline_trials=[1.0, 1.0, 1.0, 1.0],
        bootstrap_samples=500,
        bootstrap_seed=0,
    )

    assert stats.ks == [5]
    assert stats.ds == pytest.approx([0.25])
    assert stats.ci_lows[0] < stats.ds[0] < stats.ci_highs[0]


def test_compute_ds_stats_omits_ci_when_trial_values_are_unavailable():
    stats = compute_ds_stats(
        nolima_runs={5: 0.5},
        nolima_baseline=1.0,
        parametric_runs={5: {"accuracy": 0.75}},
        parametric_baseline={"accuracy": 1.0},
        bootstrap_samples=500,
    )

    assert stats.ks == [5]
    assert stats.ds == pytest.approx([0.25])
    assert stats.ci_lows == [0.25]
    assert stats.ci_highs == [0.25]


def test_dual_axis_plot_shades_ds_ci_band(monkeypatch, tmp_path):
    captured = {}

    def fake_save_figure(fig, path, keep_title=False):
        captured["fig"] = fig

    monkeypatch.setattr(dissociation_score, "save_figure", fake_save_figure)

    make_dual_axis_plot(
        method_data={
            "LOCOS": DSStats(
                ks=[1, 5],
                ds=[0.1, 0.3],
                ci_lows=[0.0, 0.2],
                ci_highs=[0.2, 0.4],
            )
        },
        parametric_caches=[("LOCOS", {1: {"accuracy": 0.95}, 5: {"accuracy": 0.9}}, {"accuracy": 1.0})],
        out_path=tmp_path / "dual.svg",
    )

    fig = captured["fig"]
    try:
        ax_right = fig.axes[1]
        assert len(ax_right.collections) >= 1
    finally:
        fig.clf()
