"""Tests for locos/plotting/heatmap_comparison.py."""

from __future__ import annotations

import numpy as np
import pytest

from locos.plotting import heatmap_comparison
from locos_eval.utils.plotting import LINE_WIDTH


def _build_figure():
    heatmaps = [
        ("Wu NiaH", np.linspace(-1.0, 1.0, 24 * 8).reshape(24, 8)),
        ("Logit Contrib", np.linspace(-0.5, 0.8, 40 * 8).reshape(40, 8)),
    ]
    fig, artists = heatmap_comparison.build_comparison_figure(heatmaps, model_name="Qwen3-8B", top_k=5)
    fig.canvas.draw()
    return fig, artists


class TestHeatmapComparisonLayout:
    def test_kde_axes_match_heatmap_height_and_keep_constant_gap(self):
        fig, artists = _build_figure()

        try:
            panel_gaps = []
            for artist in artists["panels"]:
                hm_bbox = artist["ax_hm"].get_position()
                kde_bbox = artist["ax_kde"].get_position()
                assert kde_bbox.y0 == pytest.approx(hm_bbox.y0, abs=1e-3)
                assert kde_bbox.y1 == pytest.approx(hm_bbox.y1, abs=1e-3)
                panel_gaps.append(kde_bbox.x0 - hm_bbox.x1)

            assert panel_gaps[0] == pytest.approx(panel_gaps[1], abs=1e-4)
        finally:
            fig.clf()

    def test_inter_panel_gap_is_kept_tight(self):
        fig, artists = _build_figure()

        try:
            first_cb = artists["panels"][0]["ax_cb"].get_position()
            second_hm = artists["panels"][1]["ax_hm"].get_position()
            assert second_hm.x0 - first_cb.x1 > 0.09
        finally:
            fig.clf()

    def test_titles_and_kde_styling_follow_requested_hierarchy(self):
        fig, artists = _build_figure()

        try:
            renderer = fig.canvas.get_renderer()
            model_title = artists["model_text"]
            method_title = artists["panels"][0]["method_text"]

            assert model_title.get_fontsize() == heatmap_comparison.MODEL_TITLE_FONT_SIZE
            assert method_title.get_fontsize() == heatmap_comparison.METHOD_TITLE_FONT_SIZE
            assert method_title.get_fontsize() < model_title.get_fontsize()

            model_bbox = model_title.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            method_bbox = method_title.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            assert 0 <= model_bbox.y0 - method_bbox.y1 < 0.055

            assert artists["panels"][0]["kde_line"].get_linewidth() == pytest.approx(LINE_WIDTH)
        finally:
            fig.clf()

    def test_method_title_is_centered_over_heatmap_and_kde_panel(self):
        fig, artists = _build_figure()

        try:
            renderer = fig.canvas.get_renderer()
            panel = artists["panels"][0]
            method_title = panel["method_text"]
            heatmap_bbox = panel["ax_hm"].get_position()
            kde_bbox = panel["ax_kde"].get_position()
            title_bbox = method_title.get_window_extent(renderer).transformed(fig.transFigure.inverted())

            title_center = (title_bbox.x0 + title_bbox.x1) / 2
            panel_center = (heatmap_bbox.x0 + kde_bbox.x1) / 2

            assert title_center == pytest.approx(panel_center, abs=0.01)
        finally:
            fig.clf()

    def test_colorbar_uses_labels_without_tick_marks(self):
        fig, artists = _build_figure()

        try:
            colorbar_axis = artists["panels"][0]["ax_cb"]

            for tick in colorbar_axis.yaxis.get_major_ticks():
                assert not tick.label1.get_visible()
                assert tick.label2.get_visible()
                assert tick.tick1line.get_markersize() == 0
                assert tick.tick2line.get_markersize() == 0
        finally:
            fig.clf()

    def test_heatmap_axis_ticks_are_close_to_plot(self):
        fig, artists = _build_figure()

        try:
            heatmap_axis = artists["panels"][0]["ax_hm"]

            assert heatmap_axis.xaxis.majorTicks[0].get_pad() == -1.5
            assert heatmap_axis.yaxis.majorTicks[0].get_pad() == -1.5
        finally:
            fig.clf()
