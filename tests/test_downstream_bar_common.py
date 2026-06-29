"""Tests for downstream bar plotting helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from locos.plotting import _downstream_bar_common as bars
from locos.plotting._paths import REPO_ROOT, default_downstream_results_root


def _write_results(variant_dir: Path, rows: list[dict], timestamp: str = "20250101") -> None:
    variant_dir.mkdir(parents=True)
    with open(variant_dir / f"results_{timestamp}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _domain(row: dict, task_dir: str) -> str | None:
    domain = row.get("metadata", {}).get("domain")
    return f"{task_dir}:{domain}" if domain is not None else None


def test_default_downstream_results_root_uses_repo_sibling_by_default(monkeypatch):
    monkeypatch.delenv("LOCOS_DOWNSTREAM_DIR", raising=False)

    assert default_downstream_results_root() == REPO_ROOT.parent / "locos-results" / "downstream_results"


def test_default_downstream_results_root_honors_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCOS_DOWNSTREAM_DIR", str(tmp_path))

    assert default_downstream_results_root() == tmp_path


def test_discover_long_form_aggregates_metrics_and_counts_each_task_dir(tmp_path):
    rows_a = [
        {"metadata": {"domain": "qa"}, "scores": {"accuracy": 1.0, "f1": 0.5}},
        {"metadata": {"domain": "qa"}, "scores": {"accuracy": 0.0, "f1": 1.0}},
        {"metadata": {"domain": "math"}, "scores": {"accuracy": 1.0, "f1": None}},
    ]
    rows_b = [
        {"metadata": {"domain": "qa"}, "scores": {"accuracy": 0.25, "f1": 0.25}},
        {"metadata": {"domain": "code"}, "scores": {"accuracy": 0.75, "f1": 0.75}},
    ]
    _write_results(tmp_path / "task_a" / "org_ModelA" / "greedy_s1", rows_a)
    _write_results(tmp_path / "task_b" / "org_ModelA" / "greedy_s1", rows_b)

    long_df, counts = bars.discover_long_form(tmp_path, ["task_a", "task_b"], _domain, ["accuracy", "f1"])

    assert counts == {"task_a:qa": 2, "task_a:math": 1, "task_b:qa": 1, "task_b:code": 1}
    row = long_df[
        (long_df["model"] == "ModelA")
        & (long_df["variant"] == "greedy")
        & (long_df["seed"] == 1)
        & (long_df["domain"] == "task_a:qa")
        & (long_df["metric"] == "accuracy")
    ].iloc[0]
    assert row["value"] == pytest.approx(0.5)


def test_discover_long_form_ignores_unknown_variants_and_missing_metrics(tmp_path):
    _write_results(
        tmp_path / "task" / "org_ModelA" / "custom_s1",
        [{"metadata": {"domain": "qa"}, "scores": {"accuracy": 1.0}}],
    )
    _write_results(
        tmp_path / "task" / "org_ModelA" / "greedy_s1",
        [{"metadata": {"domain": "qa"}, "scores": {"other": 1.0}}],
    )

    long_df, counts = bars.discover_long_form(tmp_path, ["task"], _domain, ["accuracy"])

    assert long_df.empty
    assert counts == {"task:qa": 1}


def test_overall_per_seed_macro_averages_domains_and_respects_exclusions(tmp_path):
    rows = [
        {"metadata": {"domain": "qa"}, "scores": {"accuracy": 1.0}},
        {"metadata": {"domain": "qa"}, "scores": {"accuracy": 0.0}},
        {"metadata": {"domain": "math"}, "scores": {"accuracy": 1.0}},
        {"metadata": {"domain": "skip"}, "scores": {"accuracy": 0.0}},
    ]
    _write_results(tmp_path / "task" / "org_ModelA" / "greedy_s1", rows)

    overall = bars.overall_per_seed(tmp_path, ["task"], _domain, ["accuracy"], excluded_domains={"task:skip"})

    assert overall.to_dict("records") == [
        {
            "model": "ModelA",
            "variant": "greedy",
            "seed": 1,
            "domain": "Overall",
            "metric": "accuracy",
            "value": pytest.approx(0.75),
        }
    ]


def test_aggregate_computes_seed_summary():
    long_df = pd.DataFrame.from_records(
        [
            {"model": "M", "variant": "greedy", "domain": "D", "metric": "accuracy", "value": 0.0},
            {"model": "M", "variant": "greedy", "domain": "D", "metric": "accuracy", "value": 1.0},
        ]
    )

    summary = bars.aggregate(long_df)

    assert summary.iloc[0]["mean"] == pytest.approx(0.5)
    assert summary.iloc[0]["std"] == pytest.approx(2**-0.5)
    assert summary.iloc[0]["n_seeds"] == 2


def test_render_multi_metric_domain_bars_writes_summary_and_delegates_plotting(tmp_path, monkeypatch):
    _write_results(
        tmp_path / "task" / "org_ModelA" / "greedy_s1",
        [{"metadata": {"domain": "qa"}, "scores": {"accuracy": 1.0}}],
    )
    out_dir = tmp_path / "figures"
    per_model_calls = []
    overall_calls = []

    def fake_make_per_model_figure(*args, **kwargs):
        per_model_calls.append((args, kwargs))

    def fake_make_overall_figure(*args, **kwargs):
        overall_calls.append((args, kwargs))

    monkeypatch.setattr(bars, "make_per_model_figure", fake_make_per_model_figure)
    monkeypatch.setattr(bars, "make_overall_figure", fake_make_overall_figure)

    bars.render_multi_metric_domain_bars(
        results_root=tmp_path,
        out_dir=out_dir,
        task_dirs=["task"],
        domain_fn=_domain,
        metrics=[("accuracy", "Accuracy")],
        task_name="Task",
        model_order=["ModelA"],
    )

    summary = pd.read_csv(out_dir / "summary.csv")
    assert set(summary["domain"]) == {"task:qa", "Overall"}
    assert len(per_model_calls) == 1
    assert len(overall_calls) == 2
