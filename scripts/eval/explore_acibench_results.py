#!/usr/bin/env python3
"""Streamlit app for exploring ACI-Bench evaluation results.

Usage:
    streamlit run scripts/eval/explore_acibench_results.py

Requires: pip install streamlit
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ACI-Bench Results Explorer",
    page_icon="\U0001fa7a",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
/* Chat bubbles */
.chat-container { margin-bottom: 1.5rem; }
.chat-bubble {
    padding: 0.75rem 1rem;
    border-radius: 12px;
    margin-bottom: 0.4rem;
    max-width: 85%;
    line-height: 1.5;
    font-size: 0.9rem;
    color: #212121;
}
.doctor-bubble {
    background-color: #e3f2fd;
    border: 1px solid #bbdefb;
    margin-right: auto;
}
.patient-bubble {
    background-color: #f3e5f5;
    border: 1px solid #e1bee7;
    margin-left: auto;
}
.speaker-label {
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
}
.doctor-label { color: #1565c0; }
.patient-label { color: #7b1fa2; }

/* Score cards */
.score-card {
    padding: 0.6rem 0.8rem;
    border-radius: 8px;
    text-align: center;
    border: 1px solid #e0e0e0;
}
.score-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #212121;
}
.score-label {
    font-size: 0.75rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Clinical note */
.clinical-note {
    background-color: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.9rem;
    line-height: 1.6;
    white-space: pre-wrap;
    color: #212121;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_results(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Dialogue parsing
# ---------------------------------------------------------------------------

_TURN_PATTERN = re.compile(r"\[(doctor|patient)\]\s*", re.IGNORECASE)


def parse_dialogue(text: str) -> list[dict[str, str]]:
    """Parse [doctor]/[patient] tagged dialogue into turns."""
    turns = []
    parts = _TURN_PATTERN.split(text)
    # parts: ['', 'doctor', ' text...', 'patient', ' text...', ...]
    i = 1
    while i < len(parts) - 1:
        speaker = parts[i].lower()
        content = parts[i + 1].strip()
        if content:
            turns.append({"speaker": speaker, "text": content})
        i += 2
    # Fallback: if no tags found, return as single block
    if not turns and text.strip():
        turns.append({"speaker": "unknown", "text": text.strip()})
    return turns


def render_dialogue(dialogue: str) -> None:
    """Render dialogue as styled chat bubbles."""
    turns = parse_dialogue(dialogue)
    html_parts = ['<div class="chat-container">']
    for turn in turns:
        speaker = turn["speaker"]
        text = turn["text"].replace("\n", "<br>")
        if speaker == "doctor":
            html_parts.append(
                f'<div class="chat-bubble doctor-bubble">'
                f'<div class="speaker-label doctor-label">Doctor</div>'
                f"{text}</div>"
            )
        elif speaker == "patient":
            html_parts.append(
                f'<div class="chat-bubble patient-bubble">'
                f'<div class="speaker-label patient-label">Patient</div>'
                f"{text}</div>"
            )
        else:
            html_parts.append(f'<div class="chat-bubble">{text}</div>')
    html_parts.append("</div>")
    st.html("".join(html_parts))


# ---------------------------------------------------------------------------
# Score rendering
# ---------------------------------------------------------------------------


def score_color(value: float, max_val: float = 1.0) -> str:
    """Return a hex background color based on score (green=good, red=bad)."""
    if value < 0:
        return "#f5f5f5"  # grey for missing
    ratio = value / max_val if max_val > 0 else 0
    if ratio >= 0.8:
        return "#e8f5e9"  # green
    elif ratio >= 0.5:
        return "#fff8e1"  # yellow
    else:
        return "#ffebee"  # red


def judge_score_color(value: float) -> str:
    """Color for judge scores (1-5 scale)."""
    if value < 0:
        return "#f5f5f5"
    return score_color(value, max_val=5.0)


def render_score_card(label: str, value: float, fmt: str = ".3f", max_val: float = 1.0) -> str:
    bg = score_color(value, max_val) if max_val <= 1 else judge_score_color(value)
    display = f"{value:{fmt}}" if value >= 0 else "N/A"
    return (
        f'<div class="score-card" style="background-color: {bg};">'
        f'<div class="score-value">{display}</div>'
        f'<div class="score-label">{label}</div>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def _parse_run_info(path: Path) -> dict[str, str]:
    """Extract run metadata from file path.

    Supports three layouts:
    - New: ``{root}/{task}/{model}/{variant}/results_{ts}.jsonl``
    - Old structured: ``{root}/{task}/{model}/{decoding}_{ts}.jsonl``
    - Legacy flat: ``{root}/aci_bench_{model}_{decoding}_{ts}.jsonl``
    """
    name = path.stem  # strip .jsonl
    parts = name.split("_")
    info: dict[str, str] = {"filename": str(path)}

    # New layout: parent = variant dir, grandparent = model dir
    variant_dir = path.parent.name
    model_dir = path.parent.parent.name if path.parent.parent != path.parent else ""
    task_dir = path.parent.parent.parent.name if path.parent.parent.parent != path.parent.parent else ""
    if task_dir and model_dir and name.startswith("results_"):
        info["variant"] = variant_dir
        first_part = variant_dir.split("_")[0]
        info["decoding"] = "ablation" if first_part in ("locos", "ablation") else first_part
        info["model"] = model_dir
        info["task"] = task_dir
        return info

    # Old structured layout: parent = model dir, grandparent = task dir
    if model_dir and parts[0] in ("locos", "ablation", "greedy"):
        info["decoding"] = "ablation" if parts[0] in ("locos", "ablation") else parts[0]
        info["variant"] = parts[0]
        info["model"] = path.parent.name
        info["task"] = model_dir if task_dir else ""
        return info

    # Legacy flat layout
    if "ablation" in parts or "locos" in parts:
        info["decoding"] = "ablation"
    elif "greedy" in parts:
        info["decoding"] = "greedy"
    else:
        info["decoding"] = "unknown"
    info["variant"] = info["decoding"]
    return info


def _file_label(path: Path) -> str:
    """Human-readable label for a results file."""
    info = _parse_run_info(path)
    variant = info.get("variant", info["decoding"])
    tag = "\U0001f7e2" if ("ablation" in variant or "locos" in variant) else "\u26aa"
    model = info.get("model", "")
    if model:
        return f"{tag} {model} / {variant} ({path.name})"
    return f"{tag} {path.name}"


def main():
    st.title("\U0001fa7a ACI-Bench Results Explorer")

    # Sidebar: file selection
    with st.sidebar:
        st.header("Load Results")
        results_dir = st.text_input("Results directory", value="eval_results")

        # Find result files, excluding checkpoint files (*_generations.jsonl)
        # Supports both structured (task/model/*.jsonl) and flat layouts
        if Path(results_dir).exists():
            all_files = sorted(Path(results_dir).rglob("*.jsonl"))
            jsonl_files = [
                f
                for f in all_files
                if "_generations" not in f.name and ("aci_bench" in f.name or "aci_bench" in str(f.parent.parent.name))
            ]
        else:
            jsonl_files = []

        if not jsonl_files:
            st.warning(f"No ACI-Bench result files found in `{results_dir}/`")
            st.stop()

        # Auto-detect if both ablation and greedy files exist
        st.header("Run Selection")
        decodings = {_parse_run_info(f)["decoding"] for f in jsonl_files}
        has_both = "ablation" in decodings and "greedy" in decodings
        compare_mode = st.checkbox("Compare two runs", value=has_both)

        if compare_mode:
            # Pre-select ablation for A and greedy for B when available
            default_a = next((i for i, f in enumerate(jsonl_files) if _parse_run_info(f)["decoding"] == "ablation"), 0)
            default_b = next((i for i, f in enumerate(jsonl_files) if _parse_run_info(f)["decoding"] == "greedy"), 0)
            col_a, col_b = st.columns(2)
            with col_a:
                file_a = st.selectbox("Run A", jsonl_files, index=default_a, format_func=_file_label, key="file_a")
            with col_b:
                file_b = st.selectbox("Run B", jsonl_files, index=default_b, format_func=_file_label, key="file_b")
            selected_files = [file_a, file_b]
            run_labels = [_parse_run_info(f)["decoding"].upper() for f in selected_files]
        else:
            selected_file = st.selectbox("Results file", jsonl_files, format_func=_file_label)
            selected_files = [selected_file]
            run_labels = [_parse_run_info(selected_file)["decoding"].upper()]

        all_records = {label: load_results(str(f)) for label, f in zip(run_labels, selected_files)}
        for label, recs in all_records.items():
            st.success(f"**{label}**: {len(recs)} samples")

        # Filters
        st.header("Filters")
        sort_metric = st.selectbox(
            "Sort by",
            [
                "sample_id",
                "rouge_l",
                "bertscore",
                "judge_normalized",
                "judge_completeness",
                "judge_accuracy",
                "judge_relevance",
            ],
        )
        sort_ascending = st.checkbox("Ascending", value=True)

        min_rouge = st.slider("Min ROUGE-L", 0.0, 1.0, 0.0, 0.05)
        min_bert = st.slider("Min BERTScore", 0.0, 1.0, 0.0, 0.05)

        # Comparison filter (only in compare mode)
        if compare_mode:
            st.header("Comparison Filter")
            compare_metric = st.selectbox(
                "Compare on metric",
                ["judge_normalized", "rouge_l", "bertscore", "judge_completeness", "judge_accuracy", "judge_relevance"],
                key="compare_metric",
            )
            compare_filter = st.radio(
                "Show samples where Run A vs Run B",
                ["All", "A > B", "A < B", "A = B"],
                horizontal=True,
                key="compare_filter",
            )
        else:
            compare_metric = None
            compare_filter = "All"

    # Use first run as primary for display
    primary_label = run_labels[0]
    records = all_records[primary_label]
    secondary = all_records.get(run_labels[1]) if compare_mode else None

    # Build lookup for secondary by sample_id
    secondary_by_id = {}
    if secondary:
        for r in secondary:
            secondary_by_id[r["sample_id"]] = r

    # Apply filters
    filtered = [
        r for r in records if r["scores"].get("rouge_l", 0) >= min_rouge and r["scores"].get("bertscore", 0) >= min_bert
    ]

    # Apply comparison filter
    if compare_mode and compare_filter != "All" and compare_metric:

        def _passes_compare(r):
            sec = secondary_by_id.get(r["sample_id"])
            if sec is None:
                return False
            a = r["scores"].get(compare_metric, -1)
            b = sec["scores"].get(compare_metric, -1)
            if a < 0 or b < 0:
                return False
            if compare_filter == "A > B":
                return a > b
            elif compare_filter == "A < B":
                return a < b
            else:  # A = B
                return abs(a - b) < 1e-6
            return True

        filtered = [r for r in filtered if _passes_compare(r)]

    # Sort
    if sort_metric == "sample_id":
        filtered.sort(key=lambda r: r["sample_id"], reverse=not sort_ascending)
    else:
        filtered.sort(
            key=lambda r: r["scores"].get(sort_metric, -1),
            reverse=not sort_ascending,
        )

    st.caption(f"Showing {len(filtered)} / {len(records)} samples")

    # Aggregate stats — show per-run if comparing
    if filtered:
        st.markdown("### Summary")
        if compare_mode and secondary:
            for label, recs in all_records.items():
                st.markdown(f"**{label}**")
                metric_cols = st.columns(6)
                for i, (metric, max_v) in enumerate(
                    [
                        ("rouge_l", 1.0),
                        ("bertscore", 1.0),
                        ("judge_normalized", 1.0),
                        ("judge_completeness", 5.0),
                        ("judge_accuracy", 5.0),
                        ("judge_relevance", 5.0),
                    ]
                ):
                    vals = [r["scores"].get(metric, -1) for r in recs if r["scores"].get(metric, -1) >= 0]
                    mean_val = sum(vals) / len(vals) if vals else -1
                    fmt = ".3f" if max_v <= 1 else ".1f"
                    metric_cols[i].markdown(
                        render_score_card(metric.replace("judge_", "").replace("_", " "), mean_val, fmt, max_v),
                        unsafe_allow_html=True,
                    )
        else:
            metric_cols = st.columns(6)
            for i, (metric, max_v) in enumerate(
                [
                    ("rouge_l", 1.0),
                    ("bertscore", 1.0),
                    ("judge_normalized", 1.0),
                    ("judge_completeness", 5.0),
                    ("judge_accuracy", 5.0),
                    ("judge_relevance", 5.0),
                ]
            ):
                vals = [r["scores"].get(metric, -1) for r in filtered if r["scores"].get(metric, -1) >= 0]
                mean_val = sum(vals) / len(vals) if vals else -1
                fmt = ".3f" if max_v <= 1 else ".1f"
                metric_cols[i].markdown(
                    render_score_card(metric.replace("judge_", "").replace("_", " "), mean_val, fmt, max_v),
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # Per-sample display
    for record in filtered:
        scores = record["scores"]
        metadata = record.get("metadata", {})
        sec_record = secondary_by_id.get(record["sample_id"]) if compare_mode else None

        # Build expander title with both runs' key scores in compare mode
        if sec_record:
            sec_scores = sec_record["scores"]
            expander_title = (
                f"Sample {record['sample_id']}  |  "
                f"ROUGE-L: {scores.get('rouge_l', 0):.3f} vs {sec_scores.get('rouge_l', 0):.3f}  "
                f"Judge: {scores.get('judge_normalized', -1):.3f} vs {sec_scores.get('judge_normalized', -1):.3f}"
            )
        else:
            expander_title = (
                f"Sample {record['sample_id']}  |  "
                f"ROUGE-L: {scores.get('rouge_l', 0):.3f}  "
                f"BERTScore: {scores.get('bertscore', 0):.3f}  "
                f"Judge: {scores.get('judge_normalized', -1):.3f}"
            )

        with st.expander(expander_title, expanded=False):
            # Score cards — show both runs side-by-side in compare mode
            _metrics = [
                ("rouge_l", "ROUGE-L", 1.0),
                ("bertscore", "BERTScore", 1.0),
                ("judge_normalized", "Judge (norm)", 1.0),
                ("judge_completeness", "Completeness", 5.0),
                ("judge_accuracy", "Accuracy", 5.0),
                ("judge_relevance", "Relevance", 5.0),
            ]
            if sec_record:
                sec_scores = sec_record["scores"]
                st.markdown(f"**{primary_label}**")
                cols_a = st.columns(6)
                for i, (metric, label, max_v) in enumerate(_metrics):
                    fmt = ".3f" if max_v <= 1 else ".0f"
                    cols_a[i].markdown(
                        render_score_card(label, scores.get(metric, -1), fmt, max_v),
                        unsafe_allow_html=True,
                    )
                st.markdown(f"**{run_labels[1]}**")
                cols_b = st.columns(6)
                for i, (metric, label, max_v) in enumerate(_metrics):
                    fmt = ".3f" if max_v <= 1 else ".0f"
                    cols_b[i].markdown(
                        render_score_card(label, sec_scores.get(metric, -1), fmt, max_v),
                        unsafe_allow_html=True,
                    )
            else:
                cols = st.columns(6)
                for i, (metric, label, max_v) in enumerate(_metrics):
                    fmt = ".3f" if max_v <= 1 else ".0f"
                    cols[i].markdown(
                        render_score_card(label, scores.get(metric, -1), fmt, max_v),
                        unsafe_allow_html=True,
                    )

            st.markdown("")

            # Dialogue (collapsed by default)
            dialogue = metadata.get("dialogue", "")
            if dialogue:
                with st.expander("Dialogue", expanded=False):
                    render_dialogue(dialogue)

            # Reference note (collapsed by default)
            with st.expander("Reference Note", expanded=False):
                st.markdown(
                    f'<div class="clinical-note">{record.get("target", "")}</div>',
                    unsafe_allow_html=True,
                )

            # Generated notes (always visible for easy comparison)
            if compare_mode and record["sample_id"] in secondary_by_id:
                sec_record = secondary_by_id[record["sample_id"]]
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"#### {primary_label}")
                    st.markdown(
                        f'<div class="clinical-note">{record.get("output", "")}</div>',
                        unsafe_allow_html=True,
                    )
                with col_b:
                    st.markdown(f"#### {run_labels[1]}")
                    st.markdown(
                        f'<div class="clinical-note">{sec_record.get("output", "")}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(f"#### Generated Note ({primary_label})")
                st.markdown(
                    f'<div class="clinical-note">{record.get("output", "")}</div>',
                    unsafe_allow_html=True,
                )

            # Judge explanations
            if compare_mode and sec_record:
                explanations_a = metadata.get("judge_explanations", {})
                explanations_b = sec_record.get("metadata", {}).get("judge_explanations", {})
                has_a = explanations_a and any(v for v in explanations_a.values())
                has_b = explanations_b and any(v for v in explanations_b.values())
                if has_a or has_b:
                    st.markdown("#### Judge Explanations")
                    col_ja, col_jb = st.columns(2)
                    with col_ja:
                        st.markdown(f"**{primary_label}**")
                        if has_a:
                            for axis, explanation in explanations_a.items():
                                if explanation:
                                    score_val = scores.get(f"judge_{axis}", -1)
                                    st.markdown(f"**{axis.capitalize()}** ({score_val:.0f}/5): {explanation}")
                        else:
                            st.caption("No judge explanations")
                    with col_jb:
                        st.markdown(f"**{run_labels[1]}**")
                        if has_b:
                            sec_scores = sec_record["scores"]
                            for axis, explanation in explanations_b.items():
                                if explanation:
                                    score_val = sec_scores.get(f"judge_{axis}", -1)
                                    st.markdown(f"**{axis.capitalize()}** ({score_val:.0f}/5): {explanation}")
                        else:
                            st.caption("No judge explanations")
            else:
                explanations = metadata.get("judge_explanations", {})
                if explanations and any(v for v in explanations.values()):
                    st.markdown("#### Judge Explanations")
                    for axis, explanation in explanations.items():
                        if explanation:
                            score_val = scores.get(f"judge_{axis}", -1)
                            st.markdown(f"**{axis.capitalize()}** ({score_val:.0f}/5): {explanation}")


if __name__ == "__main__":
    main()
