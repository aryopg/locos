"""Unit tests for locos/datasets.py.

Tests dataset abstraction for both NIAH and NoLiMa probing datasets.
No GPU required.
"""

import json

import pytest

from locos.utils.datasets import (
    RetrievalTrial,
    _expand_nolima_template,
    _make_corrupted_needle,
    build_niah_dataset,
    build_nolima_dataset,
    load_niah_needles,
    load_nolima_needle_set,
    stratified_sample,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def niah_dir(tmp_path):
    """Create a minimal NIAH data directory."""
    needles = [
        {
            "needle": "The secret code is alpha-bravo-charlie.",
            "question": "What is the secret code?",
            "real_needle": "alpha-bravo-charlie",
        },
        {
            "needle": "The password is delta-echo-foxtrot.",
            "question": "What is the password?",
            "real_needle": "delta-echo-foxtrot",
        },
        {
            "needle": "The key is golf-hotel-india.",
            "question": "What is the key?",
            "real_needle": "golf-hotel-india",
        },
    ]
    (tmp_path / "needles.jsonl").write_text("\n".join(json.dumps(n) for n in needles))
    # Create haystack parts
    for i in range(1, 4):
        part_dir = tmp_path / f"part{i}"
        part_dir.mkdir()
        (part_dir / "p1.plain.txt").write_text("Filler text. " * 5000)

    return tmp_path


@pytest.fixture
def nolima_dir(tmp_path):
    """Create a minimal NoLiMa data directory."""
    needle_set = [
        {
            "id": "0401",
            "reasoning_type": "world_knowledge",
            "system_prompt": "",
            "task_template": (
                "You will answer a question based on the following book snippet:\n\n"
                "{haystack}\n\nQuestion: {question}\n\nReturn only the final answer."
            ),
            "needle": "Actually, {CHAR} lives next to {1}.",
            "questions": {
                "onehop": "Which character lives next to {1}?",
                "twohop": "Which character has been to {2}?",
            },
            "character_set": [
                "Yuki",
                "Stuart",
                "Katie",
                "Veronica",
                "Gary",
                "Megan",
                "Calvin",
                "Mandy",
                "Diana",
                "Caleb",
            ],
            "tests": {
                "T01_C02": {
                    "input_args": ["the Kiasma museum", "Helsinki"],
                },
                "T02_C02": {
                    "input_args": ["the Louvre", "Paris"],
                },
            },
        },
        {
            "id": "0402",
            "reasoning_type": "world_knowledge",
            "system_prompt": "",
            "task_template": ("Based on the book:\n\n{haystack}\n\nQuestion: {question}\n\nAnswer:"),
            "needle": "In 2013, {CHAR} visited {1}.",
            "questions": {
                "onehop": "Which character visited {1}?",
            },
            "character_set": [
                "Yuki",
                "Stuart",
                "Katie",
                "Veronica",
                "Gary",
                "Megan",
                "Calvin",
                "Mandy",
                "Diana",
                "Caleb",
            ],
            "tests": {
                "T03_C02": {
                    "input_args": ["the British Museum"],
                },
            },
        },
    ]
    (tmp_path / "needle_set.json").write_text(json.dumps(needle_set))

    # Create haystack files
    haystack_dir = tmp_path / "haystack" / "rand_shuffle"
    haystack_dir.mkdir(parents=True)
    for i in range(1, 6):
        (haystack_dir / f"rand_book_{i}.txt").write_text("Book text. " * 5000)

    return tmp_path


# ---------------------------------------------------------------------------
# Tests: Template expansion
# ---------------------------------------------------------------------------


def test_expand_nolima_template_basic():
    result = _expand_nolima_template(
        "Actually, {CHAR} lives next to {1}.",
        "Yuki",
        ["the Kiasma museum"],
    )
    assert result == "Actually, Yuki lives next to the Kiasma museum."


def test_expand_nolima_template_multiple_args():
    result = _expand_nolima_template(
        "In 2013, {CHAR} saw the {1} painting at {2} in {3}.",
        "Stuart",
        ["Mona Lisa", "the Louvre", "Paris"],
    )
    assert result == "In 2013, Stuart saw the Mona Lisa painting at the Louvre in Paris."


def test_expand_nolima_template_no_args():
    result = _expand_nolima_template("{CHAR} is here.", "Katie", [])
    assert result == "Katie is here."


# ---------------------------------------------------------------------------
# Tests: Corrupted needle
# ---------------------------------------------------------------------------


def test_make_corrupted_needle_remove():
    result = _make_corrupted_needle(
        "{CHAR} lives at {1}.",
        "Yuki",
        ["Yuki", "Stuart"],
        ["the museum"],
        corruption="remove",
    )
    assert result is None


def test_make_corrupted_needle_scramble():
    result = _make_corrupted_needle(
        "{CHAR} lives at {1}.",
        "Yuki",
        ["Yuki", "Stuart", "Katie"],
        ["the museum"],
        corruption="scramble",
    )
    assert result is not None
    assert "Yuki" not in result
    assert "the museum" in result
    # Should contain one of the other characters
    assert "Stuart" in result or "Katie" in result


# ---------------------------------------------------------------------------
# Tests: NIAH dataset
# ---------------------------------------------------------------------------


def test_load_niah_needles(niah_dir):
    needles = load_niah_needles(niah_dir)
    assert len(needles) == 3
    assert needles[0]["real_needle"] == "alpha-bravo-charlie"


def test_load_niah_needles_missing(tmp_path):
    with pytest.raises(AssertionError, match=r"needles\.jsonl not found"):
        load_niah_needles(tmp_path / "nonexistent")


def test_build_niah_dataset(niah_dir):
    trials = build_niah_dataset(
        niah_dir,
        context_lengths=[1000, 2000],
        depth_percents=[0, 50, 100],
    )
    # 3 needles × 2 lengths × 3 depths = 18
    assert len(trials) == 18

    trial = trials[0]
    assert isinstance(trial, RetrievalTrial)
    assert trial.dataset_name == "niah"
    assert trial.niah_needle_idx is not None
    assert trial.nolima_entry_id is None
    assert "Based on the content of the book" in trial.question
    assert trial.corrupted_needle is None  # default: remove


def test_niah_trial_ids_unique(niah_dir):
    trials = build_niah_dataset(
        niah_dir,
        context_lengths=[1000, 2000],
        depth_percents=[0, 50],
    )
    ids = [t.trial_id for t in trials]
    assert len(ids) == len(set(ids)), "Trial IDs must be unique"


# ---------------------------------------------------------------------------
# Tests: NoLiMa dataset
# ---------------------------------------------------------------------------


def test_load_nolima_needle_set(nolima_dir):
    entries = load_nolima_needle_set(nolima_dir)
    assert len(entries) == 2
    assert entries[0]["id"] == "0401"


def test_load_nolima_needle_set_missing(tmp_path):
    with pytest.raises(AssertionError, match=r"NoLiMa needle_set\.json not found"):
        load_nolima_needle_set(tmp_path / "nonexistent")


def test_build_nolima_dataset_onehop(nolima_dir):
    trials = build_nolima_dataset(
        nolima_dir,
        context_lengths=[1000],
        depth_percents=[50],
        question_type="onehop",
        max_characters_per_entry=1,
    )
    # Entry 0401 has 2 tests, entry 0402 has 1 test, both have onehop
    # With 1 char per entry: (2 + 1) tests × 1 char × 1 length × 1 depth = 3
    assert len(trials) == 3

    trial = trials[0]
    assert isinstance(trial, RetrievalTrial)
    assert trial.dataset_name == "nolima"
    assert trial.nolima_entry_id == "0401"
    assert trial.nolima_question_type == "onehop"
    # Answer should be a character name
    assert trial.answer_text in [
        "Yuki",
        "Stuart",
        "Katie",
        "Veronica",
        "Gary",
        "Megan",
        "Calvin",
        "Mandy",
        "Diana",
        "Caleb",
    ]


def test_build_nolima_dataset_twohop_skips_missing(nolima_dir):
    """Entries without the requested question type are skipped."""
    trials = build_nolima_dataset(
        nolima_dir,
        context_lengths=[1000],
        depth_percents=[50],
        question_type="twohop",
        max_characters_per_entry=1,
    )
    # Only entry 0401 has twohop (2 tests). Entry 0402 only has onehop.
    assert len(trials) == 2


def test_build_nolima_dataset_multiple_characters(nolima_dir):
    trials = build_nolima_dataset(
        nolima_dir,
        context_lengths=[1000],
        depth_percents=[50],
        question_type="onehop",
        max_characters_per_entry=3,
    )
    # (2 + 1) tests × 3 chars × 1 length × 1 depth = 9
    assert len(trials) == 9


def test_nolima_trial_ids_unique(nolima_dir):
    trials = build_nolima_dataset(
        nolima_dir,
        context_lengths=[1000, 2000],
        depth_percents=[0, 50],
        question_type="onehop",
        max_characters_per_entry=2,
    )
    ids = [t.trial_id for t in trials]
    assert len(ids) == len(set(ids)), "Trial IDs must be unique"


def test_nolima_needle_text_expanded(nolima_dir):
    trials = build_nolima_dataset(
        nolima_dir,
        context_lengths=[1000],
        depth_percents=[50],
        question_type="onehop",
        max_characters_per_entry=1,
    )
    trial = trials[0]
    # Needle should be fully expanded (no {CHAR} or {1} remaining)
    assert "{CHAR}" not in trial.needle_text
    assert "{1}" not in trial.needle_text


def test_nolima_corruption_remove(nolima_dir):
    trials = build_nolima_dataset(
        nolima_dir,
        context_lengths=[1000],
        depth_percents=[50],
        question_type="onehop",
        corruption="remove",
        max_characters_per_entry=1,
    )
    for trial in trials:
        assert trial.corrupted_needle is None


def test_nolima_corruption_scramble(nolima_dir):
    trials = build_nolima_dataset(
        nolima_dir,
        context_lengths=[1000],
        depth_percents=[50],
        question_type="onehop",
        corruption="scramble",
        max_characters_per_entry=1,
    )
    for trial in trials:
        assert trial.corrupted_needle is not None
        # Corrupted needle should not contain the original character
        assert trial.answer_text not in trial.corrupted_needle


# ---------------------------------------------------------------------------
# Tests: RetrievalTrial properties
# ---------------------------------------------------------------------------


def test_trial_dataset_name_niah():
    trial = RetrievalTrial(
        trial_id="niah_1000_50_0",
        haystack_text="text",
        needle_text="needle",
        answer_text="answer",
        question="question",
        context_length=1000,
        depth_percent=50,
        niah_needle_idx=0,
    )
    assert trial.dataset_name == "niah"


def test_trial_dataset_name_nolima():
    trial = RetrievalTrial(
        trial_id="nolima_0401_T01_Yuki_1000_50",
        haystack_text="text",
        needle_text="needle",
        answer_text="Yuki",
        question="question",
        context_length=1000,
        depth_percent=50,
        nolima_entry_id="0401",
        nolima_question_type="onehop",
    )
    assert trial.dataset_name == "nolima"


# ---------------------------------------------------------------------------
# Tests: prompt_template
# ---------------------------------------------------------------------------


def test_nolima_prompt_template_populated(nolima_dir):
    """NoLiMa trials have prompt_template with {haystack} placeholder."""
    trials = build_nolima_dataset(
        nolima_dir,
        context_lengths=[1000],
        depth_percents=[50],
        question_type="onehop",
        max_characters_per_entry=1,
    )
    for trial in trials:
        assert trial.prompt_template is not None
        assert "{haystack}" in trial.prompt_template
        # Question should be expanded (no {1} etc.)
        assert "{1}" not in trial.prompt_template
        assert "{CHAR}" not in trial.prompt_template
        # The question text should appear in the template
        assert trial.question in trial.prompt_template


def test_niah_prompt_template_is_none(niah_dir):
    """NIAH trials have no prompt_template (uses simple concatenation)."""
    trials = build_niah_dataset(
        niah_dir,
        context_lengths=[1000],
        depth_percents=[50],
    )
    for trial in trials:
        assert trial.prompt_template is None


# ---------------------------------------------------------------------------
# Tests: stratified_sample
# ---------------------------------------------------------------------------


def test_stratified_sample_returns_requested_count(nolima_dir):
    trials = build_nolima_dataset(
        nolima_dir,
        context_lengths=[1000, 2000],
        depth_percents=[0, 50, 100],
        question_type="onehop",
        max_characters_per_entry=3,
    )
    sampled = stratified_sample(trials, num_examples=5)
    assert len(sampled) == 5


def test_stratified_sample_all_if_fewer(nolima_dir):
    """If fewer trials than requested, return all."""
    trials = build_nolima_dataset(
        nolima_dir,
        context_lengths=[1000],
        depth_percents=[50],
        question_type="onehop",
        max_characters_per_entry=1,
    )
    sampled = stratified_sample(trials, num_examples=999)
    assert len(sampled) == len(trials)


def test_stratified_sample_covers_entries(nolima_dir):
    """Stratified sampling covers different entries."""
    trials = build_nolima_dataset(
        nolima_dir,
        context_lengths=[1000],
        depth_percents=[50],
        question_type="onehop",
        max_characters_per_entry=3,
    )
    # 9 trials total: entry 0401 (2 tests × 3 chars = 6), entry 0402 (1 test × 3 chars = 3)
    sampled = stratified_sample(trials, num_examples=4)
    entry_ids = {t.nolima_entry_id for t in sampled}
    # Should cover both entries
    assert len(entry_ids) == 2, f"Expected both entries, got {entry_ids}"


def test_stratified_sample_covers_depths(nolima_dir):
    """Stratified sampling covers different depths."""
    trials = build_nolima_dataset(
        nolima_dir,
        context_lengths=[1000],
        depth_percents=[0, 50, 100],
        question_type="onehop",
        max_characters_per_entry=1,
    )
    sampled = stratified_sample(trials, num_examples=6)
    depths = {t.depth_percent for t in sampled}
    # Should cover multiple depths
    assert len(depths) >= 2, f"Expected multiple depths, got {depths}"


def test_stratified_sample_deterministic(nolima_dir):
    """Same seed produces same sample."""
    trials = build_nolima_dataset(
        nolima_dir,
        context_lengths=[1000, 2000],
        depth_percents=[0, 50],
        question_type="onehop",
        max_characters_per_entry=2,
    )
    s1 = stratified_sample(trials, num_examples=5, seed=42)
    s2 = stratified_sample(trials, num_examples=5, seed=42)
    assert [t.trial_id for t in s1] == [t.trial_id for t in s2]
