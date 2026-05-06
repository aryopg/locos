"""Shared dataset abstraction for retrieval head detection.

Provides a unified RetrievalTrial dataclass and dataset builders for both
NIAH (Wu et al. 2024) and NoLiMa (Adobe Research, ICML 2025) probing datasets.

Both the Wu et al. behavioral scorer (detect_retrieval_heads.py) and the
CRI causal scorer (detect_cri.py) consume the same RetrievalTrial format.
"""

from __future__ import annotations

import json
import random
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RetrievalTrial:
    """One trial for retrieval head detection.

    Both NIAH and NoLiMa trials share this format. The detection scripts
    consume these uniformly.
    """

    trial_id: str
    """Unique key for checkpoint/resume (e.g., 'niah_1000_50_0' or 'nolima_0401_T17_Yuki_2000_25')."""

    haystack_text: str
    """Background text (the 'haystack') before needle insertion."""

    needle_text: str
    """The needle sentence to insert into the haystack."""

    answer_text: str
    """Gold answer string (for ROUGE gating in Wu et al. / logprob target in CRI)."""

    question: str
    """Question text to append after the context.

    For NIAH: 'Based on the content of the book, Question: ... Answer:'
    For NoLiMa: the expanded question (e.g., 'Which character lives next to the Kiasma museum?').
    """

    context_length: int
    """Target context length in tokens (including needle)."""

    depth_percent: float
    """Needle insertion position (0-100%) through the haystack."""

    # --- Optional fields (all have defaults) ---

    prompt_template: str | None = None
    """Full prompt template with {haystack} placeholder, filled at insertion time.

    For NoLiMa: the task_template with {question} already expanded but {haystack}
    left as a placeholder (e.g., 'You will answer ... {haystack} ... Question: Which
    character lives next to the Kiasma museum? ...').
    For NIAH: None (NIAH uses a fixed prompt format in the detection script).
    """

    corrupted_needle: str | None = None
    """For CRI: replacement needle text for the corrupted run.
    None means 'remove the needle entirely' (default corruption).
    A string means 'insert this instead' (e.g., character-swapped needle)."""

    # NoLiMa-specific metadata (optional, for analysis)
    nolima_question_type: str | None = None
    """'onehop', 'twohop', or 'twohop2' (NoLiMa only)."""

    nolima_entry_id: str | None = None
    """NoLiMa needle_set entry ID (e.g., '0401')."""

    # NIAH-specific metadata (optional, for analysis)
    niah_needle_idx: int | None = None
    """Index into the NIAH needles list (0, 1, or 2)."""

    @property
    def dataset_name(self) -> str:
        if self.nolima_entry_id is not None:
            return "nolima"
        return "niah"


# ---------------------------------------------------------------------------
# NIAH dataset builder
# ---------------------------------------------------------------------------


def load_niah_needles(haystack_dir: Path) -> list[dict]:
    """Load needle/question/answer triples from needles.jsonl."""
    needles_path = haystack_dir / "needles.jsonl"
    assert needles_path.exists(), (
        f"needles.jsonl not found at {needles_path}. " f"Run: python locos/download_haystack_data.py"
    )
    lines = [json.loads(line) for line in needles_path.read_text().strip().splitlines()]
    assert len(lines) >= 1, "needles.jsonl is empty"
    for line in lines:
        assert (
            "needle" in line and "question" in line and "real_needle" in line
        ), f"Each line must have 'needle', 'question', 'real_needle' keys, got: {list(line.keys())}"
    return lines


def load_niah_haystack_texts(haystack_dir: Path, max_tokens: int) -> list[str]:
    """Load per-needle haystack texts (part1/part2/part3)."""
    texts = []
    for part_idx in range(1, 4):
        plain_path = haystack_dir / f"part{part_idx}" / "p1.plain.txt"
        raw_path = haystack_dir / f"part{part_idx}" / "p1.txt"

        if plain_path.exists():
            text = plain_path.read_text()
        elif raw_path.exists():
            data = json.loads(raw_path.read_text())
            text = "\n\n".join(r["row"]["text"] for r in data["rows"] if r["row"].get("text"))
        else:
            raise AssertionError(
                f"Haystack part{part_idx} not found at {haystack_dir}. " f"Run: python locos/download_haystack_data.py"
            )

        while len(text.split()) < max_tokens:
            text += "\n\n" + text
        texts.append(text)

    return texts


def build_niah_dataset(
    haystack_dir: Path,
    context_lengths: list[int],
    depth_percents: list[float],
    max_tokens: int | None = None,
) -> list[RetrievalTrial]:
    """Build NIAH trials matching Wu et al.'s grid.

    Args:
        haystack_dir: Directory with needles.jsonl and part1/part2/part3.
        context_lengths: List of context lengths in tokens.
        depth_percents: List of depth percentages (0-100).
        max_tokens: Max token count for haystack expansion (default: max of context_lengths).

    Returns:
        List of RetrievalTrial objects.
    """
    if max_tokens is None:
        max_tokens = max(context_lengths) if context_lengths else 50000

    needles = load_niah_needles(haystack_dir)
    haystack_texts = load_niah_haystack_texts(haystack_dir, max_tokens)
    assert len(haystack_texts) == len(needles), f"Expected {len(needles)} haystack parts, got {len(haystack_texts)}"

    trials = []
    for ctx_len in context_lengths:
        for depth in depth_percents:
            for needle_idx, needle_data in enumerate(needles):
                trial_id = f"niah_{ctx_len}_{int(depth)}_{needle_idx}"
                question = f"Based on the content of the book, Question: " f"{needle_data['question']}\nAnswer:"
                trials.append(
                    RetrievalTrial(
                        trial_id=trial_id,
                        haystack_text=haystack_texts[needle_idx],
                        needle_text=needle_data["needle"],
                        answer_text=needle_data["real_needle"],
                        question=question,
                        context_length=ctx_len,
                        depth_percent=depth,
                        corrupted_needle=None,  # CRI: remove needle
                        niah_needle_idx=needle_idx,
                    )
                )
    return trials


# ---------------------------------------------------------------------------
# NoLiMa dataset builder
# ---------------------------------------------------------------------------


def _expand_nolima_template(template: str, char_name: str, input_args: list[str]) -> str:
    """Expand a NoLiMa template with character name and numbered args.

    Templates use {CHAR} for character name and {1}, {2}, {3}, {4} for
    1-indexed arguments from input_args.
    """
    result = template.replace("{CHAR}", char_name)
    for i, arg in enumerate(input_args, start=1):
        result = result.replace(f"{{{i}}}", arg)
    return result


def load_nolima_needle_set(nolima_dir: Path, variant: str = "needle_set") -> list[dict]:
    """Load NoLiMa needle set JSON.

    Args:
        nolima_dir: Directory containing needle_set.json etc.
        variant: Which needle set to load ('needle_set', 'needle_set_hard').
    """
    path = nolima_dir / f"{variant}.json"
    assert path.exists(), (
        f"NoLiMa {variant}.json not found at {path}. " f"Run: python locos/download_haystack_data.py --dataset nolima"
    )
    entries = json.loads(path.read_text())
    assert isinstance(entries, list), f"Expected list, got {type(entries)}"
    assert len(entries) >= 1, f"{variant}.json is empty"

    for entry in entries:
        required = {"id", "needle", "questions", "character_set", "tests", "task_template"}
        missing = required - set(entry.keys())
        assert not missing, f"Entry {entry.get('id', '?')} missing keys: {missing}"

    return entries


def load_nolima_haystack_texts(nolima_dir: Path, max_tokens: int) -> list[str]:
    """Load NoLiMa haystack book texts from rand_shuffle/.

    Returns list of 5 book texts (rand_book_1..5).
    """
    haystack_dir = nolima_dir / "haystack" / "rand_shuffle"
    texts = []
    for i in range(1, 6):
        path = haystack_dir / f"rand_book_{i}.txt"
        if not path.exists():
            # Try without rand_shuffle subdirectory
            path = nolima_dir / "haystack" / f"rand_book_{i}.txt"
        assert path.exists(), (
            f"NoLiMa haystack rand_book_{i}.txt not found at {haystack_dir}. "
            f"Run: python locos/download_haystack_data.py --dataset nolima"
        )
        text = path.read_text()
        while len(text.split()) < max_tokens:
            text += "\n\n" + text
        texts.append(text)
    return texts


def _make_corrupted_needle(
    needle_template: str,
    original_char: str,
    character_set: list[str],
    input_args: list[str],
    corruption: str = "scramble",
) -> str | None:
    """Create a corrupted needle for CRI.

    Args:
        corruption: 'remove' returns None (delete needle entirely),
                    'scramble' swaps the character name with another.
    """
    if corruption == "remove":
        return None

    # Scramble: pick a different character
    alternatives = [c for c in character_set if c != original_char]
    if not alternatives:
        return None  # fallback to removal if only one character
    alt_char = random.choice(alternatives)
    return _expand_nolima_template(needle_template, alt_char, input_args)


def build_nolima_dataset(
    nolima_dir: Path,
    context_lengths: list[int],
    depth_percents: list[float],
    question_type: str = "onehop",
    variant: str = "needle_set",
    max_tokens: int | None = None,
    corruption: str = "remove",
    max_characters_per_entry: int | None = None,
    seed: int = 42,
) -> list[RetrievalTrial]:
    """Build NoLiMa trials for retrieval head detection.

    Args:
        nolima_dir: Directory with needle_set.json and haystack/.
        context_lengths: List of context lengths in tokens.
        depth_percents: List of depth percentages (0-100).
        question_type: 'onehop', 'twohop', or 'twohop2'.
        variant: 'needle_set' or 'needle_set_hard'.
        max_tokens: Max token count for haystack expansion.
        corruption: 'remove' or 'scramble' (for CRI corrupted needle).
        max_characters_per_entry: Limit characters per entry (None = all 10).
        seed: Random seed for character selection.

    Returns:
        List of RetrievalTrial objects.
    """
    if max_tokens is None:
        max_tokens = max(context_lengths) if context_lengths else 50000

    entries = load_nolima_needle_set(nolima_dir, variant)
    haystack_texts = load_nolima_haystack_texts(nolima_dir, max_tokens)

    trials = []
    skipped_no_question = 0

    for entry_idx, entry in enumerate(entries):
        # Check if this entry has the requested question type
        if question_type not in entry["questions"]:
            skipped_no_question += 1
            continue

        question_template = entry["questions"][question_type]
        needle_template = entry["needle"]
        task_template = entry["task_template"]
        character_set = entry["character_set"]
        entry_id = entry["id"]

        # Cycle through haystack texts for different entries
        haystack_text = haystack_texts[entry_idx % len(haystack_texts)]

        for test_id, test_data in entry["tests"].items():
            input_args = test_data["input_args"]

            # Determine which characters to use
            chars_to_use = list(character_set)
            if max_characters_per_entry is not None:
                chars_to_use = chars_to_use[:max_characters_per_entry]

            for char_name in chars_to_use:
                # Expand templates
                needle_text = _expand_nolima_template(needle_template, char_name, input_args)
                question_text = _expand_nolima_template(question_template, char_name, input_args)

                # Build prompt template: expand {question} but leave {haystack}
                # as a placeholder for the detection script to fill at insertion time.
                # This preserves NoLiMa's intended prompt framing.
                full_prompt_template = task_template.replace("{question}", question_text)

                # The gold answer is always the character name for NoLiMa
                answer_text = char_name

                # Create corrupted needle for CRI
                corrupted = _make_corrupted_needle(needle_template, char_name, character_set, input_args, corruption)

                for ctx_len in context_lengths:
                    for depth in depth_percents:
                        trial_id = f"nolima_{entry_id}_{test_id}_{char_name}" f"_{ctx_len}_{int(depth)}"
                        trials.append(
                            RetrievalTrial(
                                trial_id=trial_id,
                                haystack_text=haystack_text,
                                needle_text=needle_text,
                                answer_text=answer_text,
                                question=question_text,
                                prompt_template=full_prompt_template,
                                context_length=ctx_len,
                                depth_percent=depth,
                                corrupted_needle=corrupted,
                                nolima_question_type=question_type,
                                nolima_entry_id=entry_id,
                            )
                        )

    # Note: With 9 entries × ~3 tests × 10 chars × 20 lengths × 10 depths,
    # this can produce ~54,000 trials. For Wu et al. scoring this is manageable
    # (each trial is one decode). For CRI, use stratified_sample() to pick a
    # balanced subset via --num-examples.

    return trials


def stratified_sample(
    trials: list[RetrievalTrial],
    num_examples: int,
    seed: int = 42,
) -> list[RetrievalTrial]:
    """Sample trials with stratification across expansion axes.

    Groups trials by (entry/needle, test_variant, depth) and samples
    uniformly across groups. This ensures coverage of all needle types,
    test variants, and depth positions rather than oversampling common
    combinations.

    For NIAH: stratifies by (needle_idx, depth_percent).
    For NoLiMa: stratifies by (entry_id, test portion of trial_id, depth_percent).
    """
    if len(trials) <= num_examples:
        return trials

    rng = random.Random(seed)

    # Build stratification key per trial
    def strat_key(t: RetrievalTrial) -> tuple:
        if t.nolima_entry_id is not None:
            # NoLiMa: group by (entry_id, depth)
            # Extract test_id from trial_id: "nolima_{entry}_{test}_{char}_{len}_{depth}"
            parts = t.trial_id.split("_")
            # entry_id and test_id are the 2nd and 3rd parts
            return (t.nolima_entry_id, parts[2] if len(parts) > 2 else "", t.depth_percent)
        else:
            # NIAH: group by (needle_idx, depth)
            return (t.niah_needle_idx, t.depth_percent)

    # Group trials by stratification key
    groups: dict[tuple, list[RetrievalTrial]] = defaultdict(list)
    for trial in trials:
        groups[strat_key(trial)].append(trial)

    # Round-robin sample from groups
    sampled: list[RetrievalTrial] = []
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    # Shuffle within each group
    for key in group_keys:
        rng.shuffle(groups[key])

    # Round-robin until we have enough
    group_iters = {key: iter(groups[key]) for key in group_keys}
    while len(sampled) < num_examples:
        added_this_round = False
        for key in group_keys:
            if len(sampled) >= num_examples:
                break
            try:
                sampled.append(next(group_iters[key]))
                added_this_round = True
            except StopIteration:
                continue
        if not added_this_round:
            break  # All groups exhausted

    return sampled


# ---------------------------------------------------------------------------
# NoLiMa prompt construction helpers
# ---------------------------------------------------------------------------


def build_nolima_question_prompt(
    entry: dict,
    question_type: str,
    char_name: str,
    input_args: list[str],
) -> str:
    """Build the full question prompt for a NoLiMa trial.

    Uses the entry's task_template with the question filled in.
    The {haystack} placeholder is left for the caller to fill
    (since it depends on needle insertion).
    """
    question_template = entry["questions"][question_type]
    question_text = _expand_nolima_template(question_template, char_name, input_args)

    # The task_template has {haystack} and {question} placeholders
    task_template = entry["task_template"]
    # Replace {question} but leave {haystack} for the caller
    prompt = task_template.replace("{question}", question_text)
    return prompt


def get_nolima_task_template(entry: dict) -> str:
    """Return the task template from a NoLiMa entry."""
    return entry["task_template"]


# ---------------------------------------------------------------------------
# NoLiMa data download
# ---------------------------------------------------------------------------

NOLIMA_HF_BASE = "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main"

NOLIMA_NEEDLE_FILES = [
    "needlesets/needle_set.json",
    "needlesets/needle_set_hard.json",
]

NOLIMA_HAYSTACK_FILES = [f"haystack/rand_shuffle/rand_book_{i}.txt" for i in range(1, 6)]


def _fetch_url(url: str, max_retries: int = 5) -> str:
    """Fetch text content from a URL with retry on rate-limiting."""
    import time as _time

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = 2**attempt  # 1, 2, 4, 8, 16 seconds
                print(f"Rate-limited (429), retrying in {wait}s... ({attempt + 1}/{max_retries})")
                _time.sleep(wait)
            else:
                raise


def download_nolima_data(output_dir: Path) -> None:
    """Download NoLiMa needle sets and haystack data.

    Downloads from the HuggingFace dataset (amodaresi/NoLiMa) into the specified directory.

    Args:
        output_dir: Target directory (e.g., data/nolima/).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download needle set files
    for rel_path in NOLIMA_NEEDLE_FILES:
        filename = Path(rel_path).name
        local_path = output_dir / filename
        if local_path.exists():
            continue

        url = f"{NOLIMA_HF_BASE}/{rel_path}"
        content = _fetch_url(url)

        # Validate JSON structure
        data = json.loads(content)
        assert isinstance(data, list), f"Expected list in {filename}, got {type(data)}"
        assert len(data) >= 1, f"{filename} is empty"

        local_path.write_text(content)

    # Download haystack files
    haystack_dir = output_dir / "haystack" / "rand_shuffle"
    haystack_dir.mkdir(parents=True, exist_ok=True)

    for rel_path in NOLIMA_HAYSTACK_FILES:
        filename = Path(rel_path).name
        local_path = haystack_dir / filename
        if local_path.exists():
            continue

        url = f"{NOLIMA_HF_BASE}/{rel_path}"
        content = _fetch_url(url)
        assert len(content) > 1000, f"Haystack file {filename} suspiciously small ({len(content)} chars)"

        local_path.write_text(content)
