"""Standalone BABILong free-form QA evaluation task.

BABILong (RMT-team/babilong) is a long-context extension of the bAbI tasks
(Weston et al., 2015). Each subset (qa1-qa20) targets a different reasoning
pattern; ``qa2`` (Two Supporting Facts) and ``qa3`` (Three Supporting Facts)
are the most non-literal because the answer cannot be reached by single-span
copying — the model has to combine two or three facts scattered across the
story to produce a short noun answer.

Prompt expects the model to think step by step and emit the final short
answer inside an ``<answer>$ANSWER</answer>`` tag. Scoring is normalized
subspan match.

Usage::

    python -m locos_eval.evals.tasks.babilong_task \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --heads retrieval_heads/Meta-Llama-3-8B-Instruct.json \\
        --subset qa2 --split 0k
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from locos_eval.evals.runner import EvalRunner, EvalSample
from locos_eval.evals.scorers import extract_answer_text, subspan_match

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompts() -> dict[str, str]:
    with open(_PROMPTS_DIR / "babilong.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

_VALID_SUBSETS = frozenset({f"qa{i}" for i in range(1, 21)})


class BABILongEval(EvalRunner):
    """BABILong free-form QA evaluation.

    The HuggingFace dataset is laid out as: config = context length
    (``0k``, ``1k``, ``2k``, ...), split = subset name (``qa1`` ... ``qa20``).
    Each row has ``input`` (the story), ``question``, and ``target``.
    """

    def __init__(
        self,
        hf_repo: str = "RMT-team/babilong",
        subset: str = "qa2",
        split: str = "0k",
        **kwargs: Any,
    ) -> None:
        assert subset in _VALID_SUBSETS, f"subset must be one of qa1..qa20, got {subset!r}"
        super().__init__(**kwargs)
        self._hf_repo = hf_repo
        self._subset = subset
        self._split = split

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def task_name(self) -> str:
        return f"babilong_{self._subset}_{self._split}"

    def system_message(self) -> str | None:
        return _load_prompts()["system"]

    def load_samples(self) -> list[EvalSample]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required for BABILong. " "Install it with: pip install datasets"
            ) from exc

        # config = context-length bucket, split = qaN
        ds = load_dataset(self._hf_repo, self._split, split=self._subset)
        records = list(ds)
        assert len(records) > 0, f"No records for subset={self._subset!r} split={self._split!r}"

        prompts = _load_prompts()
        user_tpl = prompts["user"]

        samples: list[EvalSample] = []
        for row in records:
            # ``input`` is the story alone; ``question`` is separate.
            # Concatenate per spec: "{input}\n{question}" goes into the user
            # template, so we store them as the {story} and {question} slots.
            prompt = user_tpl.format(story=row["input"], question=row["question"])
            sample = EvalSample(
                prompt=prompt,
                target=row["target"],
                metadata={
                    "subset": self._subset,
                    "split": self._split,
                    "question": row["question"],
                },
            )
            samples.append(sample)
        return samples

    def score(self, output: str, sample: EvalSample) -> dict[str, float]:
        """Normalized subspan match against the gold target.

        We first try to pull the contents of an ``<answer>…</answer>`` tag.
        If the model didn't emit a tag (e.g. ablation broke instruction
        following), we fall back to scoring against the full output so a
        correct-but-untagged answer still counts.
        """
        tagged = extract_answer_text(output)
        candidate = tagged if tagged is not None else output
        is_match = subspan_match(candidate, sample.target)
        return {
            "accuracy": 1.0 if is_match else 0.0,
            "tag_present": 1.0 if tagged is not None else 0.0,
        }

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    @staticmethod
    def add_task_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--hf-repo",
            type=str,
            default="RMT-team/babilong",
            help="HuggingFace dataset repository (default: RMT-team/babilong)",
        )
        parser.add_argument(
            "--subset",
            type=str,
            required=True,
            choices=sorted(_VALID_SUBSETS),
            help="bAbI subset (qa1..qa20). qa2 / qa3 are recommended for non-literal multi-hop tests.",
        )
        parser.add_argument(
            "--split",
            type=str,
            default="0k",
            help="Context-length bucket (default: 0k). Other options: 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k.",
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="BABILong free-form QA evaluation")
    EvalRunner.add_common_args(parser)
    BABILongEval.add_task_args(parser)
    args = parser.parse_args()
    EvalRunner.resolve_args(args)

    task = BABILongEval(
        hf_repo=args.hf_repo,
        subset=args.subset,
        split=args.split,
        model=args.model,
        heads=args.heads,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sampling_top_p=args.sampling_top_p,
        sampling_top_k=args.sampling_top_k,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        limit=args.limit,
        output_dir=args.output_dir,
        decoding=args.decoding,
        heads_label=args.heads_label,
        num_heads=args.num_heads,
        random_seed=args.random_seed,
        sampling_seed=args.sampling_seed,
        ablation_mode=args.ablation_mode,
        num_calibration=args.num_calibration,
        enforce_eager=args.enforce_eager,
    )
    task.run(score_only=args.score_only)


if __name__ == "__main__":
    main()
