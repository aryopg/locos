"""Standalone MuSiQue multi-hop QA evaluation task.

MuSiQue (bdsaglam/musique) is a multi-hop open-book QA dataset where every
question requires composing 2-4 facts from a pool of 20 distractor-rich
paragraphs (Trivedi et al., 2022). Because the answer cannot be reached by
single-span copying, MuSiQue is a strong test of *non-literal* retrieval
(synthesis across passages), complementing summarisation tasks like
ACI-Bench / XSum.

Prompt asks the model to reason step by step and place the final short
answer inside an ``<answer>$ANSWER</answer>`` tag. Scoring is normalized
subspan match against ``answer`` and any of ``answer_aliases`` — credit if
the prediction contains any of them.

Usage::

    python -m locos_eval.evals.tasks.musique_task \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --heads retrieval_heads/Meta-Llama-3-8B-Instruct.json \\
        --subset answerable --split validation
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
    with open(_PROMPTS_DIR / "musique.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def _format_paragraphs(paragraphs: list[dict], prompts: dict[str, str]) -> str:
    """Verbalize the 20 paragraphs in their dataset order (matches MedRAG style)."""
    passage_tpl = prompts.get("passage", "[{index}] {title}\n{content}")
    sep = prompts.get("passage_separator", "\n\n")
    lines = [
        passage_tpl.format(
            index=i,
            title=p["title"],
            content=p["paragraph_text"],
        )
        for i, p in enumerate(paragraphs, 1)
    ]
    return sep.join(lines)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class MuSiQueEval(EvalRunner):
    """MuSiQue multi-hop open-book QA evaluation."""

    def __init__(
        self,
        hf_repo: str = "bdsaglam/musique",
        subset: str = "answerable",
        split: str = "validation",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._hf_repo = hf_repo
        self._subset = subset
        self._split = split

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def task_name(self) -> str:
        return f"musique_{self._subset}"

    def system_message(self) -> str | None:
        return _load_prompts()["system"]

    def load_samples(self) -> list[EvalSample]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required for MuSiQue. " "Install it with: pip install datasets"
            ) from exc

        ds = load_dataset(self._hf_repo, self._subset, split=self._split)
        records = list(ds)
        assert len(records) > 0, f"No records for subset={self._subset!r} split={self._split!r}"

        prompts = _load_prompts()
        user_tpl = prompts["user"]

        samples: list[EvalSample] = []
        for row in records:
            passages_text = _format_paragraphs(row["paragraphs"], prompts)
            prompt = user_tpl.format(paragraphs=passages_text, question=row["question"])
            # Gold answer set: primary answer plus any aliases.
            aliases = list(row.get("answer_aliases") or [])
            gold = [row["answer"], *aliases]
            sample = EvalSample(
                prompt=prompt,
                target=row["answer"],
                metadata={
                    "id": row["id"],
                    "subset": self._subset,
                    "split": self._split,
                    "question": row["question"],
                    "answer_aliases": aliases,
                    "gold_answers": gold,
                    "n_hops": int(row["id"].split("hop")[0]) if "hop" in row["id"] else None,
                },
            )
            samples.append(sample)
        return samples

    def score(self, output: str, sample: EvalSample) -> dict[str, float]:
        """Normalized subspan match against ``answer`` ∪ ``answer_aliases``."""
        tagged = extract_answer_text(output)
        candidate = tagged if tagged is not None else output
        gold_answers: list[str] = sample.metadata.get("gold_answers") or [sample.target]
        is_match = any(subspan_match(candidate, g) for g in gold_answers if g)
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
            default="bdsaglam/musique",
            help="HuggingFace dataset repository (default: bdsaglam/musique)",
        )
        parser.add_argument(
            "--subset",
            type=str,
            default="answerable",
            help="MuSiQue subset (default: answerable; also supports 'full').",
        )
        parser.add_argument(
            "--split",
            type=str,
            default="validation",
            help="Dataset split (default: validation).",
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="MuSiQue multi-hop QA evaluation")
    EvalRunner.add_common_args(parser)
    MuSiQueEval.add_task_args(parser)
    args = parser.parse_args()
    EvalRunner.resolve_args(args)

    task = MuSiQueEval(
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
