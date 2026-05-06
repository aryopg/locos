"""Standalone MedRAG medical QA evaluation task.

Evaluates medical question answering with BM25-retrieved PubMed passages.
Supports 5 sub-datasets: MMLU-Med, MedQA, MedMCQA, PubMedQA, SuperGPQA-Med.

Usage::
    python -m locos_eval.evals.tasks.medrag_task \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --heads retrieval_heads/Meta-Llama-3-8B-Instruct.json \
        --dataset-name medqa --top-k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from locos_eval.evals.runner import EvalRunner, EvalSample
from locos_eval.evals.scorers import extract_answer_letter

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompts() -> dict[str, str]:
    """Load prompt templates from YAML."""
    with open(_PROMPTS_DIR / "medrag.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def _format_prompt(
    question: str,
    options: dict[str, str],
    retrieved_passages: list[dict],
    top_k: int,
    prompts: dict[str, str],
) -> str:
    """Format the user message with passages, question, and options."""
    passages = retrieved_passages[:top_k]

    passage_tpl = prompts.get("passage", "[{index}] {title}\n{content}")
    passage_sep = prompts.get("passage_separator", "\n\n")

    passage_lines = [passage_tpl.format(index=i, content=p["content"]) for i, p in enumerate(passages, 1)]
    passages_text = passage_sep.join(passage_lines)

    sorted_keys = sorted(options.keys())
    option_lines = [f"{key}. {options[key]}" for key in sorted_keys]
    options_text = "\n".join(option_lines)

    user_tpl = prompts["user"]
    return user_tpl.format(passages=passages_text, question=question, options=options_text)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class MedRAGEval(EvalRunner):
    """Standalone MedRAG medical QA evaluation task.

    Loads medical QA samples from HuggingFace, formats prompts with
    BM25-retrieved passages, generates answers, and scores by MCQ accuracy.
    """

    def __init__(
        self,
        hf_repo: str = "aryopg/medrag-bm25-pubmed",
        dataset_name: str | None = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        assert top_k > 0, f"top_k must be positive, got {top_k}"

        super().__init__(**kwargs)
        self._hf_repo = hf_repo
        self._dataset_name = dataset_name
        self._top_k = top_k

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def task_name(self) -> str:
        parts = ["medrag"]
        if self._dataset_name:
            parts.append(self._dataset_name)
        parts.append(f"top{self._top_k}")
        return "_".join(parts)

    def system_message(self) -> str | None:
        prompts = _load_prompts()
        return prompts["system"]

    def load_samples(self) -> list[EvalSample]:
        """Load MedRAG samples from HuggingFace and format prompts."""
        try:
            from datasets import load_dataset as _load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required for MedRAG. " "Install it with: pip install datasets"
            ) from exc

        ds = _load_dataset(self._hf_repo, split="train")
        records = list(ds)

        if self._dataset_name is not None:
            records = [r for r in records if r["dataset"] == self._dataset_name]

        assert len(records) > 0, (
            f"No records found" f"{' for dataset ' + self._dataset_name if self._dataset_name else ''}"
        )

        prompts = _load_prompts()

        samples: list[EvalSample] = []
        for row in records:
            prompt = _format_prompt(
                question=row["question"],
                options=row["options"],
                retrieved_passages=row["retrieved_passages"],
                top_k=self._top_k,
                prompts=prompts,
            )
            sample = EvalSample(
                prompt=prompt,
                target=row["answer"],
                metadata={
                    "dataset": row["dataset"],
                    "question": row["question"],
                },
            )
            samples.append(sample)

        return samples

    def score(self, output: str, sample: EvalSample) -> dict[str, float]:
        """Score by MCQ accuracy: extract answer letter and compare to target."""
        predicted = extract_answer_letter(output)
        correct = sample.target.strip().upper()
        is_correct = predicted == correct
        return {"accuracy": 1.0 if is_correct else 0.0}

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    @staticmethod
    def add_task_args(parser: argparse.ArgumentParser) -> None:
        """Add MedRAG-specific CLI arguments."""
        parser.add_argument(
            "--hf-repo",
            type=str,
            default="aryopg/medrag-bm25-pubmed",
            help="HuggingFace dataset repository (default: aryopg/medrag-bm25)",
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            default=None,
            help="Filter to a specific sub-dataset (e.g. medqa, medmcqa)",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=5,
            help="Number of retrieved passages to include (default: 5)",
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="MedRAG medical QA evaluation")
    EvalRunner.add_common_args(parser)
    MedRAGEval.add_task_args(parser)
    args = parser.parse_args()
    EvalRunner.resolve_args(args)

    task = MedRAGEval(
        hf_repo=args.hf_repo,
        dataset_name=args.dataset_name,
        top_k=args.top_k,
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
        sampling_seed=args.sampling_seed,
        ablation_mode=args.ablation_mode,
        num_calibration=args.num_calibration,
        enforce_eager=args.enforce_eager,
    )
    task.run(score_only=args.score_only)


if __name__ == "__main__":
    main()
