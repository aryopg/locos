"""Standalone NQ-Swap context faithfulness evaluation task.

NQ-Swap provides questions with counterfactual contexts where the original
answer entity has been swapped to a different entity.  A faithful model should
answer with the substituted answer (following context), not the original
(from parametric memory).

Metrics:
- ``sub_em``: Subspan exact match against the substituted answer.
  Higher means the model is more faithful to context (good for LOCOS ablation).
- ``org_em``: Subspan exact match against the original answer.
  Higher means the model relies on parametric memory (bad).

Usage::

    python -m locos_eval.evals.tasks.nq_swap_task \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --heads retrieval_heads/Meta-Llama-3-8B-Instruct.json
"""

from __future__ import annotations

import argparse

from locos_eval.evals.runner import EvalRunner, EvalSample
from locos_eval.evals.scorers import subspan_match

# ---------------------------------------------------------------------------
# Prompt template (matches the Inspect AI version)
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = (
    "Answer the following question based on the provided context. "
    "Give only the answer, no explanation.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


# ---------------------------------------------------------------------------
# Task implementation
# ---------------------------------------------------------------------------


class NQSwapEval(EvalRunner):
    """NQ-Swap context faithfulness evaluation."""

    def __init__(
        self,
        *args,
        hf_repo: str = "aryopg/nq-swap",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._hf_repo = hf_repo

    def task_name(self) -> str:
        return "nq_swap"

    def system_message(self) -> str | None:
        return None

    def load_samples(self) -> list[EvalSample]:
        """Load NQ-Swap examples from HuggingFace.

        Each row has ``sub_context``, ``question``, ``sub_answer``,
        ``org_answer``.  The prompt uses the substituted context.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required for NQ-Swap. " "Install it with: pip install datasets"
            ) from exc

        ds = load_dataset(self._hf_repo, split="test")

        samples: list[EvalSample] = []
        for row in ds:
            prompt = _PROMPT_TEMPLATE.format(
                context=row["sub_context"],
                question=row["question"],
            )
            sample = EvalSample(
                prompt=prompt,
                target=row["sub_answer"],
                metadata={
                    "question": row["question"],
                    "sub_context": row["sub_context"],
                    "sub_answer": row["sub_answer"],
                    "org_answer": row["org_answer"],
                },
            )
            samples.append(sample)

        assert len(samples) > 0, f"No samples loaded from {self._hf_repo}"
        return samples

    def score(self, output: str, sample: EvalSample) -> dict[str, float]:
        """Score output against both substituted and original answers."""
        sub_answer = sample.metadata["sub_answer"]
        org_answer = sample.metadata["org_answer"]

        return {
            "sub_em": 1.0 if subspan_match(output, sub_answer) else 0.0,
            "org_em": 1.0 if subspan_match(output, org_answer) else 0.0,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NQ-Swap context faithfulness evaluation (standalone)",
    )
    EvalRunner.add_common_args(parser)
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="aryopg/nq-swap",
        help="HuggingFace dataset repository (default: aryopg/nq-swap)",
    )
    args = parser.parse_args()
    EvalRunner.resolve_args(args)

    runner = NQSwapEval(
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
        hf_repo=args.hf_repo,
        decoding=args.decoding,
        heads_label=args.heads_label,
        num_heads=args.num_heads,
        sampling_seed=args.sampling_seed,
        ablation_mode=args.ablation_mode,
        num_calibration=args.num_calibration,
        enforce_eager=args.enforce_eager,
    )
    runner.run(score_only=args.score_only)
