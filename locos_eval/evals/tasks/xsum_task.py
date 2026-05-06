"""Standalone XSum summarization faithfulness evaluation task.

XSum (EdinburghNLP/xsum on HuggingFace) is an abstractive summarization
benchmark.  Each example contains a BBC article and a one-sentence reference
summary.  We evaluate generated summaries with ROUGE-L and BERTScore.

Usage::

    python -m locos_eval.evals.tasks.xsum_task \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --heads retrieval_heads/Meta-Llama-3-8B-Instruct.json

"""

from __future__ import annotations

import argparse

from locos_eval.evals.runner import EvalRunner, EvalSample

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Summarize the given article in exactly "
    "one sentence. Output only the summary sentence, nothing else."
)


class XSumEval(EvalRunner):
    """XSum summarization faithfulness evaluation.

    Loads examples from the ``EdinburghNLP/xsum`` dataset on HuggingFace
    and scores generated one-sentence summaries with ROUGE-L and BERTScore.
    """

    def __init__(
        self,
        *,
        hf_repo: str = "EdinburghNLP/xsum",
        split: str = "test",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._hf_repo = hf_repo
        self._split = split

    # ------------------------------------------------------------------
    # EvalRunner hooks
    # ------------------------------------------------------------------

    def task_name(self) -> str:
        return "xsum_faithfulness"

    def system_message(self) -> str | None:
        return _SYSTEM_PROMPT

    def load_samples(self) -> list[EvalSample]:
        """Load XSum examples from HuggingFace.

        Each row has ``document`` (input article) and ``summary`` (reference).
        The prompt is just the document text; the system message instructs
        the model to produce a one-sentence summary.
        """
        try:
            from datasets import load_dataset as _load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required for XSum. " "Install it with: pip install datasets"
            ) from exc

        ds = _load_dataset(self._hf_repo, split=self._split)

        samples: list[EvalSample] = []
        for row in ds:
            samples.append(
                EvalSample(
                    prompt=row["document"],
                    target=row["summary"],
                    metadata={"id": row["id"]},
                )
            )

        return samples

    def score(self, output: str, sample: EvalSample) -> dict[str, float]:
        from locos_eval.evals.scorers import bertscore_f1, factkb_score, rouge_l_score

        return {
            "rouge_l": rouge_l_score(output, sample.target),
            "bertscore": bertscore_f1(output, sample.target),
            "factkb": factkb_score(output, sample.prompt),
        }

    def score_all(self, outputs: list[str], samples: list[EvalSample]) -> list[dict[str, float]]:
        """Batch scoring — runs BERTScore and FactKB once for all outputs."""
        from locos_eval.evals.scorers import (
            bertscore_f1_batch,
            factkb_score_batch,
            rouge_l_score,
        )

        references = [s.target for s in samples]
        sources = [s.prompt for s in samples]
        rouge_scores = [rouge_l_score(o, r) for o, r in zip(outputs, references)]
        bert_scores = bertscore_f1_batch(outputs, references)
        factkb_scores = factkb_score_batch(outputs, sources)

        return [
            {"rouge_l": r, "bertscore": b, "factkb": f} for r, b, f in zip(rouge_scores, bert_scores, factkb_scores)
        ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="XSum summarization faithfulness evaluation")
    EvalRunner.add_common_args(parser)
    parser.add_argument("--hf-repo", type=str, default="EdinburghNLP/xsum", help="HuggingFace dataset repo")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    args = parser.parse_args()
    EvalRunner.resolve_args(args)

    task = XSumEval(
        hf_repo=args.hf_repo,
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
        sampling_seed=args.sampling_seed,
        ablation_mode=args.ablation_mode,
        num_calibration=args.num_calibration,
        enforce_eager=args.enforce_eager,
    )
    task.run(score_only=args.score_only)


if __name__ == "__main__":
    main()
