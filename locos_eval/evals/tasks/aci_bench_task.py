"""Standalone ACI-Bench dialogue-to-note evaluation task.

ACI-Bench (Ambient Clinical Intelligence Benchmark) evaluates a model's ability
to generate clinical notes from doctor-patient dialogues.  We use the D2N
(dialogue-to-note) subset with ROUGE-L, BERTScore, and LLM-judge metrics.

Usage::

    python -m locos_eval.evals.tasks.aci_bench_task \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --heads retrieval_heads/Meta-Llama-3-8B-Instruct.json

    # With few-shot examples and custom judge model:
    python -m locos_eval.evals.tasks.aci_bench_task \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --heads retrieval_heads/Meta-Llama-3-8B-Instruct.json \
        --n-shot 2 --judge-model claude-haiku-4-5-20251001

    # Disable LLM judge:
    python -m locos_eval.evals.tasks.aci_bench_task \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --heads retrieval_heads/Meta-Llama-3-8B-Instruct.json \
        --judge-model none
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from locos_eval.evals.runner import EvalResult, EvalRunner, EvalSample

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_JUDGE_AXES = ("completeness", "accuracy", "relevance")


def _load_prompts() -> dict[str, str]:
    """Load prompt templates from YAML."""
    with open(_PROMPTS_DIR / "aci_bench.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _format_prompt(
    test_dialogue: str,
    train_examples: list[dict[str, Any]],
    prompts: dict[str, str],
) -> str:
    """Build the user-message prompt with optional few-shot examples."""
    separator = prompts.get("few_shot_separator", "\n\n---\n\n")
    example_tpl = prompts["few_shot_example"]
    user_tpl = prompts["user"]

    parts: list[str] = []
    for i, ex in enumerate(train_examples, start=1):
        parts.append(example_tpl.format(index=i, dialogue=ex["inputs"], note=ex["target"]))

    test_prompt = user_tpl.format(dialogue=test_dialogue)

    if parts:
        return separator.join(parts) + separator + test_prompt
    return test_prompt


class ACIBenchEval(EvalRunner):
    """ACI-Bench dialogue-to-note evaluation.

    Loads examples from the ``aryopg/aci-bench-d2n`` dataset on HuggingFace
    and scores generated clinical notes with ROUGE-L, BERTScore, and an
    LLM judge (completeness, accuracy, relevance).
    """

    def __init__(
        self,
        *,
        hf_repo: str = "aryopg/aci-bench-d2n",
        n_shot: int = 0,
        judge_model: str = "claude-haiku-4-5-20251001",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert n_shot >= 0, f"n_shot must be non-negative, got {n_shot}"
        self._hf_repo = hf_repo
        self._n_shot = n_shot
        self._judge_model = judge_model if judge_model != "none" else None
        self._prompts = _load_prompts()

    # ------------------------------------------------------------------
    # EvalRunner hooks
    # ------------------------------------------------------------------

    def task_name(self) -> str:
        return "aci_bench"

    def system_message(self) -> str | None:
        return self._prompts["system"]

    def load_samples(self) -> list[EvalSample]:
        """Load ACI-Bench D2N samples from HuggingFace.

        Each row has ``inputs`` (dialogue) and ``target`` (clinical note).
        When ``n_shot > 0``, training examples are prepended to the prompt.
        """
        try:
            from datasets import load_dataset as _load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required for ACI-Bench. " "Install it with: pip install datasets"
            ) from exc

        test_ds = _load_dataset(self._hf_repo, split="test")
        test_records = list(test_ds)
        assert len(test_records) > 0, "No test records found"

        train_examples: list[dict[str, Any]] = []
        if self._n_shot > 0:
            train_ds = _load_dataset(self._hf_repo, split="train")
            train_records = list(train_ds)
            assert len(train_records) >= self._n_shot, (
                f"Requested {self._n_shot} few-shot examples but only "
                f"{len(train_records)} training samples available"
            )
            train_examples = train_records[: self._n_shot]

        samples: list[EvalSample] = []
        for record in test_records:
            prompt = _format_prompt(record["inputs"], train_examples, self._prompts)
            samples.append(
                EvalSample(
                    prompt=prompt,
                    target=record["target"],
                    metadata={
                        "dialogue": record["inputs"],
                        "n_shot": self._n_shot,
                    },
                )
            )

        return samples

    def score(self, output: str, sample: EvalSample) -> dict[str, float]:
        from locos_eval.evals.scorers import bertscore_f1, rouge_l_score

        scores: dict[str, float] = {
            "rouge_l": rouge_l_score(output, sample.target),
            "bertscore": bertscore_f1(output, sample.target),
        }
        if self._judge_model is not None:
            scores.update(self._run_judge(output, sample))
        else:
            scores.update({f"judge_{a}": -1.0 for a in _JUDGE_AXES})
            scores["judge_normalized"] = -1.0
        return scores

    def score_all(self, outputs: list[str], samples: list[EvalSample]) -> list[dict[str, float]]:
        """Batch scoring — runs BERTScore once for all outputs, then per-sample judge."""
        from rich.progress import track as _track

        from locos_eval.evals.scorers import bertscore_f1_batch, rouge_l_score

        references = [s.target for s in samples]
        rouge_scores = [rouge_l_score(o, r) for o, r in zip(outputs, references)]
        bert_scores = bertscore_f1_batch(outputs, references)

        all_scores: list[dict[str, float]] = []
        for output, sample, rouge, bert in _track(
            list(zip(outputs, samples, rouge_scores, bert_scores)),
            description="Scoring ACI-Bench...",
        ):
            scores: dict[str, float] = {"rouge_l": rouge, "bertscore": bert}
            if self._judge_model is not None:
                scores.update(self._run_judge(output, sample))
            else:
                scores.update({f"judge_{a}": -1.0 for a in _JUDGE_AXES})
                scores["judge_normalized"] = -1.0
            all_scores.append(scores)

        return all_scores

    def _run_judge(self, output: str, sample: EvalSample) -> dict[str, float]:
        """Call the LLM judge and return per-axis scores.

        Scores are returned in the dict. Explanations are stored in
        ``sample.metadata["judge_explanations"]`` so they persist to JSONL.

        Returns:
            Dict with ``judge_completeness``, ``judge_accuracy``,
            ``judge_relevance`` (int 1-5), and ``judge_normalized`` (float 0-1).
        """
        from locos_eval.evals.scorers import call_llm_judge

        dialogue = sample.metadata.get("dialogue", "")
        user_prompt = self._prompts["judge_user"].format(
            dialogue=dialogue,
            reference_note=sample.target,
            generated_note=output,
        )

        result = call_llm_judge(
            system_prompt=self._prompts["judge_system"],
            user_prompt=user_prompt,
            model=self._judge_model,
        )

        axis_scores: dict[str, int] = {}
        axis_explanations: dict[str, str] = {}
        for axis in _JUDGE_AXES:
            axis_data = result.get(axis, {})
            if isinstance(axis_data, dict):
                raw_score = axis_data.get("score", -1)
                axis_explanations[axis] = axis_data.get("explanation", "")
            elif isinstance(axis_data, int | float):
                # Handle case where judge returns bare score (e.g. "relevance": 4)
                raw_score = axis_data
                axis_explanations[axis] = ""
            else:
                raw_score = -1
                axis_explanations[axis] = ""

            if isinstance(raw_score, int | float) and 1 <= raw_score <= 5:
                axis_scores[axis] = int(raw_score)
            else:
                logger.warning("Invalid judge score for %s: %r, defaulting to -1", axis, raw_score)
                axis_scores[axis] = -1

        # Store explanations in sample metadata so they flow to EvalResult
        sample.metadata["judge_explanations"] = axis_explanations

        scores: dict[str, float] = {f"judge_{axis}": float(axis_scores[axis]) for axis in _JUDGE_AXES}

        # Normalized mean: (mean - 1) / 4, mapping [1,5] to [0,1]
        valid_scores = [s for s in axis_scores.values() if s > 0]
        if valid_scores:
            raw_mean = sum(valid_scores) / len(valid_scores)
            scores["judge_normalized"] = (raw_mean - 1) / 4
        else:
            scores["judge_normalized"] = -1

        return scores


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def rejudge_failed_scores(
    results_path: str,
    judge_model: str = "claude-haiku-4-5-20251001",
) -> None:
    """Re-run LLM judge on samples with failed judge scores (-1).

    Loads a results JSONL, identifies samples where any judge axis is -1,
    re-runs the judge on those, and overwrites the file with updated scores.
    No GPU or model loading required.

    Usage::

        python -m locos_eval.evals.tasks.aci_bench_task \
            --rejudge eval_results/aci_bench/.../greedy_20260402.jsonl \
            --judge-model claude-haiku-4-5-20251001
    """
    from rich.console import Console
    from rich.progress import track as _track

    from locos_eval.evals.scorers import call_llm_judge

    console = Console()
    path = Path(results_path)
    assert path.exists(), f"Results file not found: {path}"
    assert judge_model and judge_model != "none", "Cannot rejudge without a judge model — pass --judge-model"

    prompts = _load_prompts()
    records = EvalResult.load_jsonl(path)
    assert len(records) > 0, f"No records found in {path}"

    # Find records needing rejudging
    needs_rejudge = []
    for i, r in enumerate(records):
        scores = r.get("scores", {})
        if any(scores.get(f"judge_{axis}", -1) == -1 for axis in _JUDGE_AXES):
            needs_rejudge.append(i)

    if not needs_rejudge:
        console.print("[green]All samples have valid judge scores — nothing to rejudge.[/green]")
        return

    console.print(
        f"[yellow]Found {len(needs_rejudge)}/{len(records)} samples "
        f"with failed judge scores — re-running judge...[/yellow]"
    )

    failed = 0
    for idx in _track(needs_rejudge, description="Re-judging..."):
        r = records[idx]
        output = r["output"]
        dialogue = r.get("metadata", {}).get("dialogue", "")
        target = r["target"]

        user_prompt = prompts["judge_user"].format(
            dialogue=dialogue,
            reference_note=target,
            generated_note=output,
        )
        result = call_llm_judge(
            system_prompt=prompts["judge_system"],
            user_prompt=user_prompt,
            model=judge_model,
        )

        # Parse judge response
        axis_scores: dict[str, int] = {}
        axis_explanations: dict[str, str] = {}
        for axis in _JUDGE_AXES:
            axis_data = result.get(axis, {})
            if isinstance(axis_data, dict):
                raw_score = axis_data.get("score", -1)
                axis_explanations[axis] = axis_data.get("explanation", "")
            elif isinstance(axis_data, int | float):
                raw_score = axis_data
                axis_explanations[axis] = ""
            else:
                raw_score = -1
                axis_explanations[axis] = ""

            if isinstance(raw_score, int | float) and 1 <= raw_score <= 5:
                axis_scores[axis] = int(raw_score)
            else:
                axis_scores[axis] = -1

        judge_scores = {f"judge_{ax}": float(axis_scores[ax]) for ax in _JUDGE_AXES}
        valid = [s for s in axis_scores.values() if s > 0]
        judge_scores["judge_normalized"] = (sum(valid) / len(valid) - 1) / 4 if valid else -1

        # Only update if the rejudge actually succeeded (at least one valid score)
        if any(s > 0 for s in axis_scores.values()):
            r["scores"].update(judge_scores)
            r.setdefault("metadata", {})["judge_explanations"] = axis_explanations
        else:
            failed += 1

    # Overwrite the file with updated records
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    succeeded = len(needs_rejudge) - failed
    console.print(f"[green]Rejudged {succeeded}/{len(needs_rejudge)} samples successfully.[/green]")
    if failed > 0:
        console.print(f"[yellow]{failed} samples still have failed scores — " f"re-run --rejudge to retry.[/yellow]")


def main() -> None:
    parser = argparse.ArgumentParser(description="ACI-Bench dialogue-to-note evaluation")
    EvalRunner.add_common_args(parser)
    # --model is required by add_common_args, but --rejudge doesn't need it
    parser._option_string_actions["--model"].required = False
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="aryopg/aci-bench-d2n",
        help="HuggingFace dataset repo",
    )
    parser.add_argument(
        "--n-shot",
        type=int,
        default=0,
        help="Number of few-shot examples from training set (default: 0)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="claude-haiku-4-5-20251001",
        help="Anthropic model for LLM judge (use 'none' to disable)",
    )
    parser.add_argument(
        "--rejudge",
        type=str,
        default=None,
        help="Path to results JSONL — re-run LLM judge on samples with failed scores (-1). " "No GPU required.",
    )
    args = parser.parse_args()
    # resolve_args needs args.model — skip for rejudge-only mode
    if args.model is not None:
        EvalRunner.resolve_args(args)

    if args.rejudge is not None:
        # Rejudge mode: only needs judge model + prompts, no GPU/model loading
        rejudge_failed_scores(
            results_path=args.rejudge,
            judge_model=args.judge_model,
        )
    else:
        if args.model is None:
            parser.error("--model is required (unless using --rejudge)")
        task = ACIBenchEval(
            hf_repo=args.hf_repo,
            n_shot=args.n_shot,
            judge_model=args.judge_model,
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
