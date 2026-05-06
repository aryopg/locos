"""Standalone LongBench-v2 long-context evaluation task.

Evaluates long-context understanding using multiple-choice questions from
LongBench-v2 (Bai et al., 2024).  Supports filtering by length category
(short/medium/long) and stores rich metadata (domain, sub_domain,
difficulty, length) for fine-grained analysis.

Usage::

    python -m locos_eval.evals.tasks.longbench_v2_task \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --heads retrieval_heads/Meta-Llama-3-8B-Instruct.json \
        --max-model-len 32768 --length short

Score-only (no GPU)::

    python -m locos_eval.evals.tasks.longbench_v2_task \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --score-only path/to/generations.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from locos_eval.evals.runner import EvalRunner, EvalSample, console
from locos_eval.evals.scorers import extract_answer_letter

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompts() -> dict[str, str]:
    """Load prompt templates from YAML."""
    with open(_PROMPTS_DIR / "longbench_v2.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Context truncation
# ---------------------------------------------------------------------------


def _measure_non_context_tokens(
    row: dict,
    prompts: dict[str, str],
    system_msg: str | None,
    tokenizer: Any,
) -> int:
    """Measure the exact token count of everything *except* the context.

    Builds the full prompt with an empty context, applies the chat template,
    and counts tokens.  This gives a precise per-sample overhead so the
    context gets the maximum possible budget.
    """
    empty_prompt = _format_prompt(
        context="",
        question=row["question"],
        choice_A=row["choice_A"],
        choice_B=row["choice_B"],
        choice_C=row["choice_C"],
        choice_D=row["choice_D"],
        prompts=prompts,
    )
    messages: list[dict[str, str]] = []
    if system_msg is not None:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": empty_prompt})

    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback: plain text (mirrors EvalRunner._format_prompt)
        parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
        parts.append("")
        formatted = "\n\n".join(parts)

    return len(tokenizer.encode(formatted, add_special_tokens=False))


def _truncate_context(
    context: str,
    tokenizer: Any,
    max_ctx_tokens: int,
) -> tuple[str, bool]:
    """Truncate *context* using a first-half + last-half strategy.

    Follows the official LongBench-v2 truncation methodology: keep the first
    ``max_ctx_tokens // 2`` tokens and the last ``max_ctx_tokens // 2`` tokens.

    Returns:
        ``(possibly_truncated_context, was_truncated)``
    """
    token_ids = tokenizer.encode(context, add_special_tokens=False)
    if len(token_ids) <= max_ctx_tokens:
        return context, False

    first_half = max_ctx_tokens // 2
    second_half = max_ctx_tokens - first_half
    truncated_ids = token_ids[:first_half] + token_ids[-second_half:]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True), True


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def _format_prompt(
    context: str,
    question: str,
    choice_A: str,
    choice_B: str,
    choice_C: str,
    choice_D: str,
    prompts: dict[str, str],
) -> str:
    """Format the user message using the YAML template."""
    return prompts["user"].format(
        context=context,
        question=question,
        choice_A=choice_A,
        choice_B=choice_B,
        choice_C=choice_C,
        choice_D=choice_D,
    )


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

_VALID_LENGTHS = frozenset({"short", "medium", "long", "all"})


class LongBenchV2Eval(EvalRunner):
    """LongBench-v2 long-context MCQ evaluation.

    Filters by length category and truncates contexts that exceed the model's
    context window using the official first-half / last-half strategy.
    Stores domain, sub_domain, difficulty, and length metadata for analysis.
    """

    def __init__(
        self,
        hf_repo: str = "zai-org/LongBench-v2",
        length: str = "short",
        **kwargs: Any,
    ) -> None:
        assert length in _VALID_LENGTHS, f"length must be one of {sorted(_VALID_LENGTHS)}, got {length!r}"
        super().__init__(**kwargs)
        self._hf_repo = hf_repo
        self._length = length

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def task_name(self) -> str:
        if self._length == "all":
            return "longbench_v2"
        return f"longbench_v2_{self._length}"

    def system_message(self) -> str | None:
        prompts = _load_prompts()
        return prompts["system"]

    def load_samples(self) -> list[EvalSample]:
        """Load LongBench-v2 samples, filter by length, and truncate contexts."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required for LongBench-v2. " "Install it with: pip install datasets"
            ) from exc

        from transformers import AutoTokenizer

        ds = load_dataset(self._hf_repo, split="train")

        # Filter by length category
        if self._length != "all":
            ds = ds.filter(lambda x: x["length"] == self._length)

        records = list(ds)
        assert len(records) > 0, f"No records found for length={self._length!r}"

        prompts = _load_prompts()

        # Load tokenizer for context truncation
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        sys_msg = self.system_message()

        samples: list[EvalSample] = []
        n_truncated = 0
        for row in records:
            context = row["context"]

            # Compute exact non-context token count for this sample
            # (question + choices + system message + chat template markup)
            non_ctx_tokens = _measure_non_context_tokens(
                row,
                prompts,
                sys_msg,
                tokenizer,
            )
            max_ctx_tokens = self._max_model_len - self._max_tokens - non_ctx_tokens
            assert max_ctx_tokens > 0, (
                f"max_model_len ({self._max_model_len}) too small for "
                f"max_tokens ({self._max_tokens}) + non-context overhead "
                f"({non_ctx_tokens}) on sample {row['_id']!r}"
            )

            # Measure original token count for metadata
            original_token_ids = tokenizer.encode(context, add_special_tokens=False)
            original_ctx_tokens = len(original_token_ids)

            # Truncate if necessary (first-half + last-half)
            context, was_truncated = _truncate_context(context, tokenizer, max_ctx_tokens)
            if was_truncated:
                n_truncated += 1

            prompt = _format_prompt(
                context=context,
                question=row["question"],
                choice_A=row["choice_A"],
                choice_B=row["choice_B"],
                choice_C=row["choice_C"],
                choice_D=row["choice_D"],
                prompts=prompts,
            )

            sample = EvalSample(
                prompt=prompt,
                target=row["answer"].strip().upper(),
                metadata={
                    "id": row["_id"],
                    "domain": row["domain"],
                    "sub_domain": row["sub_domain"],
                    "difficulty": row["difficulty"],
                    "length": row["length"],
                    "context_tokens": original_ctx_tokens,
                    "truncated": was_truncated,
                },
            )
            samples.append(sample)

        if n_truncated > 0:
            console.print(
                f"[yellow]Truncated {n_truncated}/{len(samples)} contexts "
                f"(first-half + last-half) to fit max_model_len="
                f"{self._max_model_len}[/yellow]"
            )

        return samples

    def score(self, output: str, sample: EvalSample) -> dict[str, float]:
        """Score by MCQ accuracy with optional compensation for unparseable answers."""
        predicted = extract_answer_letter(output)
        correct = sample.target.strip().upper()

        if predicted is None:
            # Compensated: random-chance credit (0.25) for unparseable outputs
            return {"accuracy": 0.0, "accuracy_compensated": 0.25}

        is_correct = predicted == correct
        score_val = 1.0 if is_correct else 0.0
        return {"accuracy": score_val, "accuracy_compensated": score_val}

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    @staticmethod
    def add_task_args(parser: argparse.ArgumentParser) -> None:
        """Add LongBench-v2-specific CLI arguments."""
        parser.add_argument(
            "--hf-repo",
            type=str,
            default="zai-org/LongBench-v2",
            help="HuggingFace dataset repository (default: zai-org/LongBench-v2)",
        )
        parser.add_argument(
            "--length",
            type=str,
            default="short",
            choices=["short", "medium", "long", "all"],
            help="Filter by length category (default: short)",
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="LongBench-v2 long-context evaluation")
    EvalRunner.add_common_args(parser)
    LongBenchV2Eval.add_task_args(parser)
    args = parser.parse_args()
    EvalRunner.resolve_args(args)

    task = LongBenchV2Eval(
        hf_repo=args.hf_repo,
        length=args.length,
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
