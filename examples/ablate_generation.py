#!/usr/bin/env python3
"""Minimal LOCOS mean-ablation generation example.

Run with a vLLM-supported model and a retrieval-head JSON file:

    python examples/ablate_generation.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --heads retrieval_heads/Meta-Llama-3-8B-Instruct_logit_contrib_nolima.json
"""

from __future__ import annotations

import argparse

from locos_eval import ablation, load_retrieval_heads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate with LOCOS retrieval-head mean ablation.")
    parser.add_argument("--model", required=True, help="HuggingFace model id or local model path.")
    parser.add_argument("--heads", required=True, help="Retrieval-head JSON path.")
    parser.add_argument("--num-heads", type=int, default=50, help="Number of top heads to ablate.")
    parser.add_argument(
        "--prompt",
        default="Answer using the context: Paris is the capital of France.\nQuestion: What is the capital of France?\nAnswer:",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from vllm import LLM

    llm = LLM(model=args.model, enforce_eager=True)
    heads = load_retrieval_heads(args.heads, num_heads=args.num_heads)
    calibration_prompts = [
        "The city council published a short report about public transit.",
        "A recipe card listed flour, salt, oil, and water.",
    ]

    with ablation(
        llm,
        heads=heads,
        decoding="ablation",
        ablation_mode="mean",
        calibration_prompts=calibration_prompts,
    ) as generator:
        print(generator.generate(args.prompt, max_tokens=args.max_tokens))


if __name__ == "__main__":
    main()
