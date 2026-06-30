"""ExperimentKey — single source of truth for experiment naming.

Replaces scattered naming logic across runner.py and shell scripts with
a frozen, hashable dataclass that deterministically produces all path
components (model_slug, variant, key, local_dir).

Can be used as a library or invoked as a CLI::

    python -m locos_eval.evals.experiment_key \
        --task babilong --model meta-llama/Meta-Llama-3-8B-Instruct \
        --decoding ablation --heads retrieval_heads/Llama-3-8B_nolima.json \
        --variant
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentKey:
    """Immutable experiment identifier.

    All path/slug properties are derived deterministically from these fields.
    """

    task: str
    """Task name, e.g. ``"babilong"``, ``"musique"``."""

    model: str
    """Full HuggingFace model name, e.g. ``"meta-llama/Meta-Llama-3-8B-Instruct"``."""

    decoding: str
    """Decoding mode: ``"greedy"`` or ``"ablation"``."""

    heads_path: str | None = None
    """Path to retrieval heads JSON, ``"random"``, or ``None`` (greedy)."""

    heads_label: str | None = None
    """Explicit label override (e.g. ``"ori"``). Takes priority over filename inference."""

    num_heads: int | None = None
    """Number of random heads (only relevant when ``heads_path="random"``)."""

    random_seed: int = 42
    """Random seed for head sampling (only relevant when ``heads_path="random"``)."""

    sampling_seed: int | None = None
    """Seed for stochastic sampling. If set, appends ``_s{seed}`` to variant."""

    ablation_mode: str = "zero"
    """When decoding='ablation': replacement strategy ('zero' or 'mean').
    'mean' prefixes the variant label with ``mean_`` so output dirs don't clash
    with existing zero-ablation runs."""

    def __post_init__(self) -> None:
        assert self.task, "task must be a non-empty string"
        assert self.model, "model must be a non-empty string"
        assert self.decoding in ("greedy", "ablation"), f"Unknown decoding mode: {self.decoding!r}"
        assert self.num_heads is None or self.num_heads > 0, f"num_heads must be positive, got {self.num_heads}"
        assert (
            self.sampling_seed is None or self.sampling_seed > 0
        ), f"sampling_seed must be positive, got {self.sampling_seed}"
        assert self.ablation_mode in (
            "zero",
            "mean",
        ), f"ablation_mode must be 'zero' or 'mean', got {self.ablation_mode!r}"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_slug(self) -> str:
        """Provider-preserving slug: ``"meta-llama_Meta-Llama-3-8B-Instruct"``."""
        return self.model.replace("/", "_")

    @property
    def job_slug(self) -> str:
        """DNS-safe slug: lowercase, dots replaced with hyphens.

        Takes only the model name after the provider slash.
        """
        return self.model.split("/")[-1].lower().replace(".", "-")

    @property
    def variant(self) -> str:
        """Variant directory name derived from decoding mode + heads label + sampling seed.

        Examples::

            greedy                        → "greedy"
            greedy + seed 1               → "greedy_s1"
            ablation + Wu NoLiMa heads    → "ablation_wu_nolima"
            ablation + Wu NoLiMa + seed 2 → "ablation_wu_nolima_s2"
            ablation + random + seed 1    → "ablation_random_s42_n20_s1"
        """
        if self.decoding == "greedy":
            base = "greedy"
        else:
            label = self._compute_label()
            # Mean-ablation variants get a "mean_" prefix on the label so they
            # don't collide with existing zero-ablation results in eval_results/.
            # ``ablation_<label>``     → zero (default, backward compatible)
            # ``ablation_mean_<label>`` → mean
            if self.decoding == "ablation" and self.ablation_mode == "mean":
                label = f"mean_{label}" if label else "mean"
            base = f"{self.decoding}_{label}" if label else self.decoding
        if self.sampling_seed is not None:
            return f"{base}_s{self.sampling_seed}"
        return base

    @property
    def key(self) -> str:
        """Canonical experiment key: ``"{task}/{model_slug}/{variant}"``."""
        return f"{self.task}/{self.model_slug}/{self.variant}"

    def local_dir(self, output_dir: str = "eval_results") -> Path:
        """Local directory for this experiment's results.

        Structure: ``{output_dir}/{task}/{model_slug}/{variant}/``
        """
        return Path(output_dir) / self.task / self.model_slug / self.variant

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_label(self) -> str:
        """Derive a short label from heads configuration.

        Priority:
        1. Explicit ``heads_label`` if set.
        2. Empty string if ``heads_path`` is ``None``.
        3. ``"random_s{seed}_n{count}"`` if ``heads_path == "random"``.
        4. Filename suffix: ``stem.rsplit("_", 1)`` — if 2 parts and last
           is alpha, return it; else ``"niah"`` (default).
        """
        # heads_label="" is treated as an explicit override returning ""
        # (variant becomes bare decoding mode, e.g. "ablation" without suffix).
        # heads_label=None falls through to filename-based inference below.
        if self.heads_label is not None:
            return self.heads_label
        if self.heads_path is None:
            return ""
        if self.heads_path == "random":
            count = self.num_heads if self.num_heads is not None else 50
            return f"random_s{self.random_seed}_n{count}"
        stem = Path(self.heads_path).stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isalpha():
            return parts[1]
        return "niah"


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print experiment naming components.",
    )
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--decoding", required=True, help="Decoding mode")
    parser.add_argument("--heads", default=None, help="Heads path or 'random'")
    parser.add_argument("--heads-label", default=None, help="Explicit heads label")
    parser.add_argument("--num-heads", type=int, default=None, help="Number of random heads")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--sampling-seed", type=int, default=None, help="Seed for stochastic sampling (appends _s{seed} to variant)"
    )
    parser.add_argument(
        "--ablation-mode",
        type=str,
        default="zero",
        choices=["zero", "mean"],
        help="Ablation replacement mode (only meaningful with --decoding ablation)",
    )

    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--variant", action="store_true", help="Print variant name")
    output.add_argument("--key", action="store_true", help="Print full key")
    output.add_argument("--model-slug", action="store_true", help="Print model slug")
    output.add_argument("--local-dir", action="store_true", help="Print local directory")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    ek = ExperimentKey(
        task=args.task,
        model=args.model,
        decoding=args.decoding,
        heads_path=args.heads,
        heads_label=args.heads_label,
        num_heads=args.num_heads,
        random_seed=args.random_seed,
        sampling_seed=args.sampling_seed,
        ablation_mode=args.ablation_mode,
    )

    if args.variant:
        print(ek.variant)
    elif args.key:
        print(ek.key)
    elif args.model_slug:
        print(ek.model_slug)
    elif args.local_dir:
        print(ek.local_dir())


if __name__ == "__main__":
    main()
