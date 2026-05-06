"""Shared eval runner base class for standalone LOCOS evaluation tasks.

Handles model loading, chat template formatting, generation loop,
result I/O, and rich console output. All eval tasks subclass EvalRunner.
"""

import argparse
import faulthandler
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

from locos_eval.evals.experiment_key import ExperimentKey
from locos_eval.evals.model_config import DEFAULTS as _CONFIG_DEFAULTS
from locos_eval.evals.model_config import load_model_config

# ---------------------------------------------------------------------------
# .env loading (module-level, before any model imports)
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
if _ENV_PATH.exists():
    with open(_ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

console = Console()


def _extract_layer_head_counts(config, model_name: str) -> tuple[int, int]:
    """Pull (num_layers, num_attention_heads) from a HuggingFace config.

    Multimodal wrappers (Gemma3ForConditionalGeneration, LLaVA, …) nest the
    text-model dims under ``config.text_config``; plain causal LMs expose
    them at the top level. Falls back through both locations.
    """
    text_config = getattr(config, "text_config", config)
    num_layers = getattr(text_config, "num_hidden_layers", None) or getattr(config, "num_hidden_layers", None)
    num_attn_heads = getattr(text_config, "num_attention_heads", None) or getattr(config, "num_attention_heads", None)
    assert num_layers is not None and num_attn_heads is not None, (
        f"Could not determine num_hidden_layers / num_attention_heads for {model_name!r} "
        f"from config (top-level keys: {list(config.to_dict().keys())})"
    )
    return num_layers, num_attn_heads


def _start_sample_watchdog(idx, sample, gen_path, timeout_s, runner):
    """Arm a per-sample watchdog timer; on expiry dump tracebacks, write a
    sentinel checkpoint entry, and hard-exit.

    A deadlocked CUDA / vLLM collective_rpc call cannot be interrupted from
    Python signal handlers, so the only way to break out is to terminate the
    process. The outer job script's resume-from-checkpoint logic will skip
    the sentinel sample on the next run.
    """
    if timeout_s is None or timeout_s <= 0:
        return None

    def _on_timeout():
        import contextlib

        msg = f"[TIMEOUT after {timeout_s:.0f}s on sample {idx}]"
        with contextlib.suppress(Exception):
            print(f"\n[ablation-watchdog] {msg} — dumping tracebacks to stderr", file=sys.stderr, flush=True)
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        with contextlib.suppress(Exception):
            runner._save_generation(gen_path, idx, msg, sample)
        os._exit(2)

    t = threading.Timer(timeout_s, _on_timeout)
    t.daemon = True
    t.start()
    return t


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class EvalSample:
    """A single evaluation sample before generation.

    Attributes:
        prompt: Formatted user-message content (pre chat-template).
        target: Gold reference for scoring.
        metadata: Arbitrary task-specific metadata.
    """

    prompt: str
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating a single sample.

    Attributes:
        sample_id: Index of the sample in the dataset.
        output: Model-generated text.
        target: Gold reference for scoring.
        scores: Metric name -> value mapping.
        metadata: Arbitrary task-specific metadata.
    """

    sample_id: int
    output: str
    target: str
    scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def save_jsonl(results: list["EvalResult"], path: Path) -> None:
        """Persist a list of EvalResults as newline-delimited JSON."""
        assert len(results) > 0, "Cannot save empty results list"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    @staticmethod
    def load_jsonl(path: Path) -> list[dict]:
        """Load results from a JSONL file as plain dicts."""
        path = Path(path)
        assert path.exists(), f"Results file not found: {path}"
        results = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results


# ---------------------------------------------------------------------------
# Base runner
# ---------------------------------------------------------------------------
class EvalRunner:
    """Base class for standalone LOCOS evaluation tasks.

    Subclasses must implement :meth:`load_samples` and :meth:`score`.
    Optionally override :meth:`task_name` and :meth:`system_message`.

    Model initialisation is lazy -- :meth:`_init_model` is called inside
    :meth:`run`, not in ``__init__``, so subclasses can set up task-specific
    state before the (expensive) model load.
    """

    def __init__(
        self,
        model: str,
        heads: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        sampling_top_p: float = 1.0,
        sampling_top_k: int = -1,
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.5,
        limit: int | None = None,
        output_dir: str = "eval_results",
        **kwargs: Any,
    ) -> None:
        assert max_tokens > 0, f"max_tokens must be positive, got {max_tokens}"
        assert temperature >= 0, f"temperature must be non-negative, got {temperature}"
        assert 0 < sampling_top_p <= 1.0, f"sampling_top_p must be in (0, 1], got {sampling_top_p}"
        assert (
            sampling_top_k == -1 or sampling_top_k > 0
        ), f"sampling_top_k must be -1 (disabled) or positive, got {sampling_top_k}"
        assert (
            max_model_len is None or max_model_len > 0
        ), f"max_model_len must be positive or None, got {max_model_len}"
        assert tensor_parallel_size > 0, f"tensor_parallel_size must be positive, got {tensor_parallel_size}"
        assert (
            0 < gpu_memory_utilization <= 1
        ), f"gpu_memory_utilization must be in (0, 1], got {gpu_memory_utilization}"
        assert limit is None or limit > 0, f"limit must be positive, got {limit}"

        self._model_name = model
        self._heads_path = heads
        decoding = kwargs.get("decoding", "greedy")
        if heads is None and decoding == "ablation":
            raise ValueError(f"--heads is required when --decoding is '{decoding}'")
        if decoding == "ablation" and kwargs.get("num_heads") is None:
            kwargs["num_heads"] = 50

        # Mean-ablation calibration knobs. Resolved into actual prompts in
        # ``run()`` after ``load_samples()`` so calibration uses the same
        # task-formatted prompts as the eval distribution.
        ablation_mode = kwargs.get("ablation_mode", "zero")
        assert ablation_mode in (
            "zero",
            "mean",
        ), f"ablation_mode must be 'zero' or 'mean', got {ablation_mode!r}"
        if ablation_mode == "mean" and decoding != "ablation":
            raise ValueError(
                f"ablation_mode={ablation_mode!r} is only meaningful with --decoding ablation, "
                f"got --decoding {decoding!r}"
            )
        num_calibration = kwargs.get("num_calibration", 50)
        assert (
            isinstance(num_calibration, int) and num_calibration > 0
        ), f"num_calibration must be positive int, got {num_calibration!r}"

        # ``enforce_eager`` gates vLLM's torch.compile + CUDA-graph capture.
        # The native ablation path depends on instance-attribute monkey-patches
        # to ``attn.forward``. torch.compile / CUDA-graph capture freezes the
        # original forward at engine init, so a non-eager ablation run silently
        # bypasses the q-replacement and produces outputs identical to greedy.
        # Greedy itself does no patching and is the only mode safe to compile.
        enforce_eager = kwargs.get("enforce_eager", True)
        if not enforce_eager and decoding == "ablation":
            console.print(
                "[yellow]WARNING: enforce_eager=False on --decoding ablation is unsafe — "
                "torch.compile bypasses the monkey-patched attn.forward, silently producing "
                "greedy outputs. Forcing enforce_eager=True.[/yellow]"
            )
            enforce_eager = True
        self._enforce_eager = enforce_eager
        self._ablation_mode = ablation_mode
        self._num_calibration = num_calibration
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = sampling_top_p
        self._top_k = sampling_top_k
        self._max_model_len = max_model_len
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._limit = limit
        self._output_dir = Path(output_dir)
        self._ablation_kwargs = kwargs  # decoding, ablation_mode, num_heads, etc.

        # Set by _init_model()
        self._wrapper = None
        self._tokenizer = None

    @property
    def experiment_key(self) -> ExperimentKey:
        """Canonical experiment key for this runner configuration."""
        return ExperimentKey(
            task=self.task_name(),
            model=self._model_name,
            decoding=self._ablation_kwargs.get("decoding", "greedy"),
            heads_path=self._heads_path,
            heads_label=self._ablation_kwargs.get("heads_label"),
            num_heads=self._ablation_kwargs.get("num_heads"),
            random_seed=self._ablation_kwargs.get("random_seed", 42),
            sampling_seed=self._ablation_kwargs.get("sampling_seed"),
            ablation_mode=self._ablation_mode,
        )

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def task_name(self) -> str:
        """Human-readable task name (defaults to class name)."""
        return self.__class__.__name__

    def system_message(self) -> str | None:
        """Optional system message prepended to every prompt."""
        return None

    def load_samples(self) -> list[EvalSample]:
        """Load evaluation samples. Must be implemented by subclass."""
        raise NotImplementedError

    def score(self, output: str, sample: EvalSample) -> dict[str, float]:
        """Score a single generation against its reference. Must be implemented by subclass."""
        raise NotImplementedError

    def score_all(self, outputs: list[str], samples: list[EvalSample]) -> list[dict[str, float]]:
        """Batch-score all generations. Override for efficient batch scoring.

        Default implementation calls ``score()`` per sample. Subclasses can
        override to batch expensive operations (e.g., BERTScore on all outputs
        at once).
        """
        return [self.score(output, sample) for output, sample in zip(outputs, samples)]

    # ------------------------------------------------------------------
    # Model initialisation (lazy)
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        """Create vLLM LLM, wrap with ablation(), and store tokenizer."""
        from vllm import LLM

        from locos_eval.wrapper import ablation

        console.print(
            Panel(
                f"[bold]Loading model:[/bold] {self._model_name}\n"
                f"[bold]Heads:[/bold] {self._heads_path or 'N/A (greedy)'}\n"
                f"[bold]Decoding:[/bold] {self._ablation_kwargs.get('decoding', 'greedy')}\n"
                f"[bold]TP:[/bold] {self._tensor_parallel_size}  "
                f"[bold]Max model len:[/bold] {self._max_model_len or 'auto'}  "
                f"[bold]GPU mem:[/bold] {self._gpu_memory_utilization}  "
                f"[bold]Eager:[/bold] {self._enforce_eager}",
                title=f"[green]{self.task_name()}[/green]",
            )
        )

        llm_kwargs: dict[str, Any] = dict(
            model=self._model_name,
            enforce_eager=self._enforce_eager,
            tensor_parallel_size=self._tensor_parallel_size,
            gpu_memory_utilization=self._gpu_memory_utilization,
        )
        if self._max_model_len is not None:
            llm_kwargs["max_model_len"] = self._max_model_len
        llm = LLM(**llm_kwargs)
        # Resolve actual max_model_len (auto-detected by vLLM when not set)
        self._max_model_len = llm.llm_engine.model_config.max_model_len
        self._tokenizer = llm.get_tokenizer()

        # Resolve random heads if requested
        heads_arg = self._heads_path
        if heads_arg == "random":
            from transformers import AutoConfig

            from locos_eval.retrieval_heads import generate_random_heads

            config = AutoConfig.from_pretrained(self._model_name)
            num_layers, num_attn_heads = _extract_layer_head_counts(config, self._model_name)
            num_heads_arg = self._ablation_kwargs.get("num_heads", 50)
            seed = self._ablation_kwargs.get("random_seed", 42)
            heads_arg = generate_random_heads(
                num_layers=num_layers,
                num_heads=num_attn_heads,
                count=num_heads_arg,
                seed=seed,
            )
            console.print(f"[bold]Random heads:[/bold] {len(heads_arg)} heads (seed={seed})")

        # Filter out runner-only kwargs before passing to ablation()
        _runner_only_keys = {
            "heads_label",
            "random_seed",
            "sampling_seed",
            "num_calibration",
            "enforce_eager",
        }
        ablation_kwargs = {k: v for k, v in self._ablation_kwargs.items() if k not in _runner_only_keys}

        # Mean ablation: format calibration prompts now that the tokenizer is
        # loaded, then hand them to ablation() so the wrapper runs the
        # calibration pass before installing replacement hooks.
        if ablation_kwargs.get("decoding") == "ablation" and ablation_kwargs.get("ablation_mode") == "mean":
            cal_samples = getattr(self, "_calibration_samples", None)
            if not cal_samples:
                raise RuntimeError(
                    "ablation_mode='mean' requires calibration samples; "
                    "run() must populate self._calibration_samples before _init_model()"
                )
            calibration_prompts = [self._format_prompt(s) for s in cal_samples]
            console.print(
                f"  [cyan]Mean-ablation calibration: {len(calibration_prompts)} "
                f"chat-formatted prompts from samples[:{len(calibration_prompts)}][/cyan]"
            )
            ablation_kwargs["calibration_prompts"] = calibration_prompts

        self._wrapper = ablation(
            llm,
            heads=heads_arg,
            **ablation_kwargs,
        )

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def _format_prompt(self, sample: EvalSample) -> str:
        """Build a chat-formatted prompt string from the sample.

        Uses ``tokenizer.apply_chat_template`` when available, otherwise
        falls back to a plain ``Role: content`` format.

        For tokenizers whose chat template exposes an ``enable_thinking``
        Jinja variable (Qwen3, gpt-oss, …) we explicitly pass
        ``enable_thinking=True`` rather than relying on the template's
        default. Reasoning models behave very differently with thinking
        on vs off, and we want the eval pipeline to lock that knob to
        ``True`` so a future template change can't silently flip it off.
        """
        messages: list[dict[str, str]] = []
        sys_msg = self.system_message()
        if sys_msg is not None:
            messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": sample.prompt})

        if hasattr(self._tokenizer, "apply_chat_template"):
            chat_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
            template = getattr(self._tokenizer, "chat_template", None) or ""
            if "enable_thinking" in template:
                chat_kwargs["enable_thinking"] = True
            return self._tokenizer.apply_chat_template(messages, **chat_kwargs)

        # Fallback: plain text format
        parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            parts.append(f"{role}: {msg['content']}")
        parts.append("")  # trailing newline
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def _heads_label(self) -> str:
        """Derive a short label from the heads filename.

        If ``--heads-label`` was provided via kwargs, that takes priority.
        Otherwise inferred from the filename suffix.

        Examples:
            ``retrieval_heads/Qwen3-32B.json``       → ``"niah"``
            ``retrieval_heads/Qwen3-32B_nolima.json`` → ``"nolima"``
            ``retrieval_heads/Qwen3-32B_cri.json``    → ``"cri"``
            ``--heads-label ori``                      → ``"ori"``
            ``None`` (greedy)                          → ``""``
            ``"random"`` (seed=42, count=20)           → ``"random_s42_n20"``
        """
        return self.experiment_key._compute_label()

    def _run_variant(self) -> str:
        """Build the variant directory name from decoding mode + heads label.

        Examples:
            greedy decoding                 → ``"greedy"``
            ablation with Wu NIAH heads     → ``"ablation_wu_niah"``
            ablation with random heads      → ``"ablation_random_s42_n20"``
        """
        return self.experiment_key.variant

    def _run_dir(self) -> Path:
        """Directory for this task/model/variant combination."""
        return self.experiment_key.local_dir(str(self._output_dir))

    def _generations_path(self) -> Path:
        """Path for the generations checkpoint file."""
        return self._run_dir() / "generations.jsonl"

    def _save_generation(self, path: Path, idx: int, output: str, sample: EvalSample) -> None:
        """Append a single generation to the checkpoint file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "sample_id": idx,
            "output": output,
            "target": sample.target,
            "metadata": sample.metadata,
        }
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_generations(self, path: Path) -> list[dict]:
        """Load previously saved generations from checkpoint."""
        if not path.exists():
            return []
        records = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def run(self, score_only: str | None = None) -> list[EvalResult]:
        """Execute the evaluation pipeline.

        Args:
            score_only: Path to a generations JSONL file. If provided, skip
                generation and run only the scoring phase.

        Steps:
            1. Load samples (+ apply limit).
            2. Phase 1: Generate outputs (with per-sample checkpointing),
               or load from ``score_only`` file.
            3. Phase 2: Score all outputs (batched).
            4. Print summary table.
            5. Save final results JSONL.

        Returns:
            List of :class:`EvalResult` instances.
        """
        samples = self.load_samples()
        assert len(samples) > 0, "load_samples() returned no samples"

        if self._limit is not None:
            samples = samples[: self._limit]
            console.print(f"[yellow]Limiting to {len(samples)} samples[/yellow]")

        if score_only is not None:
            # Score-only mode: load generations from file
            console.rule("[bold]Score-only mode[/bold]")
            gen_records = self._load_generations(Path(score_only))
            assert len(gen_records) > 0, f"No generations found in {score_only}"
            console.print(f"  Loaded {len(gen_records)} generations from {score_only}")

            outputs = [r["output"] for r in gen_records]
            # Use the samples list (truncated to match generation count)
            samples = samples[: len(gen_records)]
            # Restore metadata from checkpoint (may have been updated during generation)
            for sample, record in zip(samples, gen_records):
                sample.metadata.update(record.get("metadata", {}))
        else:
            # Phase 1: Generate with checkpointing
            gen_path = self._generations_path()
            existing = self._load_generations(gen_path)
            start_idx = len(existing)

            if start_idx >= len(samples):
                # All samples already generated — skip model loading
                console.rule("[bold]Phase 1: Generation (skipped — checkpoint complete)[/bold]")
                console.print(f"  [green]All {len(samples)} samples already generated[/green]")
                console.print(f"  Checkpoint: {gen_path}")
                outputs = [r["output"] for r in existing[: len(samples)]]
                # Restore metadata from checkpoint
                for sample, record in zip(samples, existing[: len(samples)]):
                    sample.metadata.update(record.get("metadata", {}))
            else:
                # Mean ablation needs calibration prompts formatted with the
                # tokenizer — we hand the raw sample subset to ``_init_model``
                # which loads the tokenizer first, then formats and runs
                # calibration before constructing the wrapper.
                self._calibration_samples = None
                if self._ablation_kwargs.get("decoding") == "ablation" and self._ablation_mode == "mean":
                    n_cal = min(self._num_calibration, len(samples))
                    self._calibration_samples = samples[:n_cal]

                self._init_model()
                console.rule("[bold]Phase 1: Generation[/bold]")

                if start_idx > 0:
                    console.print(
                        f"  [yellow]Resuming from sample {start_idx} "
                        f"({start_idx}/{len(samples)} already generated)[/yellow]"
                    )

                outputs: list[str] = [r["output"] for r in existing]

                sampling_seed = self._ablation_kwargs.get("sampling_seed")
                if sampling_seed is not None:
                    import torch

                    torch.manual_seed(sampling_seed)
                    console.print(f"  [dim]torch.manual_seed({sampling_seed})[/dim]")

                # Per-sample watchdog: if a single generate() call hangs (e.g. a deadlocked
                # collective_rpc on a dead worker), dump tracebacks and hard-exit so the
                # outer job script can restart and resume from checkpoint instead of
                # waiting indefinitely. Off by default; opt in with SAMPLE_TIMEOUT_S.
                sample_timeout_s = float(os.environ.get("SAMPLE_TIMEOUT_S", "0") or 0)

                remaining = samples[start_idx:]

                if getattr(self._wrapper, "supports_batch", False):
                    # Native vLLM pipeline (greedy / ablation): hand the whole
                    # remaining list to vLLM in a single call so its scheduler
                    # can continuously batch as many prompts as memory allows.
                    # FIXME: per-sample checkpointing is not possible during
                    # this call — if the job dies mid-generate, we lose all
                    # progress on `remaining` and resume from start_idx on
                    # restart. Acceptable trade for ablation/greedy throughput
                    # at long context; revisit with vLLM's streaming engine
                    # API if crash-resilience becomes important.
                    console.print(
                        f"  [cyan]Batched generation: {len(remaining)} prompts in one vLLM call "
                        f"(continuous batching)[/cyan]"
                    )
                    prompts = [self._format_prompt(s) for s in remaining]
                    batch_outputs = self._wrapper.generate(
                        prompts,
                        max_tokens=self._max_tokens,
                        temperature=self._temperature,
                        top_p=self._top_p,
                        top_k=self._top_k,
                    )
                    assert isinstance(batch_outputs, list) and len(batch_outputs) == len(remaining), (
                        f"batched wrapper.generate returned {type(batch_outputs).__name__} "
                        f"of length {len(batch_outputs) if hasattr(batch_outputs, '__len__') else '?'}, "
                        f"expected list of length {len(remaining)}"
                    )
                    for idx, (sample, output) in enumerate(zip(remaining, batch_outputs), start=start_idx):
                        outputs.append(output)
                        self._save_generation(gen_path, idx, output, sample)
                else:
                    # Manual per-sample loop (fallback for any wrapper that
                    # does not set supports_batch=True).
                    for idx, sample in enumerate(
                        track(remaining, description=f"Generating {self.task_name()}...", console=console),
                        start=start_idx,
                    ):
                        prompt = self._format_prompt(sample)
                        watchdog = _start_sample_watchdog(idx, sample, gen_path, sample_timeout_s, self)
                        try:
                            output = self._wrapper.generate(
                                prompt,
                                max_tokens=self._max_tokens,
                                temperature=self._temperature,
                                top_p=self._top_p,
                                top_k=self._top_k,
                            )
                        finally:
                            if watchdog is not None:
                                watchdog.cancel()
                        outputs.append(output)
                        self._save_generation(gen_path, idx, output, sample)

                console.print(f"  Generated {len(outputs)} outputs (checkpoint: {gen_path})")

        # Phase 2: Score all outputs
        console.rule("[bold]Phase 2: Scoring[/bold]")
        all_scores = self.score_all(outputs, samples)

        results: list[EvalResult] = []
        for idx, (sample, output, scores) in enumerate(zip(samples, outputs, all_scores)):
            result = EvalResult(
                sample_id=idx,
                output=output,
                target=sample.target,
                scores=scores,
                metadata=sample.metadata,
            )
            results.append(result)

        self._print_summary(results)

        # Save final results + config sidecar
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        decoding = self._ablation_kwargs.get("decoding", "greedy")
        run_dir = self._run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)

        out_path = run_dir / f"results_{timestamp}.jsonl"
        EvalResult.save_jsonl(results, out_path)

        config = {
            "task": self.task_name(),
            "model": self._model_name,
            "heads": self._heads_path,
            "heads_label": self._heads_label() if self._heads_path else None,
            "decoding": decoding,
            "variant": self._run_variant(),
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "max_model_len": self._max_model_len,
            "tensor_parallel_size": self._tensor_parallel_size,
            "gpu_memory_utilization": self._gpu_memory_utilization,
            "sampling_seed": self._ablation_kwargs.get("sampling_seed"),
            "limit": self._limit,
            "timestamp": timestamp,
            **{k: v for k, v in self._ablation_kwargs.items() if k not in ("decoding", "sampling_seed")},
        }
        config_path = run_dir / f"results_{timestamp}_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        console.print(f"\n[green]Results saved to:[/green] {out_path}")
        console.print(f"[green]Config  saved to:[/green] {config_path}")

        return results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self, results: list[EvalResult]) -> None:
        """Print a rich table with the mean of each score metric."""
        if not results:
            return

        # Collect all metric keys across results
        all_keys: list[str] = []
        for r in results:
            for k in r.scores:
                if k not in all_keys:
                    all_keys.append(k)

        table = Table(title=f"{self.task_name()} Summary (n={len(results)})")
        table.add_column("Metric", style="bold")
        table.add_column("Mean", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")

        for key in all_keys:
            values = [r.scores[key] for r in results if key in r.scores]
            if values:
                mean_val = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                table.add_row(key, f"{mean_val:.4f}", f"{min_val:.4f}", f"{max_val:.4f}")

        console.print(table)

    # ------------------------------------------------------------------
    # CLI argument helper
    # ------------------------------------------------------------------

    @staticmethod
    def add_common_args(parser: argparse.ArgumentParser) -> None:
        """Add standard CLI arguments shared by all eval tasks."""
        parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
        parser.add_argument(
            "--heads",
            type=str,
            default=None,
            help="Path to retrieval heads JSON file, or 'random' for random heads "
            "(required for --decoding ablation)",
        )
        parser.add_argument(
            "--model-config",
            type=str,
            default=None,
            help="Path to a YAML file with per-model defaults (auto-discovered from "
            "evals/configs/{ModelName}.yaml when omitted)",
        )
        parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens to generate per sample")
        parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (0 = greedy)")
        parser.add_argument(
            "--sampling-top-p", type=float, default=None, help="Top-p (nucleus) sampling threshold (1.0 = disabled)"
        )
        parser.add_argument("--sampling-top-k", type=int, default=None, help="Top-k sampling (-1 = disabled)")
        parser.add_argument(
            "--max-model-len",
            type=int,
            default=None,
            help="Maximum model context length (auto-detected from model config if omitted)",
        )
        parser.add_argument("--tp", type=int, default=None, help="Tensor parallel size")
        parser.add_argument("--gpu-mem", type=float, default=None, help="GPU memory utilization (0, 1]")
        parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (for debugging)")
        parser.add_argument("--output-dir", type=str, default="eval_results", help="Directory for result files")
        parser.add_argument(
            "--decoding", type=str, default="greedy", choices=["greedy", "ablation"], help="Decoding strategy"
        )
        parser.add_argument(
            "--ablation-mode",
            type=str,
            default="zero",
            choices=["zero", "mean"],
            help="Ablation replacement strategy when --decoding ablation. "
            "'zero' writes 0; 'mean' writes a per-(layer, head) mean q activation captured "
            "by a calibration pass (matches locos nolima_ablation).",
        )
        parser.add_argument(
            "--num-calibration",
            type=int,
            default=50,
            help="Number of samples used for mean-ablation calibration (default: 50). "
            "Calibration prompts are taken from the first N loaded samples and fed once "
            "through the model with max_tokens=1.",
        )
        eager = parser.add_mutually_exclusive_group()
        eager.add_argument(
            "--enforce-eager",
            dest="enforce_eager",
            action="store_true",
            default=True,
            help="Pass enforce_eager=True to vLLM (default). Required for --decoding ablation.",
        )
        eager.add_argument(
            "--no-enforce-eager",
            dest="enforce_eager",
            action="store_false",
            help="Allow vLLM to torch.compile + capture CUDA graphs. Only safe for "
            "--decoding greedy; on --decoding ablation it is silently overridden "
            "back to True (compile freezes the unpatched attn.forward, producing "
            "greedy outputs).",
        )
        parser.add_argument(
            "--heads-label",
            type=str,
            default=None,
            help="Override heads label for output directory (e.g. 'ori', 'cri'). "
            "Auto-inferred from heads filename if not set.",
        )
        parser.add_argument(
            "--num-heads",
            type=int,
            default=None,
            help="Number of top-ranked heads to use (applies to both file-based and random heads)",
        )
        parser.add_argument("--random-seed", type=int, default=42, help="Random seed for --heads random")
        parser.add_argument(
            "--sampling-seed",
            type=int,
            default=None,
            help="Seed for reproducible stochastic sampling (appends _s{seed} to variant)",
        )
        parser.add_argument(
            "--score-only", type=str, default=None, help="Path to generations JSONL — skip generation, run scoring only"
        )

    @staticmethod
    def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
        """Merge per-model YAML config with CLI arguments.

        Layering (last wins):
            hardcoded defaults  <  _default.yaml  <  {Model}.yaml  <  --model-config  <  CLI args

        Call this after ``parser.parse_args()`` and before constructing the runner.
        """
        # Map CLI dest names → model_config keys (handles tp→tensor_parallel_size, gpu_mem→gpu_memory_utilization)
        _CLI_TO_CONFIG = {
            "max_tokens": "max_tokens",
            "temperature": "temperature",
            "sampling_top_p": "sampling_top_p",
            "sampling_top_k": "sampling_top_k",
            "max_model_len": "max_model_len",
            "tp": "tensor_parallel_size",
            "gpu_mem": "gpu_memory_utilization",
        }

        model_cfg = load_model_config(args.model, getattr(args, "model_config", None))

        for cli_dest, cfg_key in _CLI_TO_CONFIG.items():
            cli_val = getattr(args, cli_dest, None)
            if cli_val is not None:
                # CLI was explicitly set — keep it
                continue
            # Fill from YAML / hardcoded default
            setattr(args, cli_dest, model_cfg.get(cfg_key, _CONFIG_DEFAULTS.get(cfg_key)))

        if args.decoding == "ablation" and args.num_heads is None:
            args.num_heads = 50

        return args
