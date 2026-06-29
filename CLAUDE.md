# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Run unit tests (no GPU needed):
    source .venv/bin/activate && pytest tests/ -v -m "not gpu"

Run GPU integration tests (requires GPU server):
    TEST_MODEL=meta-llama/Meta-Llama-3-8B-Instruct \
    HEADS_JSON=retrieval_heads/Meta-Llama-3-8B-Instruct.json \
    pytest tests/ -v -m gpu

Install (local, macOS):
    uv venv --python /path/to/python3.11 --system-site-packages
    uv pip install -e ".[dev]" --python .venv/bin/python --no-deps
    uv pip install pytest pytest-mock pytest-asyncio pyyaml rich --python .venv/bin/python

Install (GPU server, CUDA 12.8):
    uv venv --python python3.12 --system-site-packages
    uv pip install -e ".[dev,eval]" \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --index-strategy unsafe-best-match \
        --python .venv/bin/python

Lint & format (run before every commit):
    source .venv/bin/activate
    ruff check --fix locos_eval/ locos/ scripts/ tests/
    black locos_eval/ locos/ scripts/ tests/
    isort locos_eval/ locos/ scripts/ tests/
    vulture locos_eval/ locos/ vulture_whitelist.py --min-confidence 80

Pre-commit hooks (installed automatically, runs on git commit):
    pre-commit install          # one-time setup
    pre-commit run --all-files  # manual full check

## Architecture

This repository contains two packages:
- `locos/` — LOCOS detector and baselines, ablation analysis
- `locos_eval/` — downstream eval framework (greedy vs ablation comparisons) + shared utilities

### Wrapper API (wrapper.py)

  ablation(llm, heads=..., decoding="greedy"|"ablation") → GreedyWrapper | AblationWrapper | AblationRPCWrapper
  - Accepts existing vLLM LLM instance
  - decoding="greedy": uses vLLM's native generation (no patching)
  - decoding="ablation" + TP=1: AblationWrapper (native q-zero patch via ablation.py hooks)
  - decoding="ablation" + TP>1: AblationRPCWrapper (RPC-dispatched q-zero hooks via rpc_ops.py)
  - heads=None required only for greedy mode

### Ablation patching (ablation.py)

  Native zero/mean ablation via PyTorch forward hooks on attention's qkv_proj.
  - patch_model_for_ablation(): installs per-head q-zero or q-mean replacement hooks
  - calibrate_mean_q_activations(): drives a calibration pass to compute mean Q activations
  - Does NOT use sequential KV caches — runs on vLLM's paged attention infrastructure

### Attention module (attention.py)

  Provides get_decoder_layers() and _get_supported_attention_classes() used by ablation.py.
  Also contains patch_model_attention_layers() / unpatch_model_attention_layers() for the
  sequential KV cache approach (still present, used by tests).
  Supported classes: LlamaAttention, Gemma3Attention, Qwen3Attention, Olmo2Attention.

### State (state.py)

  AblationState tracks active/masked_pass_active flags, retrieval head config, and two
  independent sequential KV caches (_base_kv, _masked_kv). Used by attention.py tests.

### Multi-GPU ablation (rpc_ops.py)

  Module-level RPC functions dispatched via llm.collective_rpc():
  - rpc_install_ablation_hooks / rpc_uninstall_ablation_hooks: zero-ablation for TP>1
  - rpc_install_q_capture_hooks / rpc_calibrate_and_install_mean_ablation: mean-ablation for TP>1
  - _remap_heads_for_tp: maps global head indices to local shard indices per TP rank

### Evaluation (evals/)

  Standalone eval framework — compares greedy vs ablation decoding.
  - runner.py: EvalRunner base class (model loading, generation loop, scoring, JSONL output)
  - scorers.py: pure scoring functions (ROUGE-L, BERTScore, MCQ extraction,
    subspan match, LLM judge via Anthropic SDK)
  - tasks/nq_swap_task.py: NQ-Swap context faithfulness (sub_EM, org_EM)
  - tasks/medrag_task.py: MedRAG medical QA (MCQ accuracy, 5 sub-datasets)
  - tasks/xsum_task.py: XSum summarization (ROUGE-L, BERTScore, FactKB)
  - tasks/aci_bench_task.py: ACI-Bench D2N dialogue-to-note
    (ROUGE-L, BERTScore, LLM-judge with Anthropic SDK)
  - tasks/longbench_v2_task.py: LongBench-v2 long-context MCQ
  - Generation checkpointing: outputs saved per-sample, auto-resumes on restart
  - Score-only mode: --score-only <generations.jsonl> reruns scoring without GPU
  - Prompts stored in evals/prompts/*.yaml (editable without code changes)
  - Decoding modes: "greedy" (default) and "ablation" (single pass with heads zeroed)
  - Random heads: --heads random --num-heads 50 --random-seed 42
  - experiment_key.py: ExperimentKey dataclass — single source of truth for naming
    - CLI: python -m locos_eval.evals.experiment_key --variant/--key/--model-slug/--local-dir
    - Used by runner.py, sync scripts, and shell job scripts
  - manifest.py: ExperimentManifest — per-experiment run tracking (JSON per variant dir)
  Model config (model_config.py + evals/configs/):
    YAML-based per-model sampling and hardware defaults with layered merge:
    - Hardcoded DEFAULTS < configs/_default.yaml < configs/{ModelName}.yaml < --model-config CLI < CLI args
    - _default.yaml: global defaults (temperature, top_p, top_k, max_tokens, tp, gpu_mem)
    - Per-model files: Qwen3-{4,8,14,32}B.yaml, gemma-3-{4,12,27}b-it.yaml,
      gemma-4-31B-it.yaml, gpt-oss-{20,120}b.yaml, Meta-Llama-3-8B-Instruct.yaml
    - null in model YAML resets to hardcoded disabled default (e.g. sampling_top_p: null → 1.0)
    - EvalRunner.resolve_args() maps CLI aliases (--tp, --gpu-mem) to config keys
  Run: python -m locos_eval.evals.tasks.<task> --model ... --heads ... --tp 4

Deploy infrastructure (deploy/):
  Local experiment job helpers.
  - job_config.sh: shared model registry (MODEL_SHORT → HF name), GPU count helpers,
    and reusable setup commands
  - model_sampling.yaml: per-model recommended sampling parameters for stochastic runs
  - jobs/: per-task/detector shell scripts (eval_nq_swap.sh, eval_medrag.sh,
    eval_xsum.sh, eval_aci_bench.sh, eval_longbench_v2.sh, detect_retrieval_heads.sh,
    detect_contrastive.sh, detect_cri.sh, detect_logit_contrib.sh,
    detect_attention_spatial.sh, detect_headkv.sh, ablation_nolima.sh,
    ablation_nolima_random.sh, ablation_parametric.sh, ablation_parametric_random.sh)
    Each integrates skip-if-done (check_experiment.py) and auto-sync (sync_results.py)

Retrieval head detection (locos/):
  Standalone package for identifying retrieval heads in transformer models.
  Organized into subpackages:
    detectors/: detection method implementations
    - detectors/behavioral.py: architecture-agnostic retrieval head detection
      using standard HF transformers (output_attentions=True, eager attention).
      Faithful reimplementation of Wu et al. / nightdessert/Retrieval_Head.
      Supports --dataset {niah,nolima} for different probing datasets.
    - detectors/contrastive.py: contrastive attention-based retrieval head detection.
    - detectors/cri.py: Causal Retrieval Importance via activation patching.
      Uses o_proj pre-hooks to capture/patch per-head activations.
      Teacher-forced evaluation (H+2 forward passes per example).
      Supports --dataset {niah,nolima}, --corruption {remove,scramble}.
    - detectors/logit_contrib.py: LOCOS — logit-contribution scoring via spatial contrast.
      Per-position formula: φ_{t,j} = α_{t,j} · u_{y_t}^T W_O^{(l,h)} v_j^{(l,h)}.
      Supports chat templates, thinking token detection/stripping, stop token
      handling, prompt suffix, per-model YAML configs, and dynamic KV caches.
    - detectors/attention_spatial.py: attention-only spatial-contrast baseline.
      Same needle-vs-off-needle contrast as logit_contrib but drops the OV
      projection (φ = α, not φ = α · u^T W_O v). Reviewer-requested controlled
      ablation isolating the OV term's contribution to LOCOS's score.
    - detectors/headkv.py: HeadKV/SnapKV-style anchor-window attention.
      Score: S = max over last K prompt tokens of (sum α to needle), per
      head. No spatial contrast, no generation, no ROUGE gate.
    utils/: shared utilities
    - utils/datasets.py: shared dataset abstraction (RetrievalTrial dataclass,
      builders for NIAH and NoLiMa, stratified sampling, NoLiMa download).
    - utils/common.py: checkpoint save/load, model loading, config extraction.
    - utils/needle_utils.py: needle insertion & position tracking.
    - utils/model_utils.py: model introspection helpers — get_decoder_layers()
      (Llama/Qwen/Gemma/VLM layouts), get_stop_token_ids(),
      format_prompt_with_chat_template(), detect_thinking_tokens(),
      strip_thinking_content(), tokenizer_adds_bos(),
      set_model_attn_impl()/get_model_attn_impl().
    analysis/: validation & analysis scripts
    - analysis/compare_heads.py: compare two retrieval head JSON files.
    - analysis/nolima_ablation.py: head ablation experiments (NoLiMa retrieval).
    - analysis/parametric_ablation.py: parametric/arithmetic ablation control experiment.
      Tests retrieval-head specificity: ablates heads and measures parametric recall
      (City-Country, PopQA) and arithmetic accuracy. Reuses ablation infrastructure
      from nolima_ablation.py. Results cached to ablation_parametric_results/.
    plotting/: visualization scripts (ablation_comparison.py, heatmap_comparison.py,
      logit_contrib_overview.py, nolima_ablation.py, parametric_ablation.py,
      score_buckets.py, score_dist.py)
    - download_haystack_data.py: download needle/haystack data from Retrieval_Head
      repo (NIAH) and Adobe Research NoLiMa repo. --dataset {niah,nolima,all}

Scripts (organized in scripts/):
  scripts/eval/:
    - run_eval.sh: convenience wrapper for standalone eval tasks
    - build_medrag_dataset.py: offline BM25 retrieval for MedRAG datasets
    - build_parametric_and_arithmetic_dataset.py: build parametric recall + arithmetic
      eval dataset (City-Country, PopQA, Arithmetic) and upload to HuggingFace
    - explore_acibench_results.py: Streamlit app for exploring ACI-Bench results
  scripts/ (root):
    - upload_results.py: upload result directories to HuggingFace Hub
    - sync_results.py: smart HF sync with per-experiment manifests
      (scans eval_results/, diffs against HF, uploads only new files)
    - check_experiment.py: skip-if-done for job scripts
      (exit 0 if experiment complete on HF, exit 1 otherwise)
    - download_heads.py: fetch retrieval heads JSON from HuggingFace Hub
      (--repo-id, --heads, --output; uses hf_hub_download with caching)

## Dependencies

Install groups (pyproject.toml):
  - [dev]: pytest, pytest-mock, pytest-asyncio, pre-commit, ruff, black, isort, vulture
  - [eval]: datasets, rouge-score, bert-score, tqdm
  - [medrag-build]: rank-bm25

## Key vLLM version notes

This code targets vLLM v0.18+.
LlamaAttention is in vllm.model_executor.models.llama.
Gemma3Attention is in vllm.model_executor.models.gemma3.
TP=1: model accessed via llm.apply_model(lambda m: m)
TP>1: model accessed via llm.collective_rpc() on each worker

Required env vars (set automatically in wrapper.py):
  VLLM_ENABLE_V1_MULTIPROCESSING=0  (lambdas can't be serialized across processes)

Required LLM kwargs (ablation mode):
  enforce_eager=True  (torch.compile freezes the unpatched attn.forward)
  gpu_memory_utilization=0.5  (default)

vLLM v0.18 attribute changes in LlamaAttention/Gemma3Attention:
  - layer_idx: removed (enumerate model.model.layers instead)
  - head_size → head_dim
  - scale → scaling

## Code style

- Linting: ruff (E/W/F/I/UP/B/SIM/RUF rules), line-length 120
- Formatting: black (line-length 120, py311)
- Import sorting: isort (black profile)
- Dead code: vulture (min-confidence 80, whitelist in vulture_whitelist.py)
- Pre-commit hooks enforce all four on every commit
- Add assertions for tensor shapes, parameter ranges, and state invariants
- Use rich for all user-facing script output (Console, Table, Panel, track)
- Store eval prompts in evals/prompts/*.yaml, not hardcoded in Python
- Leave FIXME comments for implementation doubts
- Write high-level doubts to docs/ markdown files
- Use locos_eval/utils/plotting.py conventions for all figures:
  - Call setup_plot_style() before creating figures
  - Use save_figure() to save — it auto-strips titles and saves legend separately
  - No titles inside plots (titles belong in LaTeX captions)
  - Default linewidth is 2 (LINE_WIDTH constant)
- Cannot run GPU tests locally — use mocks for unit tests

## Data & results

Retrieval heads JSON files, detection outputs, ablation analyses, downstream
eval results, and logs live on HuggingFace Hub at `aryopg/locos-results`
(`HF_RESULTS_REPO`). Downstream results are stored under the
`downstream_results/` prefix in the same repo. Use `scripts/download_heads.py`
to fetch heads locally. The `retrieval_heads/` directory in-repo was removed in
favor of HF storage.

## Known limitations

- Ablation runs vLLM's native paged attention with monkey-patched Q tensors.
  torch.compile / CUDA-graph capture must be disabled (enforce_eager=True) or
  the patch is bypassed silently.
- Only patches LlamaAttention, Gemma3Attention, Qwen3Attention, and Olmo2Attention
  (Olmo2Attention also serves Olmo 3); Mistral/Qwen2 need extension.
- Multi-GPU (tensor_parallel_size > 1) uses collective_rpc which adds per-hook
  RPC overhead at setup/teardown but NOT per-token overhead (generation goes
  through vLLM's native scheduler).
- macOS: GPU tests require a remote server. Unit tests run locally.
