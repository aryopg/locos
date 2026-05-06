#!/usr/bin/env bash
# DeCoRe Evaluation Runner (standalone — no Inspect AI)
#
# Usage:
#   ./scripts/eval/run_eval.sh <task_module> [extra args...]
#
# Examples:
#   ./scripts/eval/run_eval.sh nq_swap_task --limit 100
#   ./scripts/eval/run_eval.sh medrag_task --dataset-name medqa --limit 100
#   ./scripts/eval/run_eval.sh xsum_task --limit 50
#   ./scripts/eval/run_eval.sh aci_bench_task --n-shot 2 --judge-model claude-haiku-4-5-20251001
#
# Environment variables:
#   DECORE_MODEL                   HuggingFace model name (default: meta-llama/Meta-Llama-3-8B-Instruct)
#   DECORE_HEADS_JSON              Path to retrieval heads JSON
#   DECORE_MAX_MODEL_LEN           Max sequence length (default: 4096)
#   DECORE_TENSOR_PARALLEL_SIZE    Number of GPUs for tensor parallelism (default: 1)
#   DECORE_GPU_MEMORY_UTILIZATION  vLLM GPU memory utilization fraction
#                                  (when unset, the per-model YAML in
#                                  locos_eval/evals/configs/{Model}.yaml is
#                                  used; only set this to override per-run)
#   DECORE_DECODING                Decoding mode: decore, greedy, or ablation (default: decore)
#   DECORE_ABLATION_MODE           Ablation replacement strategy: zero or mean (default: zero;
#                                  only forwarded when DECORE_DECODING=ablation)
#   DECORE_NUM_HEADS               Number of top heads to use (ablation default: 50)
#   NUM_HEADS                      Alias for DECORE_NUM_HEADS from deploy launchers
#   DECORE_RANDOM_SEED             Random seed for --heads random (default: 42)
#   DECORE_OUTPUT_DIR              Output directory for results (default: eval_results)
#   DECORE_TEMPERATURE             Sampling temperature (default: 0.0)
#   DECORE_TOP_P                   Top-p (nucleus) sampling threshold (default: 1.0)
#   DECORE_TOP_K_SAMPLING          Top-k sampling (-1 = disabled) (default: -1)
#   DECORE_SAMPLING_SEED           Seed for reproducible stochastic sampling (default: unset)
#   DECORE_HEADS_LABEL             Override heads label for variant naming (default: unset)
#   DECORE_ENFORCE_EAGER           Force enforce_eager=true/false. When unset (default),
#                                  eager is enabled for --decoding decore and ablation
#                                  (both rely on monkey-patched attn.forward, which
#                                  torch.compile / CUDA graphs would bypass) and disabled
#                                  for greedy so vLLM can torch.compile + capture CUDA
#                                  graphs. Setting DECORE_ENFORCE_EAGER=false on an
#                                  ablation run is a known footgun (silently produces
#                                  greedy outputs) — the script overrides it back to true
#                                  with a warning.
set -euo pipefail

MODEL="${DECORE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
HEADS="${DECORE_HEADS_JSON:-retrieval_heads/Meta-Llama-3-8B-Instruct.json}"
MAX_MODEL_LEN="${DECORE_MAX_MODEL_LEN:-4096}"
TP_SIZE="${DECORE_TENSOR_PARALLEL_SIZE:-1}"
GPU_MEM_UTIL="${DECORE_GPU_MEMORY_UTILIZATION:-}"
DECODING="${DECORE_DECODING:-decore}"
ABLATION_MODE="${DECORE_ABLATION_MODE:-zero}"
NUM_HEADS="${DECORE_NUM_HEADS:-${NUM_HEADS:-}}"
RANDOM_SEED="${DECORE_RANDOM_SEED:-42}"
OUTPUT_DIR="${DECORE_OUTPUT_DIR:-eval_results}"
TEMPERATURE="${DECORE_TEMPERATURE:-0.0}"
TOP_P="${DECORE_TOP_P:-1.0}"
TOP_K_SAMPLING="${DECORE_TOP_K_SAMPLING:--1}"
SAMPLING_SEED="${DECORE_SAMPLING_SEED:-}"
HEADS_LABEL="${DECORE_HEADS_LABEL:-}"
TASK="${1:?Usage: run_eval.sh <task_module> [args...]}"
shift

NUM_HEADS_FLAG=""
[[ "$DECODING" == "ablation" && -z "$NUM_HEADS" ]] && NUM_HEADS="50"
[[ -n "$NUM_HEADS" ]] && NUM_HEADS_FLAG="--num-heads ${NUM_HEADS}"

# --ablation-mode is only meaningful for --decoding ablation; the runner
# rejects mean+non-ablation, so only forward when actually ablating.
ABLATION_MODE_FLAG=""
[[ "$DECODING" == "ablation" ]] && ABLATION_MODE_FLAG="--ablation-mode ${ABLATION_MODE}"

RANDOM_SEED_FLAG=""
[[ "$HEADS" == "random" ]] && RANDOM_SEED_FLAG="--random-seed ${RANDOM_SEED}"

SAMPLING_SEED_FLAG=""
[[ -n "$SAMPLING_SEED" ]] && SAMPLING_SEED_FLAG="--sampling-seed ${SAMPLING_SEED}"

HEADS_LABEL_FLAG=""
[[ -n "$HEADS_LABEL" ]] && HEADS_LABEL_FLAG="--heads-label ${HEADS_LABEL}"

# Only pass --gpu-mem when explicitly set so the per-model YAML defaults take
# effect. DeCoRe / ablation bypasses vLLM's paged KV cache, and the right
# fraction depends on per-GPU model size — see evals/configs/{Model}.yaml.
GPU_MEM_FLAG=""
[[ -n "$GPU_MEM_UTIL" ]] && GPU_MEM_FLAG="--gpu-mem ${GPU_MEM_UTIL}"

# Eager mode gating. Both the manual DeCoRe contrastive loop and the native
# ablation path rely on monkey-patched attn.forward (instance-attribute
# replacement). torch.compile / CUDA-graph capture freezes the original
# forward at engine init, so a non-eager ablation run silently bypasses the
# q-replacement and produces outputs identical to greedy. Greedy itself does
# no patching and is safe to compile.
#
# Auto-policy: greedy → no-eager (compile speedup); decore/ablation → eager.
# DECORE_ENFORCE_EAGER overrides the policy, with one guardrail: an explicit
# DECORE_ENFORCE_EAGER=false on an ablation run is a known footgun, so we
# override it back to --enforce-eager and warn loudly.
ENFORCE_EAGER_FLAG=""
ENFORCE_EAGER_OVERRIDE="${DECORE_ENFORCE_EAGER:-}"
if [[ -n "$ENFORCE_EAGER_OVERRIDE" ]]; then
    case "$ENFORCE_EAGER_OVERRIDE" in
        true|1|yes)  ENFORCE_EAGER_FLAG="--enforce-eager" ;;
        false|0|no)
            if [[ "$DECODING" == "ablation" ]]; then
                echo "WARNING: DECORE_ENFORCE_EAGER=$ENFORCE_EAGER_OVERRIDE on --decoding ablation is unsafe " \
                     "(torch.compile bypasses our attn.forward patch, silently producing greedy outputs). " \
                     "Overriding back to --enforce-eager." >&2
                ENFORCE_EAGER_FLAG="--enforce-eager"
            else
                ENFORCE_EAGER_FLAG="--no-enforce-eager"
            fi
            ;;
        *) echo "ERROR: DECORE_ENFORCE_EAGER must be true/false, got '$ENFORCE_EAGER_OVERRIDE'" >&2; exit 1 ;;
    esac
elif [[ "$DECODING" == "greedy" ]]; then
    ENFORCE_EAGER_FLAG="--no-enforce-eager"
else
    # decore + ablation → eager (both rely on monkey-patched attn.forward)
    ENFORCE_EAGER_FLAG="--enforce-eager"
fi

exec python -m "locos_eval.evals.tasks.${TASK}" \
    --model "${MODEL}" \
    --heads "${HEADS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tp "${TP_SIZE}" \
    ${GPU_MEM_FLAG} \
    --decoding "${DECODING}" \
    --output-dir "${OUTPUT_DIR}" \
    --temperature "${TEMPERATURE}" \
    --sampling-top-p "${TOP_P}" \
    --sampling-top-k "${TOP_K_SAMPLING}" \
    ${NUM_HEADS_FLAG} \
    ${ABLATION_MODE_FLAG} \
    ${RANDOM_SEED_FLAG} \
    ${SAMPLING_SEED_FLAG} \
    ${HEADS_LABEL_FLAG} \
    ${ENFORCE_EAGER_FLAG} \
    "$@"
