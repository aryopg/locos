#!/usr/bin/env bash
# Legacy eval launcher — launches evaluation jobs via kblaunch for all
# model/task combinations. For the generic multi-namespace launcher that
# accepts arbitrary job scripts, see launch_multi_ns.sh.
#
# Usage:
#   ./deploy/launch_evals.sh                    # launch all jobs
#   ./deploy/launch_evals.sh --dry-run          # print commands without launching
#   ./deploy/launch_evals.sh --limit 100        # limit each eval to 100 samples
#   ./deploy/launch_evals.sh --limit 100 --dry-run
#   MODELS="Qwen/Qwen3.5-2B" ./deploy/launch_evals.sh  # override model list
#
# Prerequisites:
#   - kblaunch installed and configured (kblaunch setup)
#   - Retrieval heads JSON files committed to the repo for each model
#   - Kubernetes secrets (aryo-secrets) configured with HF_TOKEN, GIT_TOKEN, etc.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Shared infrastructure (model registry, docker image, setup helpers)
# ---------------------------------------------------------------------------

source "${SCRIPT_DIR}/job_config.sh"

# ---------------------------------------------------------------------------
# Eval-specific configuration
# ---------------------------------------------------------------------------

# Eval tasks (standalone — no Inspect AI)
# MedRAG is split per sub-dataset for independent logging/reporting.
# Format: "task_module [extra_args...]"
TASKS=(
    "nq_swap_task"
    "xsum_task"
    "medrag_task --dataset-name mmlu_med"
    "medrag_task --dataset-name medqa"
    "medrag_task --dataset-name medmcqa"
    "medrag_task --dataset-name pubmedqa"
    "medrag_task --dataset-name supergpqa_med"
    "aci_bench_task --n-shot 2 --judge-model claude-haiku-4-5-20251001"
)

# Max sequence length per task (override if needed)
MAX_MODEL_LEN="${DECORE_MAX_MODEL_LEN:-4096}"

# Derive task slug from task entry (e.g. "medrag_task --dataset-name medqa" → "medrag-medqa")
task_slug() {
    local entry="$1"
    local module slug dataset
    module=$(echo "$entry" | awk '{print $1}')
    slug=$(echo "$module" | sed 's/_task$//' | tr '_' '-')
    dataset=$(echo "$entry" | sed -n 's/.*--dataset-name[[:space:]][[:space:]]*\([^[:space:]][^[:space:]]*\).*/\1/p')
    if [[ -n "$dataset" ]]; then
        slug="${slug}-$(echo "$dataset" | tr '_' '-')"
    fi
    echo "$slug"
}

# Decoding modes to evaluate (override with DECODINGS env var, space-separated).
# Default: "decore" only. Set to "decore greedy ablation" for full comparison.
if [[ -n "${DECODINGS:-}" ]]; then
    IFS=' ' read -ra DECODING_LIST <<< "$DECODINGS"
else
    DECODING_LIST=("decore")
fi

# Build the full pod command for a given model/task/decoding combination.
# Usage: build_job_command <model> <heads> <gpus> <task_entry> <decoding> <limit_flag>
build_job_command() {
    local model="$1" heads="$2" gpus="$3" task_entry="$4" decoding="$5" limit_flag="${6:-}"
    local task_module task_extra_args tslug

    task_module=$(echo "$task_entry" | awk '{print $1}')
    task_extra_args=$(echo "$task_entry" | cut -s -d' ' -f2-)
    tslug=$(task_slug "$task_entry")
    local mslug
    mslug=$(model_slug "$model")

    cat <<EOF
$(setup_commands)

echo '=== Running ${task_entry} with ${model} (${decoding}) ==='

DECORE_MODEL='${model}' \\
DECORE_HEADS_JSON='${heads}' \\
DECORE_MAX_MODEL_LEN='${MAX_MODEL_LEN}' \\
DECORE_TENSOR_PARALLEL_SIZE='${gpus}' \\
DECORE_DECODING='${decoding}' \\
./scripts/eval/run_eval.sh ${task_module} ${task_extra_args} ${limit_flag}

echo '=== Uploading results ==='
python scripts/upload_results.py ./logs --repo-id '${HF_RESULTS_REPO}' --path-in-repo '${mslug}/${tslug}'

echo '=== Done ==='
EOF
}

GPU_PRODUCT="${GPU_PRODUCT:-NVIDIA-H100-80GB-HBM3}"

if [[ -n "${MODELS:-}" ]]; then
    IFS=' ' read -ra MODEL_LIST <<< "$MODELS"
else
    MODEL_LIST=("${DEFAULT_MODELS[@]}")
fi

# Parse arguments
DRY_RUN=false
LIMIT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        --limit)    LIMIT="$2"; shift 2 ;;
        *)          echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Build the --limit flag (empty string if not set)
LIMIT_FLAG=""
[[ -n "$LIMIT" ]] && LIMIT_FLAG="--limit ${LIMIT}"

# ---------------------------------------------------------------------------
# Launch jobs
# ---------------------------------------------------------------------------

job_count=0

for model in "${MODEL_LIST[@]}"; do
    slug=$(job_slug "$model")
    heads=$(heads_path "$model")
    gpus=$(gpu_count_for_model "$model")

    for decoding in "${DECODING_LIST[@]}"; do
        for task_entry in "${TASKS[@]}"; do
            tslug=$(task_slug "$task_entry")
            # Include decoding mode in job name for uniqueness
            if [[ "$decoding" == "decore" ]]; then
                job_name="decore-${slug}-${tslug}"
            else
                job_name="${decoding}-${slug}-${tslug}"
            fi

            # Greedy mode doesn't need heads
            local_heads="$heads"
            [[ "$decoding" == "greedy" ]] && local_heads=""

            run_cmd=$(build_job_command "$model" "$local_heads" "$gpus" "$task_entry" "$decoding" "$LIMIT_FLAG")

            if [[ "$DRY_RUN" == true ]]; then
                echo "--- [DRY RUN] ${job_name} (${gpus}x GPU, TP=${gpus}) ---"
                echo "  Model:    ${model}"
                echo "  Task:     ${task_entry}"
                echo "  Decoding: ${decoding}"
                echo "  Heads:    ${local_heads:-N/A}"
                echo "  TP:       ${gpus}"
                [[ -n "$LIMIT" ]] && echo "  Limit:    ${LIMIT} samples"
                echo ""
            else
                echo "Launching: ${job_name} (${gpus}x GPU, ${model}, ${task_entry}, ${decoding})"
                kblaunch launch \
                    --job-name "${job_name}" \
                    --docker-image "${DOCKER_IMAGE}" \
                    --gpu-limit "${gpus}" \
                    --gpu-product "${GPU_PRODUCT}" \
                    --cpu-request 32 \
                    --ram-request 256Gi \
                    --command "${run_cmd}" \
                    --secrets-env-vars aryo-secrets
            fi

            job_count=$((job_count + 1))
        done
    done
done

echo ""
echo "Total jobs: ${job_count}"
[[ "$DRY_RUN" == true ]] && echo "(dry run — no jobs were launched)"
