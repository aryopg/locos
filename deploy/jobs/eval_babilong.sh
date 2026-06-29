#!/usr/bin/env bash
# Eval: BABILong free-form QA (subsets qa2, qa3 by default; split 0k)
#
# Iterates the requested subsets and runs each as a separate task. Skip-if-done
# and per-subset HF sync mirror eval_medrag.sh.
set -euo pipefail

DECODING="${DECODING:-ablation}"
ABLATION_MODE="${ABLATION_MODE:-zero}"
SUBSETS="${SUBSETS:-qa2 qa3}"
SPLIT="${SPLIT:-0k}"
OUTPUT_DIR="${OUTPUT_DIR:-downstream_results}"
HF_EVAL_PREFIX="${HF_EVAL_PREFIX:-${OUTPUT_DIR}}"
export HF_EVAL_PREFIX

echo "=== BABILong eval: ${MODEL} (${DECODING}, subsets='${SUBSETS}', split=${SPLIT}) ==="
echo "=== Output dir: ${OUTPUT_DIR} | HF prefix: ${HF_EVAL_PREFIX} ==="

# Download retrieval heads from HF if not present locally
ensure_heads "${HEADS:-}"

for subset in ${SUBSETS}; do
    TASK_NAME="babilong_${subset}_${SPLIT}"
    echo "=== BABILong subset: ${subset} (split=${SPLIT}) ==="

    # Skip if already complete on HF
    # Exit codes: 0=skip, 1=run, 2=check failed (abort to avoid blind re-runs)
    if [[ "${FORCE:-}" != "true" ]]; then
        check_exit=0
        python scripts/check_experiment.py \
            --repo-id "${HF_RESULTS_REPO}" \
            --task "${TASK_NAME}" \
            --model "${MODEL}" \
            --decoding "${DECODING}" \
            --heads "${HEADS:-}" \
            --heads-label "${HEADS_LABEL:-}" \
            --sampling-seed "${SAMPLING_SEED:-}" \
            --ablation-mode "${ABLATION_MODE}" \
            --hf-prefix "${HF_EVAL_PREFIX}" || check_exit=$?
        if [[ $check_exit -eq 0 ]]; then
            echo "=== SKIP: ${TASK_NAME} already complete ==="
            continue
        elif [[ $check_exit -ge 2 ]]; then
            echo "ERROR: check_experiment.py failed (exit $check_exit) for ${TASK_NAME}. Aborting." >&2
            exit 1
        fi
    fi

    MODEL="${MODEL}" \
    HEADS_JSON="${HEADS}" \
    DECODING="${DECODING}" \
    ABLATION_MODE="${ABLATION_MODE}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}" \
    TENSOR_PARALLEL_SIZE="${GPUS}" \
    TEMPERATURE="${TEMPERATURE:-0.0}" \
    TOP_P="${TOP_P:-1.0}" \
    TOP_K_SAMPLING="${TOP_K_SAMPLING:--1}" \
    SAMPLING_SEED="${SAMPLING_SEED:-}" \
    HEADS_LABEL="${HEADS_LABEL:-}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    ./scripts/eval/run_eval.sh babilong_task \
        --subset "${subset}" \
        --split "${SPLIT}" \
        ${LIMIT:+--limit $LIMIT}

    echo "=== Syncing results (${subset}, prefix=${HF_EVAL_PREFIX}) ==="
    python scripts/sync_results.py \
        --repo-id "${HF_RESULTS_REPO}" \
        --local-dir "./${OUTPUT_DIR}" \
        --hf-prefix "${HF_EVAL_PREFIX}" || echo "WARNING: sync_results.py failed for ${subset} — results saved locally but NOT uploaded to HF" >&2
done
