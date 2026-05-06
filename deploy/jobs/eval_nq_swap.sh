#!/usr/bin/env bash
# Eval: NQ-Swap context faithfulness (sub_EM, org_EM)
set -euo pipefail

DECODING="${DECODING:-ablation}"
ABLATION_MODE="${ABLATION_MODE:-zero}"

echo "=== NQ-Swap eval: ${MODEL} (${DECODING}) ==="

# Download retrieval heads from HF if not present locally
ensure_heads "${HEADS:-}"

# Skip if already complete on HF
# Exit codes: 0=skip, 1=run, 2=check failed (abort to avoid blind re-runs)
if [[ "${FORCE:-}" != "true" ]]; then
    check_exit=0
    python scripts/check_experiment.py \
        --repo-id "${HF_DOWNSTREAM_REPO}" \
        --task nq_swap \
        --model "${MODEL}" \
        --decoding "${DECODING}" \
        --heads "${HEADS:-}" \
        --heads-label "${HEADS_LABEL:-}" \
        --sampling-seed "${SAMPLING_SEED:-}" \
        --ablation-mode "${ABLATION_MODE}" || check_exit=$?
    if [[ $check_exit -eq 0 ]]; then
        echo "=== SKIP: already complete on HF ==="
        exit 0
    elif [[ $check_exit -ge 2 ]]; then
        echo "ERROR: check_experiment.py failed (exit $check_exit). Aborting to avoid blind re-run." >&2
        exit 1
    fi
    # check_exit=1 → needs running, continue
fi

MODEL="${MODEL}" \
HEADS_JSON="${HEADS}" \
DECODING="${DECODING}" \
ABLATION_MODE="${ABLATION_MODE}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}" \
TENSOR_PARALLEL_SIZE="${GPUS}" \
TEMPERATURE="${TEMPERATURE:-0.0}" \
TOP_P="${TOP_P:-1.0}" \
TOP_K_SAMPLING="${TOP_K_SAMPLING:--1}" \
SAMPLING_SEED="${SAMPLING_SEED:-}" \
HEADS_LABEL="${HEADS_LABEL:-}" \
./scripts/eval/run_eval.sh nq_swap_task ${LIMIT:+--limit $LIMIT}

echo "=== Syncing results to HF ==="
python scripts/sync_results.py \
    --repo-id "${HF_DOWNSTREAM_REPO}" \
    --local-dir ./eval_results || echo "WARNING: sync_results.py failed — results saved locally but NOT uploaded to HF" >&2
