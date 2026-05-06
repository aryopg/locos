#!/usr/bin/env bash
# Eval: ACI-Bench D2N dialogue-to-note
set -euo pipefail

DECODING="${DECODING:-decore}"
ABLATION_MODE="${ABLATION_MODE:-zero}"
N_SHOT="${N_SHOT:-2}"
JUDGE_MODEL="${JUDGE_MODEL:-claude-haiku-4-5-20251001}"

echo "=== ACI-Bench eval: ${MODEL} (${DECODING}) ==="

# Download retrieval heads from HF if not present locally
ensure_heads "${HEADS:-}"

# Skip if already complete on HF
# Exit codes: 0=skip, 1=run, 2=check failed (abort to avoid blind re-runs)
if [[ "${FORCE:-}" != "true" ]]; then
    check_exit=0
    python scripts/check_experiment.py \
        --repo-id "${HF_DOWNSTREAM_REPO}" \
        --task aci_bench \
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
fi

DECORE_MODEL="${MODEL}" \
DECORE_HEADS_JSON="${HEADS}" \
DECORE_DECODING="${DECODING}" \
DECORE_ABLATION_MODE="${ABLATION_MODE}" \
DECORE_MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}" \
DECORE_TENSOR_PARALLEL_SIZE="${GPUS}" \
DECORE_TEMPERATURE="${TEMPERATURE:-0.0}" \
DECORE_TOP_P="${TOP_P:-1.0}" \
DECORE_TOP_K_SAMPLING="${TOP_K_SAMPLING:--1}" \
DECORE_SAMPLING_SEED="${SAMPLING_SEED:-}" \
DECORE_HEADS_LABEL="${HEADS_LABEL:-}" \
./scripts/eval/run_eval.sh aci_bench_task \
    --n-shot "${N_SHOT}" \
    --judge-model "${JUDGE_MODEL}" \
    ${LIMIT:+--limit $LIMIT}

echo "=== Syncing results to HF ==="
python scripts/sync_results.py \
    --repo-id "${HF_DOWNSTREAM_REPO}" \
    --local-dir ./eval_results || echo "WARNING: sync_results.py failed — results saved locally but NOT uploaded to HF" >&2
