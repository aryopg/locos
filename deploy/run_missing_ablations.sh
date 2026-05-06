#!/usr/bin/env bash
# Launch missing ablation experiments:
#   1. Random-head control ablations with seeds 43 and 44 (6 main paper models)
#   2. CRI-detected head ablations (Qwen3-8B and gemma-3-12b-it only)
#
# Run from repo root:
#   ./deploy/run_missing_ablations.sh [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/launch_multi_ns.sh"

MAIN_MODELS="google/gemma-3-12b-it google/gemma-3-27b-it Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B allenai/Olmo-3.1-32B-Instruct"
CRI_MODELS="Qwen/Qwen3-8B google/gemma-3-12b-it"

DRY_RUN_FLAG=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN_FLAG="--dry-run"
fi

# ---------------------------------------------------------------------------
# 1. Random-head control ablations: seeds 43 and 44, all 6 main paper models
# ---------------------------------------------------------------------------

for SEED in 43 44; do
    echo ""
    echo "======================================================="
    echo " Random ablations — seed=${SEED}"
    echo "======================================================="

    echo ""
    echo "--- NIAH random (seed=${SEED}) ---"
    MODELS="${MAIN_MODELS}" \
        "${LAUNCHER}" deploy/jobs/ablation_niah_random.sh \
        --env RANDOM_SEED="${SEED}" \
        ${DRY_RUN_FLAG}

    echo ""
    echo "--- NoLiMa random (seed=${SEED}) ---"
    MODELS="${MAIN_MODELS}" \
        "${LAUNCHER}" deploy/jobs/ablation_nolima_random.sh \
        --env RANDOM_SEED="${SEED}" \
        ${DRY_RUN_FLAG}

    echo ""
    echo "--- Parametric random (seed=${SEED}) ---"
    MODELS="${MAIN_MODELS}" \
        "${LAUNCHER}" deploy/jobs/ablation_parametric_random.sh \
        --env RANDOM_SEED="${SEED}" \
        ${DRY_RUN_FLAG}
done

# ---------------------------------------------------------------------------
# 2. CRI-detected head ablations: Qwen3-8B and gemma-3-12b-it only
# ---------------------------------------------------------------------------

echo ""
echo "======================================================="
echo " CRI-detected head ablations (Qwen3-8B, gemma-3-12b-it)"
echo "======================================================="

echo ""
echo "--- NIAH (CRI heads) ---"
MODELS="${CRI_MODELS}" HEADS_METHOD="cri_first_token_logit_diff" \
    "${LAUNCHER}" deploy/jobs/ablation_niah.sh \
    ${DRY_RUN_FLAG}

echo ""
echo "--- NoLiMa (CRI heads) ---"
MODELS="${CRI_MODELS}" HEADS_METHOD="cri_first_token_logit_diff" \
    "${LAUNCHER}" deploy/jobs/ablation_nolima.sh \
    ${DRY_RUN_FLAG}

echo ""
echo "--- Parametric (CRI heads) ---"
MODELS="${CRI_MODELS}" HEADS_METHOD="cri_first_token_logit_diff" \
    "${LAUNCHER}" deploy/jobs/ablation_parametric.sh \
    ${DRY_RUN_FLAG}
