#!/usr/bin/env bash
# Ablation control: parametric/arithmetic eval with random head selections
set -euo pipefail

ABLATION_MODE="${ABLATION_MODE:-mean}"
VALUES="${VALUES:-1 5 10 20 50 100}"
DATASET="${DATASET:-aryopg/parametric-arithmetic-eval}"
RANDOM_SEED="${RANDOM_SEED:-42}"

echo "=== Parametric/arithmetic ablation (random heads, seed=${RANDOM_SEED}): ${MODEL} (${ABLATION_MODE}) ==="

python locos/analysis/parametric_ablation.py \
    --model "${MODEL}" \
    --random-heads \
    --seed "${RANDOM_SEED}" \
    --mode top-k \
    --values ${VALUES} \
    --ablation-mode "${ABLATION_MODE}" \
    --dataset "${DATASET}" \
    --include-baseline \
    ${LIMIT:+--limit $LIMIT}

echo "=== Uploading results ==="
python scripts/upload_results.py ./ablation_parametric_results \
    --repo-id "${HF_RESULTS_REPO}" \
    --path-in-repo "ablation_parametric_results"
