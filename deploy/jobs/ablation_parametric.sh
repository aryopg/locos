#!/usr/bin/env bash
# Ablation: parametric/arithmetic eval with varying head selections
set -euo pipefail

ABLATION_MODE="${ABLATION_MODE:-mean}"
VALUES="${VALUES:-1 5 10 20 50 100}"
DATASET="${DATASET:-aryopg/parametric-arithmetic-eval}"

echo "=== Parametric/arithmetic ablation: ${MODEL} (${ABLATION_MODE}) ==="

# Download retrieval heads from HF if not present locally
ensure_heads "${HEADS}"

python locos/analysis/parametric_ablation.py \
    --model "${MODEL}" \
    --heads "${HEADS}" \
    --mode top-k \
    --values ${VALUES} \
    --ablation-mode "${ABLATION_MODE}" \
    --dataset "${DATASET}" \
    ${LIMIT:+--limit $LIMIT}

echo "=== Uploading results ==="
python scripts/upload_results.py ./ablation_parametric_results \
    --repo-id "${HF_RESULTS_REPO}" \
    --path-in-repo "ablation_parametric_results"
