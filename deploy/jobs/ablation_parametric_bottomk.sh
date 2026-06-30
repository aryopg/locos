#!/usr/bin/env bash
# Ablation control: parametric/arithmetic eval with bottom-k heads (specificity control)
set -euo pipefail

# Shared helpers (ensure_heads, model registry). Run from the repo root.
source deploy/job_config.sh

ABLATION_MODE="${ABLATION_MODE:-mean}"
VALUES="${VALUES:-1 5 10 20 50}"
DATASET="${DATASET:-aryopg/parametric-arithmetic-eval}"

echo "=== Parametric/arithmetic ablation (bottom-k heads): ${MODEL} (${ABLATION_MODE}) ==="

# Download retrieval heads from HF if not present locally
ensure_heads "${HEADS}"

python locos/analysis/parametric_ablation.py \
    --model "${MODEL}" \
    --heads "${HEADS}" \
    --mode bottom-k \
    --values ${VALUES} \
    --ablation-mode "${ABLATION_MODE}" \
    --dataset "${DATASET}" \
    ${LIMIT:+--limit $LIMIT}

echo "=== Uploading results ==="
python scripts/upload_results.py ./ablation_parametric_results \
    --repo-id "${HF_RESULTS_REPO}" \
    --path-in-repo "ablation_parametric_results"
