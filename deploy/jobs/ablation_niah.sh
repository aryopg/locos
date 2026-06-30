#!/usr/bin/env bash
# Ablation: NIAH retrieval performance with varying head selections
set -euo pipefail

# Shared helpers (ensure_heads, model registry). Run from the repo root.
source deploy/job_config.sh

ABLATION_MODE="${ABLATION_MODE:-mean}"
VALUES="${VALUES:-1 5 10 20 50}"
MAX_LENGTH="${MAX_LENGTH:-5000}"
NUM_LENGTHS="${NUM_LENGTHS:-10}"
NUM_DEPTHS="${NUM_DEPTHS:-10}"

echo "=== NIAH ablation: ${MODEL} (${ABLATION_MODE}) ==="
echo "    grid: ${NUM_LENGTHS} lengths x ${NUM_DEPTHS} depths, max_length=${MAX_LENGTH}"

# Download NIAH haystack data if not present locally
python locos/download_haystack_data.py --dataset niah

# Download retrieval heads from HF if not present locally
ensure_heads "${HEADS}"

python locos/analysis/nolima_ablation.py \
    --dataset niah \
    --model "${MODEL}" \
    --heads "${HEADS}" \
    --mode top-k \
    --values ${VALUES} \
    --ablation-mode "${ABLATION_MODE}" \
    --max-length "${MAX_LENGTH}" \
    --num-lengths "${NUM_LENGTHS}" \
    --num-depths "${NUM_DEPTHS}" \
    ${LIMIT:+--limit $LIMIT}

echo "=== Uploading results ==="
python scripts/upload_results.py ./ablation_results \
    --repo-id "${HF_RESULTS_REPO}" \
    --path-in-repo "ablation_results"
