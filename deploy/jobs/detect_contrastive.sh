#!/usr/bin/env bash
# Detection: Contrastive attention-based retrieval head detection
set -euo pipefail

DATASET="${DATASET:-nolima}"
MAX_LENGTH="${MAX_LENGTH:-50000}"
NUM_LENGTHS="${NUM_LENGTHS:-20}"
TOP_K="${TOP_K:-10}"

echo "=== Contrastive detection: ${MODEL} (${DATASET}) ==="

python locos/detectors/contrastive.py \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --max-length "${MAX_LENGTH}" \
    --num-lengths "${NUM_LENGTHS}" \
    --top-k "${TOP_K}" \
    --resume

echo "=== Uploading results ==="
python scripts/upload_results.py ./retrieval_heads \
    --repo-id "${HF_RESULTS_REPO}" \
    --path-in-repo "${MODEL_SLUG}/retrieval_heads"
