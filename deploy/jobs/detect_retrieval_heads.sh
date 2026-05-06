#!/usr/bin/env bash
# Detection: Original Wu et al. retrieval head detection
set -euo pipefail

DATASET="${DATASET:-niah}"
MAX_LENGTH="${MAX_LENGTH:-50000}"
NUM_LENGTHS="${NUM_LENGTHS:-20}"

echo "=== Retrieval head detection (Wu et al.): ${MODEL} (${DATASET}) ==="

python locos/detectors/behavioral.py \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --max-length "${MAX_LENGTH}" \
    --num-lengths "${NUM_LENGTHS}" \
    --resume

echo "=== Uploading results ==="
python scripts/upload_results.py ./retrieval_heads \
    --repo-id "${HF_RESULTS_REPO}" \
    --path-in-repo "${MODEL_SLUG}/retrieval_heads"
