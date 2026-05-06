#!/usr/bin/env bash
# Detection: HeadKV-style anchor-window attention scoring (no generation, no ROUGE gate)
set -euo pipefail

DATASET="${DATASET:-nolima}"
MAX_LENGTH="${MAX_LENGTH:-5000}"
QUESTION_TYPE="${QUESTION_TYPE:-onehop}"
MAX_CHARACTERS_PER_ENTRY="${MAX_CHARACTERS_PER_ENTRY:-3}"
NUM_LENGTHS="${NUM_LENGTHS:-10}"
NUM_DEPTHS="${NUM_DEPTHS:-10}"
ANCHOR_WINDOW="${ANCHOR_WINDOW:-8}"

python locos/download_haystack_data.py --dataset nolima

echo "=== HeadKV detection: ${MODEL} (${DATASET}, K=${ANCHOR_WINDOW}) ==="

python locos/detectors/headkv.py \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --max-length "${MAX_LENGTH}" \
    --question-type "${QUESTION_TYPE}" \
    --max-characters-per-entry "${MAX_CHARACTERS_PER_ENTRY}" \
    --num-lengths "${NUM_LENGTHS}" \
    --num-depths "${NUM_DEPTHS}" \
    --anchor-window "${ANCHOR_WINDOW}" \
    --chat-template \
    --resume

echo "=== Uploading results ==="
python scripts/upload_results.py ./retrieval_heads \
    --repo-id "${HF_RESULTS_REPO}" \
    --path-in-repo "retrieval_heads"
