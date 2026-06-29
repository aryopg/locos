#!/usr/bin/env bash
# Detection: direct logit attribution baseline (LOCOS without spatial contrast)
set -euo pipefail

DATASET="${DATASET:-nolima}"
MAX_LENGTH="${MAX_LENGTH:-5000}"
QUESTION_TYPE="${QUESTION_TYPE:-onehop}"
MAX_CHARACTERS_PER_ENTRY="${MAX_CHARACTERS_PER_ENTRY:-3}"
NUM_LENGTHS="${NUM_LENGTHS:-10}"
NUM_DEPTHS="${NUM_DEPTHS:-10}"
MAX_DECODE_STEPS="${MAX_DECODE_STEPS:-50}"

python locos/download_haystack_data.py --dataset nolima

echo "=== DLA detection: ${MODEL} (${DATASET}) ==="

python locos/detectors/dla.py \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --max-length "${MAX_LENGTH}" \
    --question-type "${QUESTION_TYPE}" \
    --max-characters-per-entry "${MAX_CHARACTERS_PER_ENTRY}" \
    --num-lengths "${NUM_LENGTHS}" \
    --num-depths "${NUM_DEPTHS}" \
    --max-decode-steps "${MAX_DECODE_STEPS}" \
    --chat-template \
    --resume

echo "=== Uploading results ==="
python scripts/upload_results.py ./retrieval_heads \
    --repo-id "${HF_RESULTS_REPO}" \
    --path-in-repo "retrieval_heads"
