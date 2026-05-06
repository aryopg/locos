#!/usr/bin/env bash
# Detection: Tuned-lens-corrected logit-contribution scoring (H3 experiment)
set -euo pipefail

DATASET="${DATASET:-nolima}"
MAX_LENGTH="${MAX_LENGTH:-5000}"
QUESTION_TYPE="${QUESTION_TYPE:-onehop}"
MAX_CHARACTERS_PER_ENTRY="${MAX_CHARACTERS_PER_ENTRY:-3}"
NUM_LENGTHS="${NUM_LENGTHS:-10}"
NUM_DEPTHS="${NUM_DEPTHS:-10}"
MAX_DECODE_STEPS="${MAX_DECODE_STEPS:-50}"

# Set exactly one of TUNED_LENS_REPO (uzaymacar-style HF repo) or TUNED_LENS_URL
# (direct URL, e.g. AlignmentResearch's Llama3-8B lens).
TUNED_LENS_REPO="${TUNED_LENS_REPO:-}"
TUNED_LENS_URL="${TUNED_LENS_URL:-}"
if [[ -z "${TUNED_LENS_REPO}" && -z "${TUNED_LENS_URL}" ]]; then
    echo "Error: set TUNED_LENS_REPO (e.g. uzaymacar/gemma-3-27b-tuned-lens) or TUNED_LENS_URL." >&2
    exit 1
fi
if [[ -n "${TUNED_LENS_REPO}" && -n "${TUNED_LENS_URL}" ]]; then
    echo "Error: TUNED_LENS_REPO and TUNED_LENS_URL are mutually exclusive." >&2
    exit 1
fi

TL_FLAGS=()
if [[ -n "${TUNED_LENS_REPO}" ]]; then
    TL_FLAGS+=(--tuned-lens "${TUNED_LENS_REPO}")
else
    TL_FLAGS+=(--tuned-lens-url "${TUNED_LENS_URL}")
fi

python locos/download_haystack_data.py --dataset nolima

echo "=== Tuned-lens logit-contrib detection: ${MODEL} (${DATASET}) ==="

python locos/detectors/logit_contrib.py \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --max-length "${MAX_LENGTH}" \
    --question-type "${QUESTION_TYPE}" \
    --max-characters-per-entry "${MAX_CHARACTERS_PER_ENTRY}" \
    --num-lengths "${NUM_LENGTHS}" \
    --num-depths "${NUM_DEPTHS}" \
    --max-decode-steps "${MAX_DECODE_STEPS}" \
    "${TL_FLAGS[@]}" \
    --chat-template \
    --resume

echo "=== Uploading results ==="
python scripts/upload_results.py ./retrieval_heads \
    --repo-id "${HF_RESULTS_REPO}" \
    --path-in-repo "retrieval_heads"
