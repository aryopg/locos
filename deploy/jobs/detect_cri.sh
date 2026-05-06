#!/usr/bin/env bash
# Detection: Causal Retrieval Importance via activation patching
set -euo pipefail

DATASET="${DATASET:-nolima}"
CORRUPTION="${CORRUPTION:-remove}"
NUM_EXAMPLES="${NUM_EXAMPLES:-100}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-5000}"
METRIC="${METRIC:-first_token_logit_diff}"

echo "=== CRI detection: ${MODEL} (${DATASET}, ${CORRUPTION}, metric=${METRIC}) ==="

python locos/detectors/cri.py \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --corruption "${CORRUPTION}" \
    --num-examples "${NUM_EXAMPLES}" \
    --context-length "${CONTEXT_LENGTH}" \
    --metric "${METRIC}" \
    --resume

echo "=== Uploading results ==="
python scripts/upload_results.py ./retrieval_heads \
    --repo-id "${HF_RESULTS_REPO}" \
    --path-in-repo "${MODEL_SLUG}/retrieval_heads"
