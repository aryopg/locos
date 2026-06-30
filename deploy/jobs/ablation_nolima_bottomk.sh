#!/usr/bin/env bash
# Ablation control: NoLiMa retrieval with bottom-k heads (specificity control)
set -euo pipefail

# Shared helpers (ensure_heads, model registry). Run from the repo root.
source deploy/job_config.sh

ABLATION_MODE="${ABLATION_MODE:-mean}"
VALUES="${VALUES:-1 5 10 20 50}"

echo "=== NoLiMa ablation (bottom-k heads): ${MODEL} (${ABLATION_MODE}) ==="

# Download NoLiMa dataset if not present locally
python locos/download_haystack_data.py --dataset nolima

# Download retrieval heads from HF if not present locally
ensure_heads "${HEADS}"

python locos/analysis/nolima_ablation.py \
    --model "${MODEL}" \
    --heads "${HEADS}" \
    --mode bottom-k \
    --values ${VALUES} \
    --ablation-mode "${ABLATION_MODE}" \
    ${LIMIT:+--limit $LIMIT}

echo "=== Uploading results ==="
python scripts/upload_results.py ./ablation_results \
    --repo-id "${HF_RESULTS_REPO}" \
    --path-in-repo "ablation_results"
