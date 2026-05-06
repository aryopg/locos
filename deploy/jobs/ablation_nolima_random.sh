#!/usr/bin/env bash
# Ablation control: NoLiMa retrieval performance with random head selections
set -euo pipefail

ABLATION_MODE="${ABLATION_MODE:-mean}"
VALUES="${VALUES:-1 5 10 20 50 100}"
RANDOM_SEED="${RANDOM_SEED:-42}"

echo "=== NoLiMa ablation (random heads, seed=${RANDOM_SEED}): ${MODEL} (${ABLATION_MODE}) ==="

# Download NoLiMa dataset if not present locally
python locos/download_haystack_data.py --dataset nolima

python locos/analysis/nolima_ablation.py \
    --model "${MODEL}" \
    --random-heads \
    --seed "${RANDOM_SEED}" \
    --mode top-k \
    --values ${VALUES} \
    --ablation-mode "${ABLATION_MODE}" \
    --include-baseline \
    ${LIMIT:+--limit $LIMIT}

echo "=== Uploading results ==="
python scripts/upload_results.py ./ablation_results \
    --repo-id "${HF_RESULTS_REPO}" \
    --path-in-repo "ablation_results"
