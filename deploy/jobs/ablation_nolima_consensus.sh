#!/usr/bin/env bash
# Ablation: NoLiMa retrieval performance using a pre-extracted consensus /
# method-only head set (counterpart to ablation_niah_consensus.sh).
#
# Same env vars as ablation_niah_consensus.sh — see that file for full docs.
set -euo pipefail

ABLATION_MODE="${ABLATION_MODE:-mean}"
SET_KIND="${SET_KIND:-consensus}"
K_PER_METHOD="${K_PER_METHOD:-unknown}"

echo "=== NoLiMa consensus ablation: ${MODEL} (${ABLATION_MODE}, set=${SET_KIND}, k_per_method=${K_PER_METHOD}) ==="

python locos/download_haystack_data.py --dataset nolima

if [[ ! -f "${HEADS}" ]]; then
    echo "Consensus heads file not found: ${HEADS}" >&2
    echo "Run locos/analysis/consensus_heads.py first." >&2
    exit 1
fi

SET_SIZE="$(python - <<PY
import json
with open("${HEADS}") as f:
    d = json.load(f)
scores = d.get("scores", d)
print(len(scores))
PY
)"

if [[ "${SET_SIZE}" -eq 0 ]]; then
    echo "Consensus set is empty; skipping ablation." >&2
    exit 0
fi

echo "Consensus set size: ${SET_SIZE}"

python locos/analysis/nolima_ablation.py \
    --dataset nolima \
    --model "${MODEL}" \
    --heads "${HEADS}" \
    --mode top-k \
    --values "${SET_SIZE}" \
    --ablation-mode "${ABLATION_MODE}" \
    ${LIMIT:+--limit $LIMIT}

echo "=== Uploading results ==="
python scripts/upload_results.py ./ablation_results \
    --repo-id "${HF_RESULTS_REPO}" \
    --path-in-repo "ablation_results"
