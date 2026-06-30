#!/usr/bin/env bash
# Ablation: NIAH retrieval performance using a pre-extracted consensus /
# method-only / lc-only head set (see
# locos/analysis/consensus_heads.py).
#
# Required env vars:
#   MODEL              HF model name
#   HEADS              Path to the consensus or *_only JSON
#                      (flat {"layer-head": [...]} format)
#   HF_RESULTS_REPO    Destination HF repo for uploaded results
#   MODEL_SLUG         Slug used in the results path
#   SET_KIND           "consensus" | "wu_only" | "lc_only" — suffixes the
#                      output dir so the three variants do not overwrite.
#
# Optional:
#   K_PER_METHOD       The k that was used to build the consensus set
#                      (for labelling only, not for re-cutting).
#   ABLATION_MODE      "mean" (default) | "zero"
#   MAX_LENGTH         Max NIAH context length (default: 5000)
#   NUM_LENGTHS        (default: 10)
#   NUM_DEPTHS         (default: 10)
#   LIMIT              Optional --limit passed through for smoke tests.
set -euo pipefail

# Shared helpers (ensure_heads, model registry). Run from the repo root.
source deploy/job_config.sh

ABLATION_MODE="${ABLATION_MODE:-mean}"
MAX_LENGTH="${MAX_LENGTH:-5000}"
NUM_LENGTHS="${NUM_LENGTHS:-10}"
NUM_DEPTHS="${NUM_DEPTHS:-10}"
SET_KIND="${SET_KIND:-consensus}"
K_PER_METHOD="${K_PER_METHOD:-unknown}"

echo "=== NIAH consensus ablation: ${MODEL} (${ABLATION_MODE}, set=${SET_KIND}, k_per_method=${K_PER_METHOD}) ==="

# Download NIAH haystack data if not present locally
python locos/download_haystack_data.py --dataset niah

# Ensure the heads JSON exists (consensus_heads.py should have produced it
# locally already; we do NOT call ensure_heads because the JSON is not on HF).
if [[ ! -f "${HEADS}" ]]; then
    echo "Consensus heads file not found: ${HEADS}" >&2
    echo "Run locos/analysis/consensus_heads.py first." >&2
    exit 1
fi

# Count heads in the consensus JSON to derive the k argument.
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

# Cache naming uses args.heads.stem, and the consensus JSON filenames already
# include SET_KIND and the per-method k (e.g. "Qwen3-8B_consensus_k20.json"),
# so no extra suffix argument is needed for disambiguation.
python locos/analysis/nolima_ablation.py \
    --dataset niah \
    --model "${MODEL}" \
    --heads "${HEADS}" \
    --mode top-k \
    --values "${SET_SIZE}" \
    --ablation-mode "${ABLATION_MODE}" \
    --max-length "${MAX_LENGTH}" \
    --num-lengths "${NUM_LENGTHS}" \
    --num-depths "${NUM_DEPTHS}" \
    ${LIMIT:+--limit $LIMIT}

echo "=== Uploading results ==="
python scripts/upload_results.py ./ablation_results \
    --repo-id "${HF_RESULTS_REPO}" \
    --path-in-repo "ablation_results"
