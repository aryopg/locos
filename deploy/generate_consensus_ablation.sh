#!/usr/bin/env bash
# Enumerate launch commands for the H1 consensus-ablation matrix:
#   six models  ×  {consensus, wu_only, lc_only}  ×  k ∈ {10, 20, 50}  ×  {niah, nolima}
# writes one `./deploy/launch_multi_ns.sh` command per cell to stdout.
#
# Preconditions
# -------------
# 1. Wu-behavioural NIAH and logit-contribution NoLiMa detection JSONs have
#    already been uploaded to ${HF_RESULTS_REPO} (or downloaded locally into
#    retrieval_heads/).
# 2. consensus_heads.py has been run for each model; the resulting JSONs
#    live under CONSENSUS_DIR (default: retrieval_heads/consensus/) with the
#    naming convention
#       <model>_consensus_k<k>.json
#       <model>_wu_only_k<k>.json
#       <model>_lc_only_k<k>.json
#
# Usage
# -----
#     ./deploy/generate_consensus_ablation.sh > /tmp/consensus_launch.sh
#     bash /tmp/consensus_launch.sh
#
# Or pipe directly to xargs -P for parallel submission.
set -euo pipefail

MODELS=(
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
    "allenai/Olmo-3.1-32B-Instruct"
)
KS=(10 20 50)
SET_KINDS=(consensus wu_only lc_only)
DATASETS=(niah nolima)
CONSENSUS_DIR="${CONSENSUS_DIR:-retrieval_heads/consensus}"

for model in "${MODELS[@]}"; do
    slug="${model##*/}"
    for k in "${KS[@]}"; do
        for set_kind in "${SET_KINDS[@]}"; do
            heads_path="${CONSENSUS_DIR}/${slug}_${set_kind}_k${k}.json"
            for ds in "${DATASETS[@]}"; do
                if [[ "${ds}" == "niah" ]]; then
                    script="deploy/jobs/ablation_niah_consensus.sh"
                else
                    script="deploy/jobs/ablation_nolima_consensus.sh"
                fi
                envs="--env HEADS=${heads_path} --env SET_KIND=${set_kind} --env K_PER_METHOD=${k}"
                echo "MODELS=\"${model}\" ./deploy/launch_multi_ns.sh ${script} ${envs}"
            done
        done
    done
done
