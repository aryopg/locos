#!/usr/bin/env bash
# Shared infrastructure helpers for all job launchers.
# Sourced by launch_evals.sh, launch_multi_ns.sh, etc.
#
# Contains: docker image, model registry, GPU counts, setup commands,
# and a generic job-command builder. Eval-specific code lives in
# the individual launcher scripts.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOCKER_IMAGE="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
# HF_RESULTS_REPO holds heads files, detection JSONs, ablation analyses, and job
# logs (everything except downstream eval results). Stable target — the broken
# native-ablation runs from before the eager-mode fix did NOT pollute it.
HF_RESULTS_REPO="${HF_RESULTS_REPO:-aryopg/decore-results}"
# HF_DOWNSTREAM_REPO holds task-level eval outputs (NQ-Swap, MuSiQue, MedRAG,
# LongBench-v2, BABILong, ACI-Bench, XSum). Split out from HF_RESULTS_REPO so
# that re-runs after the enforce_eager fix start fresh and the broken
# zero-effect ablation results from before that fix can't contaminate them.
HF_DOWNSTREAM_REPO="${HF_DOWNSTREAM_REPO:-aryopg/locos_downstream_results}"

# Model → GPU count lookup (bash 3.2 compatible — no associative arrays)
gpu_count_for_model() {
    case "$1" in
        meta-llama/Meta-Llama-3-8B-Instruct)    echo 1 ;;
        google/gemma-3-4b-it)                   echo 1 ;;
        google/gemma-3-12b-it)                  echo 1 ;;
        google/gemma-3-27b-it)                  echo 2 ;;
        Qwen/Qwen3-4B)                          echo 1 ;;
        Qwen/Qwen3-8B)                          echo 1 ;;
        Qwen/Qwen3-14B)                         echo 1 ;;
        Qwen/Qwen3-32B)                         echo 2 ;;
        allenai/Olmo-3.1-32B-Instruct)          echo 2 ;;
        allenai/Olmo-3-7B-Instruct)             echo 1 ;;
        openai/gpt-oss-20b)                     echo 2 ;;
        openai/gpt-oss-120b)                    echo 4 ;;
        *)                                      echo 2 ;;  # default
    esac
}

# Default model list (override with MODELS env var, space-separated)
DEFAULT_MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "google/gemma-3-4b-it"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
    "allenai/Olmo-3.1-32B-Instruct"
    "allenai/Olmo-3-7B-Instruct"
    "openai/gpt-oss-20b"
    "openai/gpt-oss-120b"
)

# ---------------------------------------------------------------------------
# Retrieval heads download
# ---------------------------------------------------------------------------

# Ensure a retrieval heads file exists locally, downloading from HF if needed.
# Skips if: empty path, "random" heads, or file already exists.
# Usage: ensure_heads "$HEADS"
ensure_heads() {
    local heads_path="$1"
    [[ -z "$heads_path" ]] && return 0
    [[ "$heads_path" == "random" ]] && return 0
    [[ -f "$heads_path" ]] && return 0

    echo "Downloading retrieval heads: ${heads_path}"
    python scripts/download_heads.py \
        --repo-id "${HF_RESULTS_REPO}" \
        --heads "$heads_path"
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Provider-preserving slug for directory paths and HF paths.
# "meta-llama/Meta-Llama-3-8B-Instruct" → "meta-llama_Meta-Llama-3-8B-Instruct"
model_slug() {
    echo "$1" | tr '/' '_'
}

# DNS-safe lowercase slug for kblaunch job names only (NOT for directory paths).
# Strips -Instruct/-it suffixes to stay within Kubernetes 63-char pod name limit.
# "meta-llama/Meta-Llama-3-8B-Instruct" → "meta-llama-3-8b"
# "google/gemma-3-4b-it" → "gemma-3-4b"
job_slug() {
    echo "$1" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '.' '-' | sed 's/-instruct$//; s/-it$//'
}

# Derive retrieval heads path from model name and HEADS_METHOD.
# HEADS_METHOD should be one of: wu_niah, wu_nolima, logit_contrib
heads_path() {
    local slug method
    slug=$(echo "$1" | sed 's|.*/||')
    method="${HEADS_METHOD:-wu_niah}"
    echo "retrieval_heads/${slug}_${method}.json"
}

# Short slug for HEADS_METHOD / HEADS_LABEL, used in job names to stay under
# the Kubernetes 63-character limit.
method_job_slug() {
    case "$1" in
        wu_niah)        echo "wn" ;;
        wu_nolima)      echo "wl" ;;
        logit_contrib_nolima|logitcontrib_nolima)  echo "lc" ;;
        logit_contrib_nolima_tuned_lens)  echo "lctl" ;;
        cri_first_token_logit_diff)  echo "cri" ;;
        random)         echo "rnd" ;;
        greedy)         echo "gr" ;;
        *)              echo "$1" | tr '_' '-' ;;
    esac
}

# Resolve task names for a job script, given EXTRA_ENVS.
# Returns newline-separated task names for use with check_experiment.py.
# Non-eval scripts (detect_*, etc.) return nothing → no pre-check.
task_names_for_script() {
    local script="$1"
    shift
    local envs=("$@")

    # Parse EXTRA_ENVS into local variables
    local DECODING="decore" TOP_K="5" LENGTH="short"
    local DATASETS="mmlu_med medqa supergpqa_med"
    for env_pair in "${envs[@]+"${envs[@]}"}"; do
        case "$env_pair" in
            DECODING=*)  DECODING="${env_pair#DECODING=}" ;;
            TOP_K=*)     TOP_K="${env_pair#TOP_K=}" ;;
            LENGTH=*)    LENGTH="${env_pair#LENGTH=}" ;;
            DATASETS=*)  DATASETS="${env_pair#DATASETS=}" ;;
        esac
    done

    local base
    base=$(basename "$script" .sh)
    case "$base" in
        eval_nq_swap)
            echo "nq_swap"
            ;;
        eval_xsum)
            echo "xsum_faithfulness"
            ;;
        eval_aci_bench)
            echo "aci_bench"
            ;;
        eval_longbench_v2)
            if [[ "$LENGTH" == "all" ]]; then
                echo "longbench_v2"
            else
                echo "longbench_v2_${LENGTH}"
            fi
            ;;
        eval_medrag)
            for ds in ${DATASETS}; do
                echo "medrag_${ds}_top${TOP_K}"
            done
            ;;
        *)
            # Non-eval script (detection, etc.) — no pre-check
            ;;
    esac
}

# Generate the setup commands that run inside each pod
setup_commands() {
    cat <<'SETUP'
set -euo pipefail
[[ "${BASH_TRACE:-}" == "true" ]] && set -x

# Install system dependencies
apt-get update -qq && apt-get install -y -qq nano byobu htop nvtop jq rsync sqlite3 libsqlite3-dev > /dev/null 2>&1

# Some EIDF namespaces (e.g. eidf029ns) mount /workspace/.cache read-only,
# which breaks uv/pip/HF default cache locations under $HOME/.cache.
# Redirect cache roots to a sibling path that lives on the writable /workspace
# mount; fall back to /tmp if /workspace itself is locked down. Set the HF
# variables explicitly because the base image may already define them, which
# takes precedence over XDG_CACHE_HOME in huggingface_hub.
export XDG_CACHE_HOME="/workspace/cache"
if ! mkdir -p "$XDG_CACHE_HOME" 2>/dev/null; then
    export XDG_CACHE_HOME="/tmp/cache"
    mkdir -p "$XDG_CACHE_HOME"
fi
export UV_CACHE_DIR="${XDG_CACHE_HOME}/uv"
export PIP_CACHE_DIR="${XDG_CACHE_HOME}/pip"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clone and install
git clone --depth 1 https://${GIT_TOKEN}@github.com/aryopg/locos_eval.git
cd locos_eval
uv venv --python python3.12 --system-site-packages
UV_HTTP_TIMEOUT=300 uv pip install -e ".[dev,eval]" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    --python .venv/bin/python
source .venv/bin/activate

# Helper: download retrieval heads from HF if not present locally
ensure_heads() {
    local heads_path="$1"
    [[ -z "$heads_path" ]] && return 0
    [[ "$heads_path" == "random" ]] && return 0
    [[ -f "$heads_path" ]] && return 0

    echo "Downloading retrieval heads: ${heads_path}"
    python scripts/download_heads.py \
        --repo-id "${HF_RESULTS_REPO}" \
        --heads "$heads_path"
}
SETUP
}

# Build a pod command from a job script file.
# Exports standard env vars, prepends setup, wraps with log capture.
# Usage: build_generic_command <model> <heads> <gpus> <script_path> [KEY=VALUE ...]
build_generic_command() {
    local model="$1" heads="$2" gpus="$3" script_path="$4"
    shift 4
    local mslug jslug
    mslug=$(model_slug "$model")
    jslug=$(job_slug "$model")
    local script_name
    script_name=$(basename "$script_path" .sh)

    # Read the job script, stripping shebang and redundant set -euo pipefail
    # (setup_commands already provides these)
    local script_body
    script_body=$(sed '1{/^#!/d;}; /^set -euo pipefail$/d' "$script_path")

    cat <<OUTER
$(setup_commands)

# --- Environment ---
export MODEL='${model}'
export HEADS='${heads}'
export GPUS='${gpus}'
export MODEL_SLUG='${mslug}'
export HF_RESULTS_REPO='${HF_RESULTS_REPO}'
export HF_DOWNSTREAM_REPO='${HF_DOWNSTREAM_REPO}'
export PYTHONUNBUFFERED=1
$(for env_pair in "$@"; do echo "export ${env_pair}"; done)

# --- Log capture ---
LOG_DIR="/tmp/decore-logs"
mkdir -p "\$LOG_DIR"
LOG_FILE="\${LOG_DIR}/decore-${jslug}-${script_name}.log"
exec > >(stdbuf -oL tee -a "\$LOG_FILE") 2>&1

# --- Job script: ${script_name} ---
${script_body}

# --- Upload logs ---
echo '=== Uploading job log ==='
python scripts/upload_results.py "\$LOG_DIR" \\
    --repo-id "\${HF_RESULTS_REPO}" \\
    --path-in-repo "\${MODEL_SLUG}/logs"
OUTER
}
