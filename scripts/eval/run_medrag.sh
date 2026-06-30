#!/usr/bin/env bash
# Run MedRAG evaluations across models, sub-datasets, and decoding variants.
#
# Usage:
#   ./scripts/eval/run_medrag.sh                    # run all combinations
#   ./scripts/eval/run_medrag.sh --dry-run           # print commands only
#   ./scripts/eval/run_medrag.sh --limit 100         # limit samples per sub-dataset
#   ./scripts/eval/run_medrag.sh --datasets "medqa mmlu_med"  # subset of sub-datasets
#   ./scripts/eval/run_medrag.sh --models "Qwen/Qwen3-8B"    # subset of models
#   ./scripts/eval/run_medrag.sh --top-ks "5 10"              # subset of top-k values
#
# Each combination runs sequentially. Use --dry-run to preview.
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model → TP size (bash 3.2 compatible — no associative arrays)
gpu_count_for_model() {
    case "$1" in
        meta-llama/Meta-Llama-3-8B-Instruct)  echo 1 ;;
        google/gemma-3-4b-it)                   echo 1 ;;
        google/gemma-3-12b-it)                  echo 2 ;;
        google/gemma-3-27b-it)                  echo 4 ;;
        Qwen/Qwen3-4B)                          echo 1 ;;
        Qwen/Qwen3-8B)                          echo 1 ;;
        Qwen/Qwen3-14B)                         echo 2 ;;
        Qwen/Qwen3-32B)                         echo 4 ;;
        *)                                      echo 2 ;;
    esac
}

DEFAULT_MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "google/gemma-3-4b-it"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
)

# All MedRAG sub-datasets
ALL_DATASETS=(mmlu_med medqa medmcqa pubmedqa supergpqa_med)

# Focus sub-datasets (uncomment to limit by default)
# ALL_DATASETS=(medqa mmlu_med supergpqa_med)

# Decoding variants: "label:decoding_mode:heads_suffix"
#   - label: used for --heads-label (empty = auto-infer)
#   - decoding_mode: greedy or ablation
#   - heads_suffix: appended to model name for heads file (e.g. "_wu_niah", "_wu_nolima")
VARIANTS=(
    ":greedy:"
    "wu_niah:ablation:_wu_niah"
    "wu_nolima:ablation:_wu_nolima"
)

# Number of retrieved passages to include
ALL_TOP_KS=(5 10 15 20)

# Shared generation settings
MAX_MODEL_LEN=8192
GPU_MEM=0.5
MAX_TOKENS=1024

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Extract short model name from full path (e.g. "Qwen/Qwen3-8B" -> "Qwen3-8B")
model_short() {
    echo "$1" | sed 's|.*/||'
}

# Build the retrieval heads path for a model + suffix
heads_path() {
    local short suffix
    short=$(model_short "$1")
    suffix="$2"
    echo "retrieval_heads/${short}${suffix}.json"
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

DRY_RUN=false
LIMIT=""
MODELS=()
DATASETS=()
TOP_KS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true; shift ;;
        --limit)      LIMIT="$2"; shift 2 ;;
        --models)     IFS=' ' read -ra MODELS <<< "$2"; shift 2 ;;
        --datasets)   IFS=' ' read -ra DATASETS <<< "$2"; shift 2 ;;
        --top-ks)     IFS=' ' read -ra TOP_KS <<< "$2"; shift 2 ;;
        *)            echo "Unknown flag: $1"; exit 1 ;;
    esac
done

[[ ${#MODELS[@]} -eq 0 ]] && MODELS=("${DEFAULT_MODELS[@]}")
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=("${ALL_DATASETS[@]}")
[[ ${#TOP_KS[@]} -eq 0 ]] && TOP_KS=("${ALL_TOP_KS[@]}")

# Build optional flags
LIMIT_FLAG=""
[[ -n "$LIMIT" ]] && LIMIT_FLAG="--limit ${LIMIT}"

# ---------------------------------------------------------------------------
# Run matrix
# ---------------------------------------------------------------------------

total=0
skipped=0

echo "========================================"
echo "MedRAG Evaluation Runner"
echo "========================================"
echo "Models:    ${#MODELS[@]}"
echo "Datasets:  ${DATASETS[*]}"
echo "Top-k:     ${TOP_KS[*]}"
echo "Variants:  ${#VARIANTS[@]} (greedy, ablation_wu_niah, ablation_wu_nolima)"
[[ -n "$LIMIT" ]] && echo "Limit:     ${LIMIT} samples"
echo "========================================"
echo ""

for model in "${MODELS[@]}"; do
    short=$(model_short "$model")
    tp=$(gpu_count_for_model "$model")

    for variant_entry in "${VARIANTS[@]}"; do
        IFS=':' read -r label decoding heads_suffix <<< "$variant_entry"

        # Build heads path (not needed for greedy)
        heads_flag=""
        heads_label_flag=""
        if [[ "$decoding" == "ablation" ]]; then
            hpath=$(heads_path "$model" "$heads_suffix")
            if [[ ! -f "$hpath" ]]; then
                echo "[SKIP] ${short} / ${decoding}_${label}: heads file not found: ${hpath}"
                skipped=$((skipped + 1))
                continue
            fi
            heads_flag="--heads ${hpath}"
            [[ -n "$label" ]] && heads_label_flag="--heads-label ${label}"
        fi

        # Display name: "greedy" or "ablation_wu_niah" / "ablation_wu_nolima"
        if [[ -n "$label" ]]; then
            display_variant="${decoding}_${label}"
        else
            display_variant="${decoding}"
        fi

        for topk in "${TOP_KS[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                total=$((total + 1))

                cmd="python -m locos_eval.evals.tasks.medrag_task \
    --model ${model} \
    --decoding ${decoding} \
    --tp ${tp} \
    --gpu-mem ${GPU_MEM} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-tokens ${MAX_TOKENS} \
    --dataset-name ${dataset} \
    --top-k ${topk} \
    ${heads_flag} ${heads_label_flag} ${LIMIT_FLAG}"

                # Clean up extra whitespace
                cmd=$(echo "$cmd" | sed 's/  */ /g; s/ *$//')

                if [[ "$DRY_RUN" == true ]]; then
                    echo "--- [${total}] ${short} / ${display_variant} / ${dataset} / top${topk} (TP=${tp}) ---"
                    echo "  ${cmd}"
                    echo ""
                else
                    echo ""
                    echo "========================================"
                    echo "[${total}] ${short} / ${display_variant} / ${dataset} / top${topk} (TP=${tp})"
                    echo "========================================"
                    eval "$cmd"
                fi
            done
        done
    done
done

echo ""
echo "========================================"
echo "Total runs: ${total}"
[[ $skipped -gt 0 ]] && echo "Skipped:    ${skipped} (missing heads files)"
[[ "$DRY_RUN" == true ]] && echo "(dry run — nothing was executed)"
echo "========================================"
