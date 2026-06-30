# Job Scripts

Bash scripts that define individual experiments. Each script runs inside a
prepared checkout after the environment variables below have been set.

## How it works

Scripts only need to run the experiment and upload their own result artifacts.
They can be invoked directly from the repository root on a local machine or any
GPU host with the project environment activated.

## Environment variables

The launcher exports these before running your script:

| Variable | Description |
| --- | --- |
| `MODEL` | HuggingFace model name (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`) |
| `HEADS` | Retrieval heads JSON path relative to repo root |
| `GPUS` | Number of GPUs allocated (equals TP size) |
| `MODEL_SLUG` | Short model name for paths (e.g., `meta-llama-3-8b-instruct`) |
| `HF_RESULTS_REPO` | HuggingFace dataset repo for uploading results |

Use `HF_RESULTS_REPO=aryopg/locos-results` for public artifacts. Downstream
eval jobs upload to the `downstream_results/` prefix in the same repo.

Set any script-specific variables in the shell before invoking the script.
Large result directories should be uploaded or copied before cleaning the
workspace.

## Template

```bash
#!/usr/bin/env bash
# One-line description of the experiment
set -euo pipefail

echo "=== Experiment name: ${MODEL} ==="

# ... run experiment ...

echo "=== Uploading results ==="
python scripts/upload_results.py ./results_dir \
    --repo-id "${HF_RESULTS_REPO}" \
    --path-in-repo "${MODEL_SLUG}/experiment-name"
```

## Existing scripts

| Script | Description |
| --- | --- |
| `eval_babilong.sh` | BABILong long-context free-form QA eval |
| `eval_musique.sh` | MuSiQue multi-hop open-book QA eval |
| `detect_logit_contrib.sh` | LOCOS write-aware OV logit-contribution retrieval head detection |
| `detect_contrastive.sh` | Contrastive attention scoring baseline |
| `detect_cri.sh` | Causal Retrieval Importance via activation patching |
| `detect_dla.sh` | Direct logit attribution baseline: LOCOS without spatial contrast |
| `detect_retrieval_heads.sh` | Original Wu et al. retrieval head detection |
| `ablation_nolima.sh` | NoLiMa retrieval performance ablation with varying head selections |
