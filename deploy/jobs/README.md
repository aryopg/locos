# Job Scripts

Bash scripts that define individual experiments. Each script runs inside a
Kubernetes pod after the repo is cloned and the venv is activated.

## How it works

The launcher (`launch_multi_ns.sh`) wraps each script with log capture and
uploads logs automatically. Scripts only need to run the experiment and upload
their own result artifacts.

## Environment variables

The launcher exports these before running your script:

| Variable | Description |
|---|---|
| `MODEL` | HuggingFace model name (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`) |
| `HEADS` | Retrieval heads JSON path relative to repo root |
| `GPUS` | Number of GPUs allocated (equals TP size) |
| `MODEL_SLUG` | Short model name for paths (e.g., `meta-llama-3-8b-instruct`) |
| `HF_RESULTS_REPO` | HuggingFace repo for uploading results |

Additional variables can be passed via `--env KEY=VALUE` on the launcher CLI.

Per-namespace settings (SSH host, GPU product, secrets, username) are
configured in `deploy/namespaces.conf`.

> **WARNING: Pods are ephemeral.** When a script exits (or crashes), the pod
> is killed and all local storage is destroyed. Every artifact you care about
> **must be uploaded before the script exits**. If a script fails midway,
> intermediate results are lost unless you explicitly checkpoint and upload
> during execution.

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
|---|---|
| `eval_nq_swap.sh` | NQ-Swap context faithfulness eval (sub_EM, org_EM) |
| `eval_xsum.sh` | XSum summarization eval (ROUGE-L, BERTScore, FactKB) |
| `eval_medrag.sh` | MedRAG medical QA eval (5 sub-datasets, MCQ accuracy) |
| `eval_aci_bench.sh` | ACI-Bench dialogue-to-note eval (ROUGE-L, BERTScore, LLM judge) |
| `detect_logit_contrib.sh` | Contrastive logit-contribution retrieval head detection |
| `detect_contrastive.sh` | Contrastive attention-based retrieval head detection |
| `detect_cri.sh` | Causal Retrieval Importance via activation patching |
| `detect_retrieval_heads.sh` | Original Wu et al. retrieval head detection |
| `ablation_nolima.sh` | NoLiMa retrieval performance ablation with varying head selections |
