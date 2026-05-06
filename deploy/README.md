# Deploy

Reusable shell entry points for running LOCOS experiments on a local machine or
on any GPU host where the repository has been installed.

## Quick Start

Install the package and dependencies first; see the repository root README and
`CLAUDE.md` for local setup commands. Then run either the Python modules
directly or one of the scripts in `deploy/jobs/` with the environment variables
it expects.

For example:

```bash
source .venv/bin/activate

MODEL="meta-llama/Meta-Llama-3-8B-Instruct" \
HEADS="retrieval_heads/Meta-Llama-3-8B-Instruct.json" \
GPUS=1 \
MODEL_SLUG="meta-llama-3-8b-instruct" \
HF_RESULTS_REPO="aryopg/locos-results" \
./deploy/jobs/eval_nq_swap.sh
```

## File Overview

```text
deploy/
  job_config.sh          # shared helpers: model registry, GPU counts, setup commands
  model_sampling.yaml    # recommended sampling parameters for eval runs
  jobs/                  # experiment scripts, one per experiment type
```

## Job Scripts

Job scripts live in `deploy/jobs/`. Each script is a self-contained bash entry
point for one experiment type. They assume the repository is already available,
the Python environment is active, and any required secrets such as `HF_TOKEN`
are present in the shell environment.

Common environment variables:

| Variable | Description |
| --- | --- |
| `MODEL` | HuggingFace model name, such as `meta-llama/Meta-Llama-3-8B-Instruct` |
| `HEADS` | Retrieval-head JSON path relative to the repo root, or `random` where supported |
| `GPUS` | Number of GPUs to use; also passed as tensor parallel size |
| `MODEL_SLUG` | Short model name for output paths |
| `HF_RESULTS_REPO` | HuggingFace repo for uploading results |

See `deploy/jobs/README.md` for the full template and the list of available
scripts.

## Writing A Job Script

Create a file in `deploy/jobs/`:

```bash
#!/usr/bin/env bash
# Brief description of the experiment
set -euo pipefail

echo "=== My experiment: ${MODEL} ==="

python my_script.py --model "${MODEL}" --some-flag "${MY_FLAG:-default}"

echo "=== Uploading results ==="
python scripts/upload_results.py ./my_results \
    --repo-id "${HF_RESULTS_REPO}" \
    --path-in-repo "${MODEL_SLUG}/my-experiment"
```

Run it from the repository root after setting the required environment
variables:

```bash
MODEL="Qwen/Qwen3-8B" \
HEADS="retrieval_heads/Qwen3-8B.json" \
GPUS=1 \
MODEL_SLUG="qwen3-8b" \
HF_RESULTS_REPO="aryopg/locos-results" \
./deploy/jobs/my_experiment.sh
```
