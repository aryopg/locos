# Deploy

Tools for launching experiments across multiple EIDF Kubernetes namespaces
and monitoring their status.

## Quick start

```bash
# 1. Configure your namespaces (see namespaces.conf section below)
vim deploy/namespaces.conf

# 2. Launch an experiment
./deploy/launch_multi_ns.sh deploy/jobs/detect_logit_contrib.sh --dry-run   # preview
./deploy/launch_multi_ns.sh deploy/jobs/detect_logit_contrib.sh             # launch

# 3. Monitor jobs
pip install textual   # or: uv pip install -e ".[monitor]"
python -m deploy.monitor
```

## File overview

```
deploy/
  namespaces.conf        # namespace configuration (SSH hosts, quotas, credentials)
  launch_multi_ns.sh     # generic multi-namespace job launcher
  launch_evals.sh        # legacy eval-only launcher (use launch_multi_ns.sh instead)
  job_config.sh          # shared helpers (model registry, GPU counts, setup commands)
  interactive.yaml       # Kubernetes manifest for interactive debug pods
  jobs/                  # experiment scripts (one per experiment type)
  monitor/               # Textual TUI for monitoring and managing jobs
```

## Launching experiments

### Step 1: Pick or write a job script

Job scripts live in `deploy/jobs/`. Each is a self-contained bash script that
runs one experiment type. See `deploy/jobs/README.md` for the full template
and available environment variables.

Existing scripts:

| Script | Description |
|--------|-------------|
| `eval_nq_swap.sh` | NQ-Swap context faithfulness eval |
| `eval_xsum.sh` | XSum summarization eval |
| `eval_medrag.sh` | MedRAG medical QA eval (all 5 sub-datasets) |
| `eval_aci_bench.sh` | ACI-Bench dialogue-to-note eval |
| `detect_logit_contrib.sh` | Contrastive logit-contribution detection |
| `detect_contrastive.sh` | Contrastive attention-based detection |
| `detect_cri.sh` | Causal Retrieval Importance detection |
| `detect_retrieval_heads.sh` | Wu et al. retrieval head detection |
| `ablation_nolima.sh` | NoLiMa ablation with varying head selections |

### Step 2: Launch

```bash
# Basic launch (iterates over all models in DEFAULT_MODELS)
./deploy/launch_multi_ns.sh deploy/jobs/detect_logit_contrib.sh

# Preview without launching
./deploy/launch_multi_ns.sh deploy/jobs/detect_logit_contrib.sh --dry-run

# Specific models only
MODELS="meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen3-8B" \
  ./deploy/launch_multi_ns.sh deploy/jobs/eval_nq_swap.sh

# Target a single namespace
./deploy/launch_multi_ns.sh deploy/jobs/eval_xsum.sh --namespace eidf106ns

# Pass extra environment variables to the job script
./deploy/launch_multi_ns.sh deploy/jobs/ablation_nolima.sh \
  --env LIMIT=50 --env ABLATION_MODE=mean

# Pin specific models to specific namespaces
./deploy/launch_multi_ns.sh deploy/jobs/eval_medrag.sh \
  --pin "google/gemma-3-27b-it=eidf106ns"
```

### Step 3: What happens

For each model, the launcher:

1. Selects a namespace (unlimited namespaces first, then limited up to quota)
2. Builds a pod command: repo clone + venv setup + your job script
3. Wraps everything with log capture (`tee` to file + upload)
4. SCPs the command to the cluster login node
5. Launches via `kblaunch` with the right GPU count, annotations, and secrets
6. Uploads the log file to HuggingFace Hub when done

### Writing a new job script

Create a file in `deploy/jobs/`:

```bash
#!/usr/bin/env bash
# Brief description of the experiment
set -euo pipefail

echo "=== My experiment: ${MODEL} ==="

# Run your experiment (env vars MODEL, HEADS, GPUS, MODEL_SLUG are available)
python my_script.py --model "${MODEL}" --some-flag "${MY_FLAG:-default}"

# Upload results (pods are ephemeral — this is mandatory!)
echo "=== Uploading results ==="
python scripts/upload_results.py ./my_results \
    --repo-id "${HF_RESULTS_REPO}" \
    --path-in-repo "${MODEL_SLUG}/my-experiment"
```

Then launch it:

```bash
./deploy/launch_multi_ns.sh deploy/jobs/my_experiment.sh --dry-run
```

> **Pods are ephemeral.** When the script exits (or crashes), the pod is
> destroyed. Every artifact must be uploaded before the script finishes.

### Resubmitting failed jobs

When a batch fails because of a transient cluster issue (e.g. a read-only
mount, a flaky node), open the monitor TUI (`python -m deploy.monitor`),
select Failed jobs (`space` to multi-select, or just put the cursor on
one), and press `R` (Shift+R) to resubmit them.

The TUI parses each failed job's old command to recover its launch
parameters, deletes the failed K8s Job, and re-launches it with a freshly
built command — so any fixes to `setup_commands` in `job_config.sh` since
the original launch take effect on the resubmitted run. Only jobs in the
`Failed` state are eligible.

## Namespace configuration

`deploy/namespaces.conf` defines your Kubernetes namespaces:

```
# namespace | ssh_host | queue | gpu_product | secrets | username | email | max_days | gpu_quota
eidf106ns | eidf-bea | eidf106ns-user-queue | NVIDIA-H100-80GB-HBM3 | aryo-secrets | aryo | aryo@ed.ac.uk | 0 | 0
eidf231ns | eidf-bi  | eidf231ns-user-queue | NVIDIA-H100-80GB-HBM3 | aryo-secrets | user | aryo@ed.ac.uk | 0 | 4
```

| Column | Description |
|--------|-------------|
| `namespace` | Kubernetes namespace name |
| `ssh_host` | SSH config entry for the cluster login node |
| `queue` | kblaunch queue name |
| `gpu_product` | GPU type (e.g., `NVIDIA-H100-80GB-HBM3`) |
| `secrets` | Kubernetes secrets name (contains `HF_TOKEN`, `GIT_TOKEN`, etc.) |
| `username` | Your username in this namespace (set as `eidf/user` annotation) |
| `email` | Your email (set as `eidf/email` annotation) |
| `max_days` | Admin-enforced pod lifetime limit in days (`0` = no limit) |
| `gpu_quota` | Soft self-governed GPU limit (`0` = no limit, prioritized for jobs) |

**Namespace prioritization:** Namespaces with `gpu_quota=0` (no limit) receive
jobs first via round-robin. Limited namespaces are filled up to their quota only
when needed.

## Installing kblaunch

The launcher relies on `kblaunch` being available on each cluster login node.
Install the custom fork (which adds annotation and secret-passing support)
once per cluster:

```bash
# SSH into each cluster login node
ssh eidf-bea   # repeat for eidf-bi, etc.

# Install kblaunch from the fork
pip install --user git+https://github.com/aryopg/kblaunch.git@main

# Verify
kblaunch --help
```

> **Note:** The upstream `kblaunch` does not support the `--annotation` and
> `--secret` flags used by `launch_multi_ns.sh`. You must use the fork above.

## SSH setup

Each `ssh_host` must have an entry in `~/.ssh/config`. The launcher uses SSH
ControlMaster to maintain persistent connections so you only enter your TOTP
once per session.

```bash
# Pre-establish connections (TOTP prompted once per host)
./deploy/launch_multi_ns.sh --connect

# Tear down when done
./deploy/launch_multi_ns.sh --disconnect
```

## Monitor TUI

A terminal UI for watching jobs across all namespaces.

```bash
# Install textual (one-time)
pip install textual  # or: uv pip install -e ".[monitor]"

# Launch
python -m deploy.monitor
```

### Keybindings

| Key | Action |
|-----|--------|
| `c` | **Connect** — establish SSH connections (TOTP prompted in terminal) |
| `r` | **Refresh** — fetch pod status and GPU usage from all namespaces |
| `/` | **Filter** — type to filter pods by name, namespace, or status |
| `space` | **Select** — toggle selection on current row (for bulk operations) |
| `m` | **Move** — move a Pending job to another namespace |
| `x` | **Delete** — delete selected job(s) with confirmation |
| `g` | **GPU stats** — show nvidia-smi output for a Running pod |
| `h` | **History** — show recent job history from local JSONL log |
| `d` | **Disconnect** — tear down all SSH sessions |
| `?` | **Help** — show keybinding reference |
| `q` | **Quit** |

### What the TUI shows

- **Resource bar** — GPU usage per namespace: total in use, yours, decore jobs, and quota
- **Pod table** — all `decore-*` pods with status, uptime, time remaining (color-coded
  warnings for namespaces with pod lifetime limits), and node
- **Log panel** — live streaming logs for Running pods, static tail for others

### Moving jobs

Select a Pending pod and press `m`. The TUI extracts the job spec from
Kubernetes, creates it in the target namespace, then deletes the original.
Running jobs cannot be moved. This is useful when a namespace is busy or
approaching its GPU quota.

### Bulk operations

Press `space` on individual pods to select them (marked with `*`), then press
`x` to delete all selected. There is no "select all" — each pod must be
selected individually.

## Legacy launcher

`deploy/launch_evals.sh` is the original eval-only launcher that predates the
generic script-based approach. It still works but only supports the hardcoded
eval task matrix. Use `launch_multi_ns.sh` for all new work.
