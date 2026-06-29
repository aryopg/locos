# Reproducing LOCOS

This guide describes the public artifact layout for reproducing the LOCOS
figures and downstream ablation claims. The canonical result repository is the
HuggingFace dataset repo `aryopg/locos-results`.

## Environment

Use Python 3.11 or newer. GPU experiments require CUDA 12.x, PyTorch wheels
compatible with CUDA 12.8, and `vllm>=0.18,<0.19`. The project does not commit a
`uv.lock` because the CUDA, PyTorch, and vLLM wheel set is platform-specific.
Use this install recipe for the public release environment:

```bash
uv venv --python python3.11 --system-site-packages
uv pip install -e ".[dev,eval]" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    --python .venv/bin/python
source .venv/bin/activate
```

Editable installs also work with pip:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,eval]" \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

Expected GPU counts are encoded in `deploy/job_config.sh`. Small 4B-14B models
usually run with TP=1, 27B-32B models with TP=2, and `openai/gpt-oss-120b` with
TP=4. Ablation runs require `--enforce-eager`, which is the default in the eval
runner.

## Artifact Layout

`aryopg/locos-results` uses these public prefixes:

```text
retrieval_heads/                 detected head scores and array sidecars
ablation_results/                NoLiMa and NIAH ablation sweeps
ablation_parametric_results/     parametric and arithmetic specificity controls
downstream_results/              BABILong and MuSiQue outputs used by the
                                 public downstream figure
logs/                            optional job logs
```

Fetch one heads file:

```bash
python scripts/download_heads.py \
    --repo-id aryopg/locos-results \
    --heads retrieval_heads/Qwen3-8B_logit_contrib_nolima.json
```

Fetch arbitrary artifacts with `huggingface_hub`:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="aryopg/locos-results",
    repo_type="dataset",
    allow_patterns=[
        "retrieval_heads/*logit_contrib_nolima.json",
        "ablation_results/*.json",
        "ablation_parametric_results/*.json",
        "downstream_results/**/*.jsonl",
        "downstream_results/**/manifest.json",
    ],
    local_dir="../locos-results",
)
PY
```

## Detection

Download probing data:

```bash
python locos/download_haystack_data.py --dataset nolima
python locos/download_haystack_data.py --dataset niah
```

Run LOCOS on NoLiMa:

```bash
python -m locos.detectors.logit_contrib \
    --model Qwen/Qwen3-8B \
    --dataset nolima \
    --nolima-variant needle_set \
    --question-type onehop \
    --min-length 1000 \
    --max-length 50000 \
    --num-lengths 20 \
    --num-depths 10 \
    --num-examples 200 \
    --max-decode-steps 50 \
    --rouge-threshold 50 \
    --seed 42 \
    --resume
```

Baselines use the same dataset and seed knobs:

```bash
python -m locos.detectors.attention_spatial --model Qwen/Qwen3-8B --dataset nolima --num-examples 200 --seed 42 --resume
python -m locos.detectors.dla --model Qwen/Qwen3-8B --dataset nolima --num-examples 200 --seed 42 --resume
python -m locos.detectors.behavioral --model Qwen/Qwen3-8B --dataset nolima --num-examples 200 --seed 42 --resume
python -m locos.detectors.cri --model Qwen/Qwen3-8B --dataset nolima --num-examples 200 --context-length 4000 --num-depths 5 --seed 42 --resume
```

Default outputs live under `retrieval_heads/`, for example
`retrieval_heads/Qwen3-8B_logit_contrib_nolima.json`.

## Ablation

Headline NoLiMa/NIAH sweeps use mean ablation, 50 calibration trials, seed 42,
and k values `1 5 10 20 50 100` unless a figure notes otherwise:

```bash
python locos/analysis/nolima_ablation.py \
    --model Qwen/Qwen3-8B \
    --heads retrieval_heads/Qwen3-8B_logit_contrib_nolima.json \
    --dataset nolima \
    --mode top-k \
    --values 1 5 10 20 50 100 \
    --ablation-mode mean \
    --num-calibration 50 \
    --include-baseline \
    --seed 42
```

Controls:

```bash
# Bottom-k specificity control.
python locos/analysis/nolima_ablation.py \
    --model Qwen/Qwen3-8B \
    --heads retrieval_heads/Qwen3-8B_logit_contrib_nolima.json \
    --mode bottom-k \
    --values 1 5 10 20 50 100 \
    --ablation-mode mean \
    --num-calibration 50 \
    --seed 42

# Random-head control. Use separate seeds for independent runs.
python locos/analysis/nolima_ablation.py \
    --model Qwen/Qwen3-8B \
    --random-heads \
    --mode top-k \
    --values 1 5 10 20 50 100 \
    --ablation-mode mean \
    --num-calibration 50 \
    --seed 42

# Attention-spatial and DLA controls reuse their corresponding heads JSON.
python locos/analysis/nolima_ablation.py \
    --model Qwen/Qwen3-8B \
    --heads retrieval_heads/Qwen3-8B_attention_spatial_nolima.json \
    --mode top-k \
    --values 1 5 10 20 50 100 \
    --ablation-mode mean \
    --num-calibration 50 \
    --seed 42
```

Parametric and arithmetic controls:

```bash
python scripts/eval/build_parametric_and_arithmetic_dataset.py

python locos/analysis/parametric_ablation.py \
    --model Qwen/Qwen3-8B \
    --heads retrieval_heads/Qwen3-8B_logit_contrib_nolima.json \
    --mode top-k \
    --values 1 5 10 20 50 100 \
    --ablation-mode mean \
    --num-calibration 50 \
    --include-baseline \
    --seed 42
```

## Downstream Evals

All downstream task runners share these core flags:

```bash
--model <HF model id>
--heads <retrieval_heads JSON or random>
--decoding greedy|ablation
--ablation-mode mean
--num-heads 50
--num-calibration 50
--tp <tensor parallel size>
--temperature 0.0
--sampling-top-p 1.0
--sampling-top-k -1
--output-dir downstream_results
```

Task commands:

```bash
python -m locos_eval.evals.tasks.babilong_task --model Qwen/Qwen3-8B --heads retrieval_heads/Qwen3-8B_logit_contrib_nolima.json --decoding ablation --ablation-mode mean --num-heads 50 --subset qa2 --split 0k --tp 1 --output-dir downstream_results
python -m locos_eval.evals.tasks.musique_task --model Qwen/Qwen3-8B --heads retrieval_heads/Qwen3-8B_logit_contrib_nolima.json --decoding ablation --ablation-mode mean --num-heads 50 --subset answerable --split validation --tp 1 --output-dir downstream_results
```

Use `--limit 2` for GPU smoke tests. Upload/sync downstream outputs with:

```bash
python scripts/sync_results.py \
    --repo-id aryopg/locos-results \
    --local-dir downstream_results \
    --hf-prefix downstream_results
```

## Plotting

`experiments/manifest.yaml` maps each public `web/assets/*` figure to its input
artifact paths and plotting command. After downloading `aryopg/locos-results`
to `../locos-results`, regenerate the web figures with the commands in that
manifest. Downstream plotters default to:

```text
../locos-results/downstream_results
```

Override with:

```bash
export LOCOS_DOWNSTREAM_DIR=/path/to/downstream_results
```

## Checks Before Advertising

Run these CPU-safe checks before advertising the repository:

```bash
ruff check locos locos_eval scripts tests
python3.11 -m compileall locos locos_eval scripts examples
python -m json.tool notebooks/locos_demo.ipynb >/dev/null
python3.11 -m pytest tests/test_standalone_surface.py tests/test_retrieval_heads.py tests/test_eval_tasks.py tests/test_logit_contrib.py tests/test_dla.py
python -m locos.detectors.logit_contrib --help
python -m locos.detectors.attention_spatial --help
python -m locos.detectors.dla --help
python -m locos_eval.evals.tasks.babilong_task --help
python -m locos_eval.evals.tasks.musique_task --help
```

GPU smoke tests should include one tiny LOCOS detection run, one greedy eval
run, one mean-ablation eval run with `--limit 2`, and one plotting command
against artifacts fetched from `aryopg/locos-results`.
