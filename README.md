# LOCOS

**Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads**

LOCOS is a toolkit for finding attention heads that matter for contextual retrieval, then testing those heads with mean or zero ablation during vLLM inference. The project page is the best visual overview; this README is the practical guide for running and extending the code.

| Resource | Path |
| --- | --- |
| Project page | [web/index.html](web/index.html) |
| Paper | https://arxiv.org/abs/2607.01002 |
| Results | https://huggingface.co/datasets/aryopg/locos_results |
| Demo notebook | [notebooks/locos_demo.ipynb](notebooks/locos_demo.ipynb) |
| Reproducibility guide | [REPRODUCING.md](REPRODUCING.md) |
| Figure manifest | [experiments/manifest.yaml](experiments/manifest.yaml) |
| vLLM ablation example | [examples/ablate_generation.py](examples/ablate_generation.py) |

## At A Glance

LOCOS scores a head by whether its OV path writes evidence for the generated answer token at source positions. The core per-position term is:

```text
phi[t,j,l,h] = alpha[t,j,l,h] * u[y_t]^T * W_O[l,h] * v[j,l,h]
```

The head score is the spatial contrast between needle/source positions and off-source positions. That distinction is what separates LOCOS from attention-only scoring and from direct logit attribution without spatial contrast.

This repo includes:

- retrieval-head detectors under `locos.detectors`
- ablation and downstream eval wrappers under `locos_eval`
- analysis and plotting scripts under `locos.analysis` and `locos.plotting`
- deployment job scripts under `deploy/jobs`
- a static project page under `web/`

## Quickstart

For the public release path, use a GPU machine with CUDA 12.x and vLLM support:

Use Python 3.11 or newer. GPU runs need a CUDA/vLLM-compatible environment; CPU-only machines are still useful for docs, plotting, notebook structure checks, and unit tests that do not touch model execution.

```bash
# Create the release environment.
make venv-gpu
source .venv/bin/activate

# Check that the public API and CPU-safe pieces import.
python -m pytest tests/test_standalone_surface.py tests/test_eval_runner.py -q
```

Download a released heads file and inspect the selected heads:

```bash
python scripts/download_heads.py \
    --repo-id aryopg/locos-results \
    --heads retrieval_heads/Qwen3-8B_logit_contrib_nolima.json

python - <<'PY'
from locos_eval import load_retrieval_heads

heads = load_retrieval_heads("retrieval_heads/Qwen3-8B_logit_contrib_nolima.json", num_heads=10)
print(heads)
PY
```

Then run either a small downstream smoke test:

```bash
python -m locos_eval.evals.tasks.babilong_task \
    --model Qwen/Qwen3-8B \
    --heads retrieval_heads/Qwen3-8B_logit_contrib_nolima.json \
    --decoding ablation \
    --ablation-mode mean \
    --num-heads 50 \
    --subset qa2 \
    --split 0k \
    --limit 2 \
    --tp 1
```

or a small LOCOS detection run:

```bash
python locos/download_haystack_data.py --dataset nolima
python -m locos.detectors.logit_contrib \
    --model Qwen/Qwen3-8B \
    --dataset nolima \
    --num-examples 2 \
    --max-length 4000
```

Full reproduction commands, artifact downloads, and figure regeneration are in [REPRODUCING.md](REPRODUCING.md).

## Install Notes

The Makefile has two environment targets:

```bash
# GPU host, full release/eval stack including vLLM.
make venv-gpu

# CPU-only local development, useful for docs and import/unit tests.
make venv HOST_PYTHON=python3.11
```

For local Mac development without vLLM/GPU work, install only what your environment supports and run CPU-safe tests with `make test ARGS="tests/test_standalone_surface.py"`.

This repository intentionally does not commit `uv.lock`: the CUDA, PyTorch, and
vLLM wheel set is platform-specific. The release install recipe is documented in
[REPRODUCING.md](REPRODUCING.md).

## Common Commands

The Makefile is the shortest public interface:

```bash
make venv-gpu
make data DATASET=nolima
make detect MODEL=Qwen/Qwen3-8B DATASET=nolima
make ablate MODEL=Qwen/Qwen3-8B HEADS=retrieval_heads/Qwen3-8B_logit_contrib_nolima.json
make test
make coverage
make notebook
```

Override the interpreter if needed:

```bash
make test PYTHON=/path/to/python
```

## Detection

Download probing data:

```bash
python locos/download_haystack_data.py --dataset nolima
python locos/download_haystack_data.py --dataset niah
```

Run LOCOS:

```bash
python -m locos.detectors.logit_contrib \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset nolima \
    --num-examples 200 \
    --resume
```

Outputs are written to `retrieval_heads/`, usually as:

```text
retrieval_heads/<ModelName>_logit_contrib_nolima.json
```

Released retrieval heads are also available from the HuggingFace dataset repo
`aryopg/locos-results` under `retrieval_heads/`.

Supported detector entrypoints:

| Method | Command | Role |
| --- | --- | --- |
| LOCOS | `python -m locos.detectors.logit_contrib` | Main method |
| Attention spatial | `python -m locos.detectors.attention_spatial` | LOCOS without OV contribution |
| DLA | `python -m locos.detectors.dla` | LOCOS without spatial contrast |
| Wu behavioral | `python -m locos.detectors.behavioral` | Token-match retrieval-head baseline |
| Contrastive attention | `python -m locos.detectors.contrastive` | Answer-contingent attention baseline |
| HeadKV | `python -m locos.detectors.headkv` | Anchor-window baseline |
| CRI | `python -m locos.detectors.cri` | Activation-patching baseline |

Load detected heads in Python:

```python
from locos_eval import load_retrieval_heads

heads = load_retrieval_heads(
    "retrieval_heads/Meta-Llama-3-8B-Instruct_logit_contrib_nolima.json",
    num_heads=50,
)
```

## Ablation

Ablation tests whether selected heads are causally important by replacing their query activations during generation. `locos_eval.ablation(...)` wraps a `vllm.LLM` instance and keeps vLLM's native scheduler/paged-attention path.

```python
from vllm import LLM
from locos_eval import ablation, load_retrieval_heads

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", enforce_eager=True)
heads = load_retrieval_heads("retrieval_heads/Meta-Llama-3-8B-Instruct_logit_contrib_nolima.json", num_heads=50)

with ablation(
    llm,
    heads=heads,
    decoding="ablation",
    ablation_mode="mean",
    calibration_prompts=["A short calibration prompt."],
) as generator:
    print(generator.generate("Question: What is the capital of France?\nAnswer:", max_tokens=32))
```

Run a downstream eval with ablation:

```bash
python -m locos_eval.evals.tasks.babilong_task \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --heads retrieval_heads/Meta-Llama-3-8B-Instruct_logit_contrib_nolima.json \
    --decoding ablation \
    --ablation-mode mean \
    --num-heads 50 \
    --subset qa2 \
    --split 0k \
    --tp 1
```

Released ablation outputs, downstream outputs, and per-experiment manifests are
available from `aryopg/locos-results`:

```text
retrieval_heads/
ablation_results/
ablation_parametric_results/
downstream_results/
```

Useful eval task modules:

```text
locos_eval.evals.tasks.babilong_task
locos_eval.evals.tasks.musique_task
```

## Analysis And Plotting

Run the main NoLiMa ablation sweep:

```bash
python locos/analysis/nolima_ablation.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --heads retrieval_heads/Meta-Llama-3-8B-Instruct_logit_contrib_nolima.json \
    --mode top-k \
    --values 1 5 10 20 50 100 \
    --ablation-mode mean \
    --include-baseline
```

Run the parametric/arithmetic specificity control:

```bash
python scripts/eval/build_parametric_and_arithmetic_dataset.py

python locos/analysis/parametric_ablation.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --heads retrieval_heads/Meta-Llama-3-8B-Instruct_logit_contrib_nolima.json \
    --mode top-k \
    --values 1 5 10 20 50 100 \
    --ablation-mode mean \
    --include-baseline
```

These controls use the public HuggingFace dataset
`aryopg/parametric-arithmetic-eval`, built by
`scripts/eval/build_parametric_and_arithmetic_dataset.py` from City-Country,
PopQA, and arithmetic samples. `locos/analysis/parametric_ablation.py` uses this
dataset by default; rebuild it only if you want to audit or modify the control
set.

Plotting scripts save figures under `figures/` and usually write legends as sibling files. Downstream plotting defaults to:

```text
../locos-results/downstream_results
```

Override it with:

```bash
export LOCOS_DOWNSTREAM_DIR=/path/to/downstream_results
```

Examples:

```bash
python -m locos.plotting.babilong_bar
python -m locos.plotting.musique_bar
```

The commands and input artifacts for the public `web/assets/*` figures are
listed in [experiments/manifest.yaml](experiments/manifest.yaml).

## Project Page

The static page lives in `web/`.

```bash
cd web
python3 -m http.server 8772
```

Open:

```text
http://127.0.0.1:8772/
```

Figures used by the page are copied into `web/assets/`.

## Repository Map

```text
locos/
  detectors/       LOCOS and baseline retrieval-head detectors
  analysis/        ablation, specificity, inventory, and revision analyses
  plotting/        publication and downstream plotting helpers
  utils/           dataset, model, checkpoint, and needle utilities

locos_eval/
  wrapper.py       GreedyWrapper, AblationWrapper, AblationRPCWrapper
  ablation.py      native vLLM zero/mean query-ablation hooks
  rpc_ops.py       tensor-parallel hook installation via collective_rpc
  evals/           downstream eval runner, tasks, configs, prompts

scripts/
  eval/            dataset builders and eval utilities
  download_heads.py
  upload_results.py
  sync_results.py

deploy/
  jobs/            GPU job scripts for detection, ablation, and evals

web/
  index.html       static project page
  assets/          paper figures for the page
```

## Development

CPU-safe checks:

```bash
python3.11 -m compileall locos locos_eval scripts examples
ruff check locos locos_eval scripts tests
python3.11 -m pytest tests/test_retrieval_heads.py tests/test_eval_runner.py tests/test_dla.py tests/test_standalone_surface.py
```

Detector helper tests:

```bash
python3.11 -m pytest \
    tests/test_attention_spatial.py \
    tests/test_headkv.py \
    tests/test_detect_cri.py \
    tests/test_logit_contrib.py
```

GPU/vLLM integration tests should be run on a machine with a compatible vLLM install and model access.

Coverage target:

```bash
make coverage PYTHON=python3.11
```

## Reproducibility

Use [REPRODUCING.md](REPRODUCING.md) for the full release checklist, GPU smoke
tests, artifact download commands, and figure regeneration commands.

## Citation And License

If you use this code, please cite LOCOS and PopQA:

```bibtex
@article{gema2026locos,
      title={Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads}, 
      author={Aryo Pradipta Gema and Beatrice Alex and Pasquale Minervini},
      year={2026},
      eprint={2607.01002},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.01002}, 
}
```

Preprint citation metadata is in [CITATION.cff](CITATION.cff). The arXiv
preprint is available at <https://arxiv.org/abs/2607.01002>.

Source code is MIT licensed; documentation, web text, figures, and non-code
assets are CC BY 4.0 licensed. See [LICENSE](LICENSE) and
[LICENSE-assets.md](LICENSE-assets.md).
