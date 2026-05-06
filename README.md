# LOCOS: Detecting Retrieval Heads via Logit Contribution Spatial Scoring

**LOCOS** identifies *retrieval heads* — the attention heads in a large language model that are causally responsible for retrieving relevant information from context during generation.

Unlike prior methods that score heads by whether they attend to the right tokens (Wu et al.) or by how much their attention shifts during answer generation (contrastive), LOCOS measures how much each head's output-value projection actually changes the model's next-token distribution towards a context token. The score at position `j` for head `(l, h)` is:

```
φ_{t,j}^{(l,h)} = α_{t,j}^{(l,h)} · u_{y_t}^T · W_O^{(l,h)} · v_j^{(l,h)}
```

where `α` is the attention weight, `v` is the value vector, `W_O` is the output projection, and `u_{y_t}` is the unembedding direction for the generated token. The per-head score is the spatial contrast between needle positions and off-needle positions. This makes LOCOS effective even when there is no lexical overlap between the question and the retrieved passage — a setting where Wu et al. scores collapse.

---

## Quick start

**1. Install**

```bash
# macOS / local dev (no GPU required for detection if you have a server)
uv venv --python /path/to/python3.11 --system-site-packages
uv pip install -e ".[dev]" --python .venv/bin/python --no-deps

# GPU server (CUDA 12.8, for running detectors)
uv venv --python python3.12 --system-site-packages
uv pip install -e ".[dev,eval]" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    --python .venv/bin/python
```

**2. Download probing data**

```bash
python locos/download_haystack_data.py --dataset nolima
```

**3. Run LOCOS**

```bash
python -m locos.detectors.logit_contrib \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset nolima
# → writes retrieval_heads/Meta-Llama-3-8B-Instruct_logit_contrib_nolima.json
```

**4. Use the detected heads to ablate at inference time**

```bash
python -m locos_eval.evals.tasks.nq_swap_task \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --heads retrieval_heads/Meta-Llama-3-8B-Instruct_logit_contrib_nolima.json \
    --decoding ablation --num-heads 50 --tp 4
```

---

## Repository structure

```
locos/       Detection methods and ablation analysis
  detectors/
    logit_contrib.py            LOCOS (this paper)
    behavioral.py               Wu et al. reimplementation (NIAH baseline)
    contrastive.py              Contrastive attention scoring
    cri.py                      Causal Retrieval Importance (activation patching)
    attention_spatial.py        Attention-only spatial baseline (no OV term)
    headkv.py                   HeadKV/SnapKV anchor-window baseline
  analysis/
    nolima_ablation.py          Ablation on NoLiMa retrieval (ROUGE-L sweep)
    parametric_ablation.py      Specificity control: parametric + arithmetic accuracy
    compare_heads.py            Compare two head JSON files
  plotting/                     Matplotlib figure scripts
  utils/                        Shared datasets, model utilities, needle insertion

locos_eval/
  evals/                        Downstream eval framework (greedy vs ablation)
    tasks/                      NQ-Swap, MedRAG, XSum, ACI-Bench, LongBench-v2, …
  wrapper.py                    GreedyWrapper + AblationWrapper + AblationRPCWrapper
  ablation.py                   Q-zeroing / Q-mean hooks (no sequential KV cache)
  rpc_ops.py                    Multi-GPU (TP>1) ablation via collective_rpc

scripts/
  download_heads.py             Fetch head JSON files from HuggingFace Hub
  upload_results.py             Upload results to HuggingFace Hub
  eval/                         Dataset builders, result explorers, plotting
```

---

## Detecting retrieval heads

### Probing datasets

Detection methods probe the model by inserting a *needle* (a fact to retrieve) into a long *haystack* (irrelevant text) and measuring each head's contribution. Two datasets are supported:

| Dataset | Flag | Description |
|---------|------|-------------|
| **NIAH** | `--dataset niah` | Classic Needle-in-a-Haystack. High lexical overlap between question and needle — easy for any attention-based method. |
| **[NoLiMa](https://github.com/adobe-research/NoLiMa)** | `--dataset nolima` | "No Literal Matching" (ICML 2025). Question uses different vocabulary from the needle. Models drop 30–70% accuracy vs NIAH. **Use this for non-literal retrieval.** |

```bash
# Download both datasets
python locos/download_haystack_data.py --dataset niah
python locos/download_haystack_data.py --dataset nolima
```

### LOCOS (recommended)

```bash
python -m locos.detectors.logit_contrib \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset nolima \
    --num-examples 200          # number of trials (default: all available)
    --context-length 4000       # context length in tokens
```

Output: `retrieval_heads/<ModelName>_logit_contrib_nolima.json`

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `nolima` | Probing dataset: `niah` or `nolima` |
| `--num-examples` | all | Number of trials (stratified by length × depth) |
| `--context-length` | varies | Context length in tokens |
| `--prompt-suffix` | `""` | Text appended to the prompt before generation (e.g. `"Answer:"`) |
| `--use-chat-template` | auto | Apply the model's chat template to prompts |
| `--resume` | off | Resume from an existing checkpoint |

### Wu et al. behavioral scoring

```bash
python -m locos.detectors.behavioral \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset niah \
    --min-length 1000 --max-length 50000
```

Reimplementation of [nightdessert/Retrieval_Head](https://github.com/nightdessert/Retrieval_Head). Scores heads by whether their argmax-attended token matches the generated token. Works well on NIAH; degrades sharply on NoLiMa because the attended token rarely matches the generated token lexically.

### CRI — Causal Retrieval Importance

```bash
python -m locos.detectors.cri \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset nolima \
    --num-examples 200 \
    --context-length 4000
```

Activation patching: clean forward captures per-head activations; corrupted forward (needle replaced with filler) establishes baseline; per-head patching measures how much restoring a head's clean activation recovers retrieval. Causally grounded but slow — requires H+2 forward passes per example, where H = layers × heads.

### Contrastive attention scoring

```bash
python -m locos.detectors.contrastive \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset nolima
```

Measures whether a head's attention to the needle is *answer-contingent* — higher during answer-generating decode steps than during other steps. Works for non-literal retrieval.

### Reviewer-requested baselines

```bash
# Attention-only spatial (LOCOS without the OV projection term)
python -m locos.detectors.attention_spatial \
    --model meta-llama/Meta-Llama-3-8B-Instruct --dataset nolima

# HeadKV/SnapKV-style anchor-window
python -m locos.detectors.headkv \
    --model meta-llama/Meta-Llama-3-8B-Instruct --dataset nolima
```

### Loading heads in Python

```python
from locos_eval import load_retrieval_heads

# Keep all heads with mean score ≥ 0.4 (default threshold)
heads = load_retrieval_heads("retrieval_heads/Meta-Llama-3-8B-Instruct_logit_contrib_nolima.json")

# Top-50 heads by score
heads = load_retrieval_heads("...", num_heads=50)

# Returns: list of (layer_idx, head_idx) tuples, sorted by importance score
```

### Fetching pre-computed heads from HuggingFace

TBD

---

## Ablation experiments

Ablation experiments verify that the detected heads are *causally necessary* for retrieval by zeroing or mean-replacing their query projections and measuring performance.

### NoLiMa ablation — does masking these heads hurt retrieval?

```bash
# Sweep top-k heads, measure ROUGE-L on NoLiMa
python locos/analysis/nolima_ablation.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --heads retrieval_heads/Meta-Llama-3-8B-Instruct_logit_contrib_nolima.json \
    --mode top-k --values 1 5 10 20 50 100 \
    --ablation-mode mean \
    --include-baseline

# Random heads control (same sweep, random head selection)
python locos/analysis/nolima_ablation.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --random-heads --seed 42 \
    --mode top-k --values 1 5 10 20 50 100 \
    --ablation-mode mean \
    --include-baseline
```

Results are cached to `ablation_results/` — re-running is instant.

### Parametric/arithmetic control — are the heads retrieval-specific?

```bash
# Build eval dataset (one-time, ~5 min)
python scripts/eval/build_parametric_and_arithmetic_dataset.py

# Run ablation: measure parametric recall and arithmetic accuracy
python locos/analysis/parametric_ablation.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --heads retrieval_heads/Meta-Llama-3-8B-Instruct_logit_contrib_nolima.json \
    --mode top-k --values 1 5 10 20 50 100 \
    --ablation-mode mean \
    --include-baseline
```

Expected finding: LOCOS heads cause large NoLiMa ROUGE-L drops but leave parametric recall and arithmetic largely intact (dissociation), confirming retrieval specificity.

### Plotting

```bash
# NoLiMa ROUGE-L vs parametric accuracy (dissociation plot)
python locos/plotting/parametric_ablation.py \
    --nolima ablation_results/nolima_ablation_logitcontrib_nolima.json \
    --parametric ablation_parametric_results/parametric_ablation_logitcontrib_nolima.json

# NIAH vs NoLiMa comparison across detection methods
python locos/plotting/ablation_comparison.py
```

Figures are saved to `figures/` with legends as separate files (no titles — add in LaTeX captions).

---

## Downstream evaluation

Compares greedy decoding vs retrieval-head ablation on standard NLP benchmarks to show that ablating LOCOS heads degrades context use across tasks.

### Installation

```bash
uv pip install -e ".[eval]" --python .venv/bin/python
# Requires GPU + vLLM ≥ 0.18
```

### Running evaluations

TBD

---

## Testing

```bash
# Unit tests (no GPU required — 448 tests)
source .venv/bin/activate && pytest tests/ -v -m "not gpu"

# GPU integration tests
TEST_MODEL=meta-llama/Meta-Llama-3-8B-Instruct \
HEADS_JSON=retrieval_heads/Meta-Llama-3-8B-Instruct.json \
pytest tests/ -v -m gpu
```

---

## Results

TBD

---

## Citation

```
TBD
```
