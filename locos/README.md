# Retrieval Head Detection

Standalone toolkit for identifying **retrieval heads** in transformer language
models — attention heads that are disproportionately responsible for copying
information from context into the model's predictions.

Detected heads are saved as JSON files consumed by LOCOS's ablation
decoding (`locos_eval`).

## Detection Methods

### 1. Behavioral Retrieval Score (`detectors/behavioral.py`)

Faithful reimplementation of
[Wu et al. / nightdessert/Retrieval_Head](https://github.com/nightdessert/Retrieval_Head).
Inserts a known "needle" fact into a haystack of text, then measures how much
each attention head focuses on the needle tokens when generating the answer.

```bash
# NIAH dataset (default)
python locos/detectors/behavioral.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct

# NoLiMa dataset (harder, multi-hop probes)
python locos/detectors/behavioral.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct --dataset nolima
```

### 2. Contrastive Detection (`detectors/contrastive.py`)

Novel approach: compares attention patterns between trials with the correct
answer present vs. absent in the context. Heads whose attention shifts
significantly are flagged as retrieval heads.

```bash
python locos/detectors/contrastive.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct --dataset nolima
```

### 3. Logit-Contribution Scoring (`detectors/logit_contrib.py`)

Measures whether a head's output at needle positions pushes the residual
stream toward the correct answer token in the unembedding space. Uses a
spatial contrast (needle vs off-needle positions) rather than temporal
(answer vs non-answer steps). See `docs/contrastive_logit_contribution_scoring.md`.

```bash
python locos/detectors/logit_contrib.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct --dataset nolima
```

### 4. Causal Retrieval Importance (`detectors/cri.py`)

Activation patching: captures each head's output on a clean forward pass,
then patches individual heads with corrupted activations and measures the
drop in answer log-probability.

```bash
python locos/detectors/cri.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct --dataset nolima
```

## Supporting Tools

| Script | Purpose |
|--------|---------|
| `utils/datasets.py` | Shared dataset abstraction (`RetrievalTrial` dataclass, NIAH/NoLiMa builders, stratified sampling) |
| `utils/common.py` | Checkpoint save/load, model loading, config extraction, head score I/O |
| `utils/needle_utils.py` | Needle insertion & position tracking |
| `utils/model_utils.py` | Model introspection helpers (device, attention impl, BOS detection) |
| `download_haystack_data.py` | Download needle/haystack data for NIAH and NoLiMa probing datasets |
| `analysis/compare_heads.py` | Compare two detection runs (e.g., NIAH vs NoLiMa, or two methods) |
| `analysis/nolima_ablation.py` | Ablation study: mask top-N heads and measure generation quality degradation |
| `plotting/score_dist.py` | Visualize per-head score distributions |
| `plotting/score_buckets.py` | Heatmap of head scores bucketed by layer and head index |
| `plotting/nolima_ablation.py` | Plot ablation study results |

## Quick Start

```bash
# 1. Install dependencies (from repo root)
pip install -e ".[eval]"

# 2. Download probing data
python locos/download_haystack_data.py --dataset all

# 3. Run detection (requires GPU)
python locos/detectors/behavioral.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset nolima

# 4. Output: retrieval_heads/Meta-Llama-3-8B-Instruct_nolima.json
```

## Output Format

Each detection method outputs a JSON file mapping `"layer-head"` keys to
per-trial score lists:

```json
{
  "0-5":  [0.92, 0.88, 0.91, ...],
  "3-12": [0.78, 0.81, 0.75, ...],
  ...
}
```

These files are loaded by `locos_eval.retrieval_heads.load_retrieval_heads()`
for use in contrastive decoding.

## Dependencies

Requires GPU access and the following packages (beyond `locos_eval` base):

- `transformers` (model loading with `output_attentions=True`)
- `torch`
- `rouge-score` (needle detection scoring)
- `numpy`
- `rich` (progress bars and tables)
- `matplotlib` (plotting scripts only)
