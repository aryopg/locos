# LOCOS Package

`locos` contains the retrieval-head detectors, analysis scripts, plotting
helpers, and dataset utilities used by the LOCOS release. The companion
`locos_eval` package contains the vLLM ablation wrapper and downstream eval
runners that consume detected heads.

LOCOS identifies retrieval heads by scoring whether a head writes answer
evidence through its OV path at source positions. The score is a spatial
contrast: contribution on needle/source tokens versus contribution on
off-source tokens. Ablation then tests whether the selected heads are causally
important for contextual retrieval.

## Detection Methods

| Method | Module | Role |
| --- | --- | --- |
| LOCOS | `locos.detectors.logit_contrib` | Main logit-contribution spatial scorer |
| Attention spatial | `locos.detectors.attention_spatial` | Same spatial contrast using attention weights only |
| DLA | `locos.detectors.dla` | Direct logit attribution without spatial contrast |
| Wu behavioral | `locos.detectors.behavioral` | Token-match retrieval-head baseline |
| Contrastive attention | `locos.detectors.contrastive` | Answer-contingent attention baseline |
| HeadKV | `locos.detectors.headkv` | Anchor-window baseline |
| CRI | `locos.detectors.cri` | Activation-patching baseline |

Run LOCOS on NoLiMa:

```bash
python -m locos.detectors.logit_contrib \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset nolima \
    --num-examples 200 \
    --resume
```

Run the attention-only spatial control:

```bash
python -m locos.detectors.attention_spatial \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset nolima \
    --num-examples 200 \
    --resume
```

## Output Format

Detectors write JSON files under `retrieval_heads/` by default. LOCOS outputs
usually follow this pattern:

```text
retrieval_heads/<ModelName>_logit_contrib_nolima.json
```

Each file maps `"layer-head"` keys to per-trial score lists, with optional
metadata depending on the detector:

```json
{
  "0-5": [0.92, 0.88, 0.91],
  "3-12": [0.78, 0.81, 0.75]
}
```

Load heads for ablation with `locos_eval.retrieval_heads.load_retrieval_heads()`:

```python
from locos_eval import load_retrieval_heads

heads = load_retrieval_heads(
    "retrieval_heads/Meta-Llama-3-8B-Instruct_logit_contrib_nolima.json",
    num_heads=50,
)
```

## Supporting Modules

| Path | Purpose |
| --- | --- |
| `utils/datasets.py` | Shared NIAH/NoLiMa trial builders and stratified sampling |
| `utils/common.py` | Checkpoint save/load, model loading, and head score I/O |
| `utils/needle_utils.py` | Needle insertion and source-position tracking |
| `utils/model_utils.py` | Model introspection helpers |
| `analysis/nolima_ablation.py` | NoLiMa/NIAH ablation sweeps |
| `analysis/parametric_ablation.py` | Parametric and arithmetic specificity controls |
| `plotting/` | Figure generation helpers |

## Reproducibility

Public heads, ablation outputs, downstream outputs, and manifests live in the
HuggingFace dataset repo `aryopg/locos-results`. See the root
`REPRODUCING.md` and `experiments/manifest.yaml` for the release artifact
layout and figure regeneration commands.
