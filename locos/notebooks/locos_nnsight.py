"""LOCOS — logit-contribution retrieval head detection (nnsight reference notebook).

Companion artifact to ``locos/detectors/logit_contrib.py``.
Same technique, expressed via the `nnsight` mech-interp library instead of
plain HuggingFace + forward hooks. Demonstrates how to fold LOCOS into the
typical nnsight workflow that mech-interp practitioners already use for
activation patching, lens analysis, etc.

Pipeline: load Qwen3-8B with `LanguageModel`, run one prefill inside
`model.trace()`, save α and v with `.save()`, compute φ offline. Math is
identical to the production detector; only the *capture* is nnsight-flavoured.

This is a *teaching* artifact: one prefill per trial, eager attention,
no KV cache reuse, no thinking-token stripping, no multi-position
aggregation. For real detection runs use the production detector.

Structure: a small **Library** of self-contained functions (each takes its
dependencies as arguments — no module-level globals), then a thin **Demo**
that loads a model and calls them. Each library function is intended to be
copy-pasteable in isolation.

Open in Jupyter / VS Code: percent-format cells (``# %%``) render as a notebook.
Run as a script: ``python -m locos.notebooks.logit_contrib_nnsight``.
"""

# %% [markdown]
# # LOCOS — logit-contribution scoring (minimal reference)
#
# **Formula.** For target position $t$ and source position $j$, layer $l$ head $h$:
# $$\varphi^{(l,h)}_{t,j} = \alpha^{(l,h)}_{t,j} \cdot u_{y_t}^\top W_O^{(l,h)} v^{(l,h)}_j$$
# where $\alpha$ is the attention probability, $v_j$ the per-head value vector,
# $W_O^{(l,h)}$ the per-head output-projection slice (shape `[hidden, head_dim]`),
# and $u_{y_t}$ the unembedding row of the answer token.
#
# **Spatial contrast.** Sum φ over needle positions vs. non-needle positions:
# $S^{(l,h)} = \mathrm{mean}_{j \in \mathrm{needle}} \varphi - \mathrm{mean}_{j \notin \mathrm{needle}} \varphi$.
# Heads with high S route information *from the needle* into the answer logit.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from nnsight import LanguageModel
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from locos.utils.datasets import build_nolima_dataset, stratified_sample
from locos.utils.needle_utils import (
    build_period_token_positions,
    build_tracked_prompt,
    get_period_tokens,
)
from locos_eval.utils.plotting import save_figure, setup_plot_style

# %% [markdown]
# ## Library
#
# Six small functions. Each takes its dependencies as arguments and returns
# plain tensors / dicts — no module-level state, no closures over globals.
# Copy any of them into your own code; only the imports above are required.

# %% [markdown]
# ### Model introspection
# Helpers that work on either an `nnsight.LanguageModel` or a plain HF
# `AutoModelForCausalLM` — for nnsight we unwrap via `._model` to read the
# real config / layer list (the proxied tree is for tracing, not introspection).
# Handles both plain CausalLM and VLM-wrapped models
# (`model.language_model.model.layers`, e.g. Gemma3ForConditionalGeneration).


# %%
def _unwrap(model):
    """Return the underlying HF model (handles ``nnsight.LanguageModel`` wrapper)."""
    return getattr(model, "_model", model)


def get_model_dims(model) -> dict:
    """Return ``{L, H, H_KV, D, GQA}``. Accepts nnsight LanguageModel or HF causal LM."""
    cfg = _unwrap(model).config
    tcfg = getattr(cfg, "text_config", cfg)
    L = tcfg.num_hidden_layers
    H = tcfg.num_attention_heads
    H_KV = getattr(tcfg, "num_key_value_heads", H)
    D = getattr(tcfg, "head_dim", tcfg.hidden_size // H)
    return {"L": L, "H": H, "H_KV": H_KV, "D": D, "GQA": H // H_KV}


def get_decoder_layers(model):
    """Return the list of decoder layers as real ``nn.Module`` objects."""
    base = _unwrap(model)
    base = getattr(base, "language_model", base)
    base = getattr(base, "model", base)
    return base.layers


# %% [markdown]
# ### Forward-pass capture (nnsight)
# One prefill inside `model.trace()` with `output_attentions=True`. We `.save()`
# two things per layer: the attention probabilities (sliced at the answer's
# query position to keep memory bounded) and the `v_proj` output (raw — we'll
# reshape after). Standard nnsight pattern; you'd write the same shape of
# code for any per-layer activation capture.
#
# Requires `attn_implementation='eager'` (sdpa/flash discard the
# probabilities) — set at load time, see Demo cell below.


# %%
def capture_alpha_v(nn_model, input_ids: torch.Tensor, target_pos: int, dims: dict):
    """One nnsight trace → ``(alpha_t [L, H, S], v_all [L, S, H_KV, D])``.

    Implementation notes:

    1. **Per-layer attention capture.** We grab attention probabilities from
       each layer's ``self_attn.output[1]`` (the second element of the
       ``(attn_output, attn_weights)`` tuple HF emits when
       ``output_attentions=True``). The alternative — ``nn_model.output.attentions`` —
       is unreliable in nnsight 0.4: the model's top-level output assembles
       ``attentions`` from ``config.output_attentions``, not the forward
       kwarg, so it can silently come back as ``None``.

    2. **Belt-and-suspenders: set the config flag.** Even with per-layer
       capture, the per-layer attention module's forward checks
       ``output_attentions``. Whether the trace kwarg actually propagates to
       per-layer ``forward()`` calls varies across nnsight versions, so we
       set ``config.output_attentions = True`` directly. Idempotent.

    3. **Save order.** nnsight enforces ``.save()`` ordering to match
       forward execution: ``v_proj.output`` fires during attention compute,
       ``self_attn.output[1]`` completes after. Across layers: 0..L-1.

    4. **Slice after the trace, not inside.** Slicing ``[1][0, :, target_pos, :]``
       on a tuple-element proxy inside the trace was failing silently
       (nnsight's ``__exit__`` swallowed the exception). Saving the whole
       attention tensor and slicing on the materialised tensor is safer.
       Memory cost: ~230 MB per layer bf16 at S=1880 (≈8 GB across 36
       layers, freed after stacking the slice).

    Requires ``attn_implementation='eager'`` (set at load time).
    """
    L, H_KV, D = dims["L"], dims["H_KV"], dims["D"]
    hf = _unwrap(nn_model)
    hf.config.output_attentions = True

    # Hoisted out of the `with` block so the locals are always bound, even if
    # the trace body raises and nnsight's __exit__ swallows the exception.
    v_saved: list = []
    alpha_saved: list = []

    with nn_model.trace(input_ids, output_attentions=True):
        for layer_idx in range(L):
            attn = nn_model.model.layers[layer_idx].self_attn
            v_saved.append(attn.v_proj.output.save())
            alpha_saved.append(attn.output[1].save())

    if not alpha_saved or alpha_saved[0] is None:
        raise RuntimeError(
            "nnsight trace did not populate per-layer attention captures. "
            "Likely cause: the model's attention forward didn't return a "
            "(attn_output, attn_weights) tuple. Verify attn_implementation='eager' "
            "and that the model truly supports output_attentions."
        )

    alpha_t = torch.stack([a[0, :, target_pos, :].detach() for a in alpha_saved], dim=0)  # [L, H, S]
    v_all = torch.stack([v.view(1, -1, H_KV, D)[0].detach() for v in v_saved], dim=0)  # [L, S, H_kv, D]
    return alpha_t, v_all


# %% [markdown]
# ### φ computation
# The formula. Precompute `u_y^⊤ W_O^{(l,h)}` (a `head_dim`-vector per head),
# then dot with `v_j^{(l,h)}` for each source position, scale by α.
# Dtype follows `alpha_t` (bf16 throughout matches the production detector;
# upcast happens later in `spatial_contrast`).


# %%
def compute_phi(alpha_t, v_all, model, layers, y_t: int, dims: dict):
    """φ_{t,j}^{(l,h)} = α_{t,j} · u_{y_t}^⊤ W_O^{(l,h)} v_j^{(l,h)}.  Returns ``[L, H, S]``.

    ``model`` may be an nnsight LanguageModel or a plain HF causal LM —
    we read static weights from the unwrapped HF model in either case.
    """
    L, H, D, GQA = dims["L"], dims["H"], dims["D"], dims["GQA"]
    seq = alpha_t.shape[-1]
    device, dtype = alpha_t.device, alpha_t.dtype
    hf = _unwrap(model)

    u_y = hf.lm_head.weight[y_t].detach().to(device=device, dtype=dtype)  # [hidden]
    contrib = torch.empty(L, H, seq, dtype=dtype, device=device)
    for layer_idx in range(L):
        W_O = layers[layer_idx].self_attn.o_proj.weight.detach().to(device=device, dtype=dtype)
        # o_proj weight: [hidden, H*D]; the H*D dim is ordered by query-head index.
        assert W_O.shape[1] == H * D, f"o_proj cols={W_O.shape[1]} != H*D={H * D}"
        W_O_h = W_O.view(W_O.shape[0], H, D)  # [hidden, H, D]
        u_WO = torch.einsum("e,ehd->hd", u_y, W_O_h)  # [H, D]
        v_l = v_all[layer_idx].repeat_interleave(GQA, dim=1)  # [S, H_kv, D] → [S, H, D]
        contrib[layer_idx] = torch.einsum("hd,jhd->hj", u_WO, v_l)
    return alpha_t * contrib


# %% [markdown]
# ### Spatial contrast → S
# Per-head score: mean φ over needle positions minus mean φ over off-needle.
# Upcasts to fp32 (the result feeds matplotlib/numpy; tensor is small).


# %%
def spatial_contrast(phi: torch.Tensor, needle_start: int, needle_end: int) -> torch.Tensor:
    """Returns ``[L, H]`` fp32 score. Positive = head writes from needle to answer logit."""
    seq = phi.shape[-1]
    mask = torch.zeros(seq, dtype=torch.bool, device=phi.device)
    mask[needle_start:needle_end] = True
    p = phi.to(torch.float32)
    return p[..., mask].mean(-1) - p[..., ~mask].mean(-1)


# %% [markdown]
# ### Prompt construction
# Wraps `build_tracked_prompt` so the caller doesn't have to manage period
# tokens, BPE-aware needle insertion, or chat-template wrapping. Returns
# everything needed to score: input_ids, target_pos (the query position whose
# logit predicts the answer's first token), y_t (that token's id), and the
# needle span.


# %%
def build_input_for_trial(trial, tokenizer, model_name: str, device) -> dict:
    """Tokenize a ``RetrievalTrial`` → ``{input_ids, target_pos, y_t, needle_start, needle_end}``."""
    h_ids = tokenizer(trial.haystack_text, add_special_tokens=False).input_ids
    n_ids = tokenizer(trial.needle_text, add_special_tokens=False).input_ids
    a_ids = tokenizer(" " + trial.answer_text, add_special_tokens=False).input_ids
    period_pos = build_period_token_positions(h_ids, get_period_tokens(model_name, tokenizer))

    ids, n_s, n_e, a_s, _ = build_tracked_prompt(
        tokenizer=tokenizer,
        haystack_tokens=h_ids,
        needle_tokens=n_ids,
        period_positions=period_pos,
        prompt_template=trial.prompt_template,
        question_tokens=None,
        answer_tokens=a_ids,
        context_length=trial.context_length,
        depth_percent=trial.depth_percent,
        use_chat_template=True,
        disable_thinking=True,
    )
    ids = ids.to(device)
    # input_ids[0, a_s] is the first answer token; the logit predicting it
    # lives at query position a_s - 1 (autoregressive offset).
    return {
        "input_ids": ids,
        "target_pos": a_s - 1,
        "y_t": ids[0, a_s].item(),
        "needle_start": n_s,
        "needle_end": n_e,
    }


# %% [markdown]
# ### End-to-end pipeline
# Composes the four steps above. **LOCOS** = LOgit-COntribution Scoring; this
# is the public entry point users would import.


# %%
def locos_score(trial, nn_model, tokenizer, layers, dims: dict, model_name: str, device) -> dict:
    """LOCOS: score one trial. ``trial → {S, phi, input_ids, target_pos, y_t, needle_start, needle_end}``."""
    p = build_input_for_trial(trial, tokenizer, model_name, device)
    alpha_t, v_all = capture_alpha_v(nn_model, p["input_ids"], p["target_pos"], dims)
    phi = compute_phi(alpha_t, v_all, nn_model, layers, p["y_t"], dims)
    s = spatial_contrast(phi, p["needle_start"], p["needle_end"])
    return {"S": s, "phi": phi, **p}


# %% [markdown]
# ## Demo
# Everything below uses only the library above.

# %%
console = Console()


def _render_top_table(top_lh, S_arr, title, *, S_per_trial=None, S_std_arr=None):
    """Render a top-k heads table. If S_per_trial is given, include per-trial breakdown."""
    table = Table(title=title, header_style="bold", title_style="bold cyan")
    table.add_column("Rank", style="dim", justify="right", width=4)
    table.add_column("Layer.Head", style="bold")
    table.add_column("S" if S_per_trial is None else "mean S", justify="right")
    if S_per_trial is not None:
        table.add_column("± std", justify="right", style="dim")
        table.add_column("per-trial S", style="dim")
    for rank, (layer_idx, head_idx) in enumerate(top_lh, start=1):
        s_val = float(S_arr[layer_idx, head_idx])
        s_style = "green" if s_val > 0 else "red"
        row = [str(rank), f"L{layer_idx}.H{head_idx}", f"[{s_style}]{s_val:+.4f}[/{s_style}]"]
        if S_per_trial is not None:
            row.append(f"{S_std_arr[layer_idx, head_idx]:.4f}")
            row.append("  ".join(f"{x:+.3f}" for x in S_per_trial[:, layer_idx, head_idx]))
        table.add_row(*row)
    return table


# %% [markdown]
# ### Load model

# %%
MODEL_NAME = "Qwen/Qwen3-8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOLIMA_DIR = _REPO_ROOT / "data" / "nolima"

console.rule("[bold cyan]LOCOS[/bold cyan] — logit-contribution scoring", style="cyan")

with console.status(f"[cyan]Loading {MODEL_NAME} via nnsight…[/cyan]", spinner="dots"):
    # nnsight wraps an HF AutoModelForCausalLM under the hood. We pass HF kwargs
    # (torch_dtype, attn_implementation, device_map) through; `dispatch=True`
    # loads the weights immediately rather than on first trace, so subsequent
    # introspection (config, layer count) returns real values.
    nn_model = LanguageModel(
        MODEL_NAME,
        device_map=DEVICE,
        dispatch=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    tokenizer = nn_model.tokenizer
    dims = get_model_dims(nn_model)
    layers = get_decoder_layers(nn_model)

H = dims["H"]
console.print(
    Panel.fit(
        f"[bold]{MODEL_NAME}[/bold]\n"
        f"L=[cyan]{dims['L']}[/cyan]  H=[cyan]{H}[/cyan]  "
        f"H_kv=[cyan]{dims['H_KV']}[/cyan]  head_dim=[cyan]{dims['D']}[/cyan]  "
        f"GQA=[cyan]{dims['GQA']}[/cyan]\n"
        f"device=[cyan]{DEVICE}[/cyan]  dtype=[cyan]bfloat16[/cyan]  "
        f"attn=[cyan]eager[/cyan]",
        title="Model loaded",
        border_style="green",
    )
)

# %% [markdown]
# ### Single trial

# %%
console.rule("[bold]Single trial[/bold]", style="dim")

trials = build_nolima_dataset(
    nolima_dir=NOLIMA_DIR,
    context_lengths=[1000, 2000, 3000, 4000, 5000],
    depth_percents=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
    question_type="onehop",
    max_characters_per_entry=3,
)
with console.status("[cyan]Scoring trial 1…[/cyan]", spinner="dots"):
    result = locos_score(trials[0], nn_model, tokenizer, layers, dims, MODEL_NAME, DEVICE)

console.print(
    Panel.fit(
        f"[bold]trial_id:[/bold]   {trials[0].trial_id}\n"
        f"[bold]answer:[/bold]     {trials[0].answer_text!r}\n"
        f"[bold]needle:[/bold]     [[cyan]{result['needle_start']}[/cyan], "
        f"[cyan]{result['needle_end']}[/cyan])  "
        f"(target_pos=[cyan]{result['target_pos']}[/cyan])\n"
        f"[bold]phi:[/bold]        {tuple(result['phi'].shape)}  [dim]{result['phi'].dtype}[/dim]",
        title="Trial 1",
        border_style="cyan",
    )
)

S_np = result["S"].cpu().numpy()
top_k = 10
flat_idx = np.argsort(S_np.ravel())[::-1][:top_k]
top_lh = [(int(i // H), int(i % H)) for i in flat_idx]
console.print(_render_top_table(top_lh, S_np, f"Top-{top_k} heads (single trial — noisy)"))

# %% [markdown]
# ### Single-trial heatmap

# %%
setup_plot_style()
fig, ax = plt.subplots(figsize=(7, 6))
vmax = float(np.abs(S_np).max())
im = ax.imshow(S_np, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
for layer_idx, head_idx in top_lh:
    ax.scatter(head_idx, layer_idx, s=40, facecolors="none", edgecolors="black", linewidths=1.2)
ax.set_xlabel("Head index")
ax.set_ylabel("Layer index")
fig.colorbar(im, ax=ax).set_label(
    r"$S = \mathrm{mean}\,\varphi_{\mathrm{needle}} - \mathrm{mean}\,\varphi_{\mathrm{off}}$"
)
_out_single = _REPO_ROOT / "figures" / "nb_logit_contrib_qwen3_8b.svg"
save_figure(fig, _out_single)
console.print(f"[dim]saved[/dim] [green]{_out_single.name}[/green]")

# %% [markdown]
# ### Multi-trial average
#
# Single-trial S is noisy: the score depends on the specific needle, question,
# depth, and context length. We sweep multiple `(context_length, depth)` configs
# and stratified-sample N trials from across them, then average S per (layer, head).
#
# **What averaging cleans up:** noise in the (layer, head) → S map. The
# *position* axis is NOT alignable across trials (different needle positions,
# sequence lengths, and needle widths), so the per-position φ plot at the
# end stays a single-trial diagnostic.

# %%
N_TRIALS = 50

console.rule(f"[bold]Multi-trial average (N={N_TRIALS})[/bold]", style="dim")

# Build a richer trial pool across context lengths and depths, then
# stratified-sample N trials evenly from it. The single (entry, test, char,
# ctx, depth) tuple is the unit of variation — different test_ids within
# the same entry already produce different needles (the needle template is
# filled with per-test ``input_args``), so we don't need entry-level dedup.
#
# Memory note: at ctx=3000, S≈3100 → full attention per layer ≈ 600 MB bf16,
# total ~22 GB across 36 layers; comfortable on H100/80GB alongside the
# 16 GB model. Bump ctx higher only if you have the headroom.
CONTEXT_LENGTHS = [1000, 2000, 3000, 4000, 5000]
DEPTH_PERCENTS = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

trial_pool = build_nolima_dataset(
    nolima_dir=NOLIMA_DIR,
    context_lengths=CONTEXT_LENGTHS,
    depth_percents=DEPTH_PERCENTS,
    question_type="onehop",
    max_characters_per_entry=1,
)
multi_trials = stratified_sample(trial_pool, N_TRIALS, seed=42)
console.print(
    f"[dim]Pool size:[/dim] {len(trial_pool)} trials across "
    f"{len(CONTEXT_LENGTHS)} ctx × {len(DEPTH_PERCENTS)} depth configs  "
    f"→ stratified-sampled [bold]{len(multi_trials)}[/bold]"
)

trial_table = Table(
    title=f"Selected {len(multi_trials)} trials",
    header_style="bold",
    title_style="bold cyan",
)
trial_table.add_column("#", style="dim", justify="right", width=3)
trial_table.add_column("trial_id", style="bold")
trial_table.add_column("ctx", justify="right", style="dim")
trial_table.add_column("depth", justify="right", style="dim")
trial_table.add_column("answer", style="green")
trial_table.add_column("needle (truncated)", style="dim")
for i, t in enumerate(multi_trials, start=1):
    trial_table.add_row(
        str(i),
        t.trial_id,
        str(t.context_length),
        f"{t.depth_percent:.0f}%",
        t.answer_text,
        t.needle_text[:50] + "…",
    )
console.print(trial_table)

# Score each, with a live progress bar. After each trial we update the bar's
# description to show the trial's top-1 head — useful for spotting variance.
S_per_trial = np.empty((len(multi_trials), dims["L"], H), dtype=np.float32)
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TextColumn("[dim]{task.fields[top]}[/dim]"),
    console=console,
) as progress:
    task = progress.add_task("[cyan]Scoring trials[/cyan]", total=len(multi_trials), top="")
    for i, t in enumerate(multi_trials):
        S_per_trial[i] = locos_score(t, nn_model, tokenizer, layers, dims, MODEL_NAME, DEVICE)["S"].cpu().numpy()
        top_idx = int(np.argmax(S_per_trial[i]))
        tl, th = top_idx // H, top_idx % H
        progress.update(
            task,
            advance=1,
            top=f"top L{tl}.H{th} S={S_per_trial[i, tl, th]:+.3f}",
        )

S_mean = S_per_trial.mean(axis=0)
S_std = S_per_trial.std(axis=0)

flat_avg = np.argsort(S_mean.ravel())[::-1][:top_k]
top_avg = [(int(i // H), int(i % H)) for i in flat_avg]
console.print(
    _render_top_table(
        top_avg,
        S_mean,
        f"Top-{top_k} heads averaged over {N_TRIALS} trials",
        S_per_trial=S_per_trial,
        S_std_arr=S_std,
    )
)

# Averaged heatmap.
fig_avg, ax_avg = plt.subplots(figsize=(7, 6))
vmax_avg = float(np.abs(S_mean).max())
im_avg = ax_avg.imshow(S_mean, aspect="auto", cmap="RdBu_r", vmin=-vmax_avg, vmax=vmax_avg, origin="lower")
for layer_idx, head_idx in top_avg:
    ax_avg.scatter(head_idx, layer_idx, s=40, facecolors="none", edgecolors="black", linewidths=1.2)
ax_avg.set_xlabel("Head index")
ax_avg.set_ylabel("Layer index")
fig_avg.colorbar(im_avg, ax=ax_avg).set_label(rf"mean $S$ over {N_TRIALS} trials")
_out_avg = _REPO_ROOT / "figures" / f"nb_logit_contrib_qwen3_8b_avg{N_TRIALS}.svg"
save_figure(fig_avg, _out_avg)
console.print(f"[dim]saved[/dim] [green]{_out_avg.name}[/green]")

# %% [markdown]
# ### Per-position φ for the top head (single trial only)
#
# A real retrieval head shows a sharp spike inside the needle window and
# ~zero elsewhere. Symlog y-axis because the spike can be 20–30× the
# off-needle bulk and a linear axis would flatten everything else to zero.
#
# Single-trial only: position axis is not alignable across trials.

# %%
top_l, top_h = top_lh[0]
phi_top = result["phi"][top_l, top_h].detach().to(torch.float32).cpu().numpy()
fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.plot(phi_top, linewidth=1.0, color="#0173B2")
ax2.axvspan(result["needle_start"], result["needle_end"], alpha=0.25, color="#DE8F05", label="needle")
ax2.axhline(0, color="black", linewidth=0.5, alpha=0.4)
ax2.set_yscale("symlog", linthresh=0.5)
ax2.set_xlabel("Source position $j$")
ax2.set_ylabel(r"$\varphi^{(l,h)}_{t,j}$")
ax2.legend(loc="upper right")
_out_phi = _REPO_ROOT / "figures" / "nb_logit_contrib_top_head_phi.svg"
save_figure(fig2, _out_phi)
console.print(f"[dim]saved[/dim] [green]{_out_phi.name}[/green]")
console.rule("[bold green]done[/bold green]", style="green")

# %% [markdown]
# ## What's missing vs. the production detector
#
# - **Few trials.** Real detection averages over thousands; small-N S is noisy
#   (e.g. heads whose population-mean S is negative can occasionally rank high
#   on N=5 just by luck — see L33.H31 in the comparison commentary).
# - **Single context length / depth.** Production sweeps both. Some retrieval
#   heads only fire at long contexts.
# - **No thinking-token handling.** Qwen3 emits `<think>...</think>`; the
#   production detector strips these and re-aligns the answer position.
# - **Single answer step.** Production scores at every answer step (e.g.
#   "Yuki Tanaka" → 2 steps), then averages. Notebook only scores the first.
# - **No ROUGE gating.** Production discards trials where the model didn't
#   actually generate the answer.
#
# The point of this notebook is the *formula*, not the protocol. Once you've
# convinced yourself φ does what it says on the tin, run the production
# detector for actual numbers.
