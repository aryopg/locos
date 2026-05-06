#!/usr/bin/env python3
"""Generate launch commands for the full experiment matrix.

Enumerates all combinations of (task, model, decoding config, seed) and prints
one ./deploy/launch_multi_ns.sh command per cell.  Retrieval heads are
downloaded from HuggingFace at job runtime via ensure_heads() in job_config.sh.

Design: Option C — all stochastic.  Every cell uses the model's recommended
sampling parameters (from deploy/model_sampling.yaml) with multiple seeds.
No deterministic (temperature=0) runs.

Usage::

    # Preview experiment counts
    python deploy/generate_experiments.py --dry-run

    # Generate greedy baselines only
    python deploy/generate_experiments.py --decodings greedy

    # Single model, single task
    python deploy/generate_experiments.py \\
        --models meta-llama/Meta-Llama-3-8B-Instruct --tasks nq_swap

    # Save to file for review, then execute
    python deploy/generate_experiments.py > /tmp/launch_all.sh
    bash /tmp/launch_all.sh
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Allow importing locos_eval.evals.experiment_key when running this script
# directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from locos_eval.evals.experiment_key import ExperimentKey  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADS_DIR = REPO_ROOT / "retrieval_heads"
SAMPLING_CONFIG = Path(__file__).resolve().parent / "model_sampling.yaml"
NAMESPACES_CONF = Path(__file__).resolve().parent / "namespaces.conf"


# ---------------------------------------------------------------------------
# SSH / cluster utilities (inlined from former deploy/monitor/ssh.py)
# ---------------------------------------------------------------------------

_SSH_CONTROL_DIR = Path(tempfile.gettempdir()) / "decore-ssh-controls"


@dataclass
class _Namespace:
    name: str
    ssh_host: str
    queue: str
    gpu_product: str
    secrets: str
    username: str = ""
    email: str = ""
    max_days: int = 0
    gpu_quota: int = 0


def _parse_namespaces(conf_path: Path) -> list[_Namespace]:
    """Parse namespaces.conf into a list of _Namespace objects."""
    ns_list = []
    for line in conf_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        ns_list.append(
            _Namespace(
                *parts[:5],
                username=parts[5] if len(parts) > 5 else "",
                email=parts[6] if len(parts) > 6 else "",
                max_days=int(parts[7]) if len(parts) > 7 and parts[7].strip().isdigit() else 0,
                gpu_quota=int(parts[8]) if len(parts) > 8 and parts[8].strip().isdigit() else 0,
            )
        )
    return ns_list


def _control_path(host: str) -> Path:
    return _SSH_CONTROL_DIR / host


def _is_connected(host: str) -> bool:
    """Check if an SSH ControlMaster socket is alive."""
    ctl = _control_path(host)
    result = subprocess.run(
        ["ssh", "-O", "check", "-S", str(ctl), host],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _shell_quote(s: str) -> str:
    """Quote a string for safe embedding in a bash -c '...' argument."""
    return "'" + s.replace("'", "'\\''") + "'"


def load_namespaces() -> list[str]:
    """Parse deploy/namespaces.conf and return the list of namespace names."""
    names: list[str] = []
    with open(NAMESPACES_CONF) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line.split("|", 1)[0].strip())
    assert names, f"No namespaces found in {NAMESPACES_CONF}"
    return names


ALL_MODELS = [
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "allenai/Olmo-3.1-32B-Instruct",
]

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASK_DEFS: dict[str, dict] = {
    "nq_swap": {
        "script": "deploy/jobs/eval_nq_swap.sh",
        "envs": {},
    },
    "aci_bench": {
        "script": "deploy/jobs/eval_aci_bench.sh",
        "envs": {},
    },
    "longbench_v2": {
        "script": "deploy/jobs/eval_longbench_v2.sh",
        "envs": {"LENGTH": "short"},
    },
    "musique": {
        "script": "deploy/jobs/eval_musique.sh",
        "envs": {"SUBSET": "answerable", "SPLIT": "validation"},
    },
    # "xsum" excluded from default matrix — add explicitly if needed
}

MEDRAG_SCRIPT = "deploy/jobs/eval_medrag.sh"
DEFAULT_MEDRAG_DATASETS = ["mmlu_med", "medqa", "supergpqa_med"]
DEFAULT_MEDRAG_TOPKS = [10]

BABILONG_SCRIPT = "deploy/jobs/eval_babilong.sh"
DEFAULT_BABILONG_SUBSETS = ["qa2", "qa3"]
DEFAULT_BABILONG_SPLIT = "0k"

# ---------------------------------------------------------------------------
# Decoding configurations
# ---------------------------------------------------------------------------

DECODING_CONFIGS: dict[str, dict] = {
    "greedy": {
        "decoding": "greedy",
        "needs_heads": False,
        "heads_suffix": None,
        "heads_label": None,
    },
    "ablation_wu_niah": {
        "decoding": "ablation",
        "needs_heads": True,
        "heads_suffix": "_wu_niah",
        "heads_label": "wu_niah",
    },
    "ablation_wu_nolima": {
        "decoding": "ablation",
        "needs_heads": True,
        "heads_suffix": "_wu_nolima",
        "heads_label": "wu_nolima",
    },
    "ablation_logitcontrib_nolima": {
        "decoding": "ablation",
        "needs_heads": True,
        "heads_suffix": "_logit_contrib_nolima",
        "heads_label": "logitcontrib_nolima",
    },
    "ablation_cri": {
        "decoding": "ablation",
        "needs_heads": True,
        "heads_suffix": "_cri_first_token_logit_diff",
        "heads_label": "cri_first_token_logit_diff",
    },
    "ablation_random": {
        "decoding": "ablation",
        "needs_heads": False,
        "heads_suffix": None,
        "heads_label": None,
        "random_heads": True,
        "num_heads": 50,
        # random_seed is tied to the per-cell sampling seed in generate_cells
        # / _build_experiment_key — each replicate ablates a different random
        # head subset so seed-averaging captures variance over the random-head
        # distribution, not just decoding noise on one fixed draw.
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def model_short_name(model: str) -> str:
    """Extract short name: 'meta-llama/Foo' → 'Foo'."""
    return model.split("/")[-1]


def heads_file(model: str, suffix: str) -> Path:
    """Return expected heads file path for a model + suffix."""
    return HEADS_DIR / f"{model_short_name(model)}{suffix}.json"


def load_sampling_config() -> dict:
    """Load per-model sampling config from YAML."""
    with open(SAMPLING_CONFIG) as f:
        return yaml.safe_load(f)


def get_model_sampling(config: dict, model: str) -> dict:
    """Get sampling params for a model, falling back to defaults."""
    defaults = config.get("defaults", {})
    overrides = config.get("models", {}).get(model, {})
    return {**defaults, **overrides}


def build_env_flags(envs: dict) -> str:
    """Convert dict to --env KEY=VALUE flags, skipping empty values."""
    parts = []
    for k, v in envs.items():
        if v is not None and str(v) != "":
            parts.append(f"--env {k}={v}")
    return " ".join(parts)


def _resolve_hf_token() -> str | None:
    """Return HF token from env, falling back to a .env file at the repo root."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import dotenv_values

            values = dotenv_values(env_path)
        except ImportError:
            values = {}
            with open(env_path) as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    values[k.strip()] = v.strip().strip("\"'")
        for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            if values.get(key):
                return values[key]
    return None


def _ssh_kubectl(host: str, namespace: str, kubectl_args: str, timeout: int = 20) -> str | tuple[None, str]:
    """Run kubectl on a remote cluster host via SSH.

    Prefers an existing SSH ControlMaster socket (set up by launch_multi_ns.sh --connect).
    Falls back to a direct BatchMode connection so the script is usable without a
    pre-established mux.
    Returns stdout string on success, (None, error_msg) tuple on any failure.

    Note: ``-n {namespace}`` is placed before ``kubectl_args`` so that shell
    pipelines/fallbacks inside ``kubectl_args`` do not accidentally receive
    the namespace flag as extra arguments.
    """
    cmd = f"kubectl -n {namespace} {kubectl_args}"
    quoted = f"bash -l -c {_shell_quote(cmd)}"

    ctl = _control_path(host)
    if ctl.exists():
        try:
            result = subprocess.run(
                ["ssh", "-S", str(ctl), host, quoted],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return result.stdout
            return None, result.stderr.strip() or f"exit {result.returncode}"
        except Exception as e:
            return None, str(e)

    # Fall back to direct SSH (needs key-based auth, no password prompt)
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, quoted],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout
        return None, result.stderr.strip() or f"exit {result.returncode}"
    except Exception as e:
        return None, str(e)


# Maps the SCRIPT_SLUG suffix in a Kubernetes job name to the base task key.
# None means the task must be derived from embedded env vars (DATASETS / LENGTH).
_EVAL_SCRIPT_TO_TASK: dict[str, str | None] = {
    "eval-nq-swap": "nq_swap",
    "eval-aci-bench": "aci_bench",
    "eval-medrag": None,
    "eval-longbench-v2": None,
    "eval-musique": None,
    "eval-babilong": None,
}


def fetch_running_keys(
    filter_namespaces: list[str] | None = None,
    workers: int = 4,
) -> set[str]:
    """Return ExperimentKey.key values for Running/Pending experiments in the cluster.

    Reads namespaces.conf, SSHes into each host, queries kubectl in parallel,
    and reconstructs ExperimentKey from the env vars embedded in each job's
    command script.  Namespaces that fail SSH are skipped with a warning.
    """
    import re

    ns_configs = _parse_namespaces(NAMESPACES_CONF)
    if filter_namespaces:
        ns_configs = [ns for ns in ns_configs if ns.name in filter_namespaces]

    # ControlMaster sockets are required — EIDF uses TOTP so BatchMode SSH
    # will always fail. Check live connections upfront and bail early with one
    # clear message rather than spamming per-namespace SSH errors.
    reachable = [ns for ns in ns_configs if _is_connected(ns.ssh_host)]
    if not reachable:
        print(
            "# WARNING: no SSH ControlMaster connections active — skipping cluster running-job check.\n"
            "#   Connect first:  ./deploy/launch_multi_ns.sh --connect\n"
            "#   Or disable:     --no-skip-running",
            file=sys.stderr,
        )
        return set()

    failed_ns: list[tuple[str, str]] = []  # (namespace, error_msg)

    def _query(ns):
        result = _ssh_kubectl(ns.ssh_host, ns.name, "get jobs -o json")
        if isinstance(result, tuple):  # (None, error_msg)
            failed_ns.append((ns.name, result[1]))
            return []
        try:
            return json.loads(result).get("items", [])
        except json.JSONDecodeError as e:
            failed_ns.append((ns.name, f"JSON parse error: {e}"))
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        ns_job_lists = list(ex.map(_query, reachable))

    if failed_ns:
        if len(failed_ns) == len(reachable):
            print(
                "# WARNING: kubectl query failed on all namespaces — "
                "ControlMaster sockets may have dropped.\n"
                "#   Reconnect with: ./deploy/launch_multi_ns.sh --connect",
                file=sys.stderr,
            )
        else:
            print("# WARNING: kubectl query failed on some namespaces:", file=sys.stderr)
        for ns_name, err in failed_ns:
            print(f"#   {ns_name}: {err}", file=sys.stderr)

    running_keys: set[str] = set()
    active_job_count = 0
    unmatched_slugs: list[str] = []

    for jobs in ns_job_lists:
        for job in jobs:
            job_name = job.get("metadata", {}).get("name", "")
            if not job_name.startswith("decore-"):
                continue

            # Skip terminal (Completed / Failed) jobs
            conditions = job.get("status", {}).get("conditions", [])
            if any(c.get("status") == "True" and c.get("type") in ("Complete", "Failed") for c in conditions):
                continue

            active_job_count += 1

            # The job command is a bash script with `export KEY=VALUE` lines
            container = job.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [{}])[0]
            cmd_parts = container.get("command", []) + container.get("args", [])
            if len(cmd_parts) >= 3 and cmd_parts[-2] == "-c":
                command_str = cmd_parts[-1]
            elif len(cmd_parts) == 1:
                command_str = cmd_parts[0]
            else:
                command_str = " ".join(cmd_parts)

            env_vars: dict[str, str] = {}
            for m in re.finditer(r"^export\s+(\w+)=(.+)$", command_str, re.MULTILINE):
                env_vars[m.group(1)] = m.group(2).strip().strip("'\"")

            model = env_vars.get("MODEL")
            if not model:
                continue

            decoding = env_vars.get("DECODING", "greedy")
            heads = env_vars.get("HEADS", "")
            heads_label = env_vars.get("HEADS_LABEL") or None
            sampling_seed_str = env_vars.get("SAMPLING_SEED", "")
            sampling_seed = int(sampling_seed_str) if sampling_seed_str.isdigit() else None
            num_heads = int(env_vars.get("NUM_HEADS", "50") or "50")
            # RANDOM_SEED is tied to SAMPLING_SEED for ablation_random cells
            # generated post-tying. Fall back to the env var (or default 42)
            # for any in-flight pre-tying jobs so their keys still match.
            random_seed_str = env_vars.get("RANDOM_SEED", "")
            random_seed = int(random_seed_str) if random_seed_str.isdigit() else 42

            # Derive task key(s) from the script-slug suffix of the job name
            task_keys: list[str] = []
            for script_slug, default_task in _EVAL_SCRIPT_TO_TASK.items():
                if f"-{script_slug}" in job_name:
                    if default_task is not None:
                        task_keys = [default_task]
                    elif script_slug == "eval-medrag":
                        top_k = env_vars.get("TOP_K", "10")
                        task_keys = [f"medrag_{ds}_top{top_k}" for ds in env_vars.get("DATASETS", "").split() if ds]
                    elif script_slug == "eval-longbench-v2":
                        length = env_vars.get("LENGTH", "all")
                        task_keys = [f"longbench_v2_{length}" if length and length != "all" else "longbench_v2"]
                    elif script_slug == "eval-musique":
                        subset = env_vars.get("SUBSET", "answerable")
                        task_keys = [f"musique_{subset}"]
                    elif script_slug == "eval-babilong":
                        split = env_vars.get("SPLIT", "0k")
                        subsets = env_vars.get("SUBSETS", "qa2 qa3").split()
                        task_keys = [f"babilong_{s}_{split}" for s in subsets if s]
                    break

            if not task_keys:
                unmatched_slugs.append(job_name)
                continue  # detection / non-eval job — not relevant

            for task_for_key in task_keys:
                if heads == "random":
                    ek = ExperimentKey(
                        task=task_for_key,
                        model=model,
                        decoding=decoding,
                        heads_path="random",
                        num_heads=num_heads,
                        random_seed=random_seed,
                        sampling_seed=sampling_seed,
                    )
                else:
                    ek = ExperimentKey(
                        task=task_for_key,
                        model=model,
                        decoding=decoding,
                        heads_path=heads if heads else None,
                        heads_label=heads_label,
                        sampling_seed=sampling_seed,
                    )
                running_keys.add(ek.key)

    print(
        f"# Found {len(running_keys)} running/pending experiments "
        f"({active_job_count} active jobs across {len(reachable)} namespaces)",
        file=sys.stderr,
    )
    if unmatched_slugs:
        print(f"#   {len(unmatched_slugs)} non-eval jobs skipped: {', '.join(unmatched_slugs)}", file=sys.stderr)
    return running_keys


def fetch_complete_keys(
    repo_id: str,
    hf_prefix: str = "downstream_results",
    repo_type: str = "dataset",
    workers: int = 16,
) -> set[str]:
    """Return the set of ``ExperimentKey.key`` values that are complete on HF.

    Strategy: one ``list_repo_files`` call to enumerate all ``manifest.json``
    files under ``{hf_prefix}/``, then parallel ``hf_hub_download`` for each
    manifest to confirm ``status == "complete"``.
    """
    from huggingface_hub import HfApi, hf_hub_download

    token = _resolve_hf_token()
    if token:
        # Older huggingface_hub versions don't accept `token=` on every call;
        # set the env var so all subsequent HF calls (including in worker
        # threads) pick it up uniformly.
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    except Exception as exc:
        print(f"WARNING: failed to list HF repo {repo_id!r}: {exc}", file=sys.stderr)
        print("WARNING: proceeding without skip-done filter", file=sys.stderr)
        return set()

    prefix = f"{hf_prefix}/"
    suffix = "/manifest.json"
    manifest_paths = [f for f in files if f.startswith(prefix) and f.endswith(suffix)]

    first_error: list[str] = []

    def _check(path: str) -> str | None:
        key = path[len(prefix) : -len(suffix)]
        try:
            local = hf_hub_download(repo_id=repo_id, filename=path, repo_type=repo_type)
            with open(local) as f:
                data = json.load(f)
        except Exception as exc:
            if not first_error:
                first_error.append(f"{path}: {type(exc).__name__}: {exc}")
            return None
        if data.get("status") != "complete":
            if not first_error:
                first_error.append(f"{path}: status={data.get('status')!r}")
            return None
        return key

    complete: set[str] = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for result in ex.map(_check, manifest_paths):
            if result:
                complete.add(result)
    print(
        f"# Found {len(complete)} complete experiments on {repo_id} (of {len(manifest_paths)} manifests)",
        file=sys.stderr,
    )
    if first_error and len(complete) < len(manifest_paths):
        print(f"# First skipped manifest: {first_error[0]}", file=sys.stderr)
    return complete


def build_launch_cmd(model: str, script: str, envs: dict, namespace: str | None = None) -> str:
    """Build a single launch_multi_ns.sh command.

    If ``namespace`` is given, pins this cell via ``--namespace`` so parallel
    launches (xargs -P N) don't race on the launcher's in-process RR state.
    """
    env_str = build_env_flags(envs)
    ns_flag = f" --namespace {namespace}" if namespace else ""
    return f'MODELS="{model}" ./deploy/launch_multi_ns.sh {script} {env_str}{ns_flag}'


# ---------------------------------------------------------------------------
# Matrix generation
# ---------------------------------------------------------------------------


def _build_experiment_key(
    *,
    task_for_key: str,
    model: str,
    dec_config: dict,
    heads_rel_path: str | None,
    sampling_seed: int | None,
) -> ExperimentKey:
    """Build the canonical ExperimentKey for a cell.

    For ``ablation_random`` cells the head-selection seed is tied to the
    sampling seed so each replicate draws a different random head subset.
    Sampling seed of ``None`` falls back to ``42`` for the head-selection
    seed (matches ExperimentKey's default).
    """
    decoding = dec_config["decoding"]
    if dec_config.get("random_heads"):
        return ExperimentKey(
            task=task_for_key,
            model=model,
            decoding=decoding,
            heads_path="random",
            num_heads=dec_config["num_heads"],
            random_seed=sampling_seed if sampling_seed is not None else 42,
            sampling_seed=sampling_seed,
        )
    return ExperimentKey(
        task=task_for_key,
        model=model,
        decoding=decoding,
        heads_path=heads_rel_path,
        heads_label=dec_config.get("heads_label"),
        sampling_seed=sampling_seed,
    )


def generate_cells(
    tasks: list[str],
    models: list[str],
    decodings: list[str],
    medrag_datasets: list[str],
    medrag_topks: list[int],
    seeds: list[int],
    babilong_subsets: list[str] | None = None,
    babilong_split: str = DEFAULT_BABILONG_SPLIT,
    namespaces: list[str] | None = None,
    complete_keys: set[str] | None = None,
    running_keys: set[str] | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Generate all experiment cells.

    Option C: every cell gets the model's sampling params + a seed.

    Cells whose ``ExperimentKey.key`` appears in ``complete_keys`` are returned
    in the third tuple element (already_done).  Cells in ``running_keys`` are
    returned in the fourth element (in_progress).  Both are excluded from the
    primary cells list so they won't be re-launched.
    """
    sampling_config = load_sampling_config()
    cells: list[dict] = []
    skipped: list[dict] = []
    already_done: list[dict] = []
    in_progress: list[dict] = []
    complete_keys = complete_keys or set()
    running_keys = running_keys or set()
    babilong_subsets = babilong_subsets if babilong_subsets is not None else list(DEFAULT_BABILONG_SUBSETS)

    # Round-robin per-cell namespace assignment. Pinning each cell avoids
    # race conditions when launches run in parallel (xargs -P N), since
    # launch_multi_ns.sh's in-process RR counter resets on every invocation.
    ns_cycle: list[str | None] = list(namespaces) if namespaces else [None]
    ns_iter_idx = 0

    # Seed is outermost so all configs for seed N complete before seed N+1 starts.
    for seed in seeds:
        for model in models:
            sp = get_model_sampling(sampling_config, model)

            for dec_name in decodings:
                dc = DECODING_CONFIGS[dec_name]

                # Heads are stored on HF (aryopg/decore-results) and downloaded
                # at job runtime by ensure_heads() in deploy/job_config.sh. No
                # local existence check here — retrieval_heads/ is not in-repo.

                # Build base env vars
                base_envs: dict[str, str] = {
                    "DECODING": dc["decoding"],
                    "TEMPERATURE": str(sp.get("temperature", 0.6)),
                    "TOP_P": str(sp.get("top_p", 0.9)),
                    "TOP_K_SAMPLING": str(sp.get("top_k", -1)),
                }

                if dc["needs_heads"]:
                    # Use relative path (runs inside pod after git clone)
                    hf = heads_file(model, dc["heads_suffix"])
                    base_envs["HEADS"] = str(hf.relative_to(REPO_ROOT))
                    if dc["heads_label"]:
                        base_envs["HEADS_LABEL"] = dc["heads_label"]
                    base_envs["NUM_HEADS"] = str(dc.get("num_heads", 50))

                if dc.get("random_heads"):
                    base_envs["HEADS"] = "random"
                    base_envs["NUM_HEADS"] = str(dc["num_heads"])
                    # Tie random-head selection to the per-cell sampling seed
                    # so each replicate samples a different head subset (see
                    # _build_experiment_key for the matching ExperimentKey).
                    base_envs["RANDOM_SEED"] = str(seed)

                heads_rel_path = base_envs.get("HEADS")  # None / "random" / "retrieval_heads/..."

                seed_envs = {**base_envs, "SAMPLING_SEED": str(seed)}

                # Expand across tasks
                for task_key in tasks:
                    if task_key == "medrag":
                        for ds in medrag_datasets:
                            for topk in medrag_topks:
                                task_envs = {
                                    **seed_envs,
                                    "DATASETS": ds,
                                    "TOP_K": str(topk),
                                }
                                task_for_key = f"medrag_{ds}_top{topk}"
                                ek = _build_experiment_key(
                                    task_for_key=task_for_key,
                                    model=model,
                                    dec_config=dc,
                                    heads_rel_path=heads_rel_path,
                                    sampling_seed=seed,
                                )
                                cell = {
                                    "model": model,
                                    "task": task_for_key,
                                    "decoding": dec_name,
                                    "seed": seed,
                                    "key": ek.key,
                                }
                                if ek.key in complete_keys:
                                    already_done.append(cell)
                                    continue
                                if ek.key in running_keys:
                                    in_progress.append(cell)
                                    continue
                                ns = ns_cycle[ns_iter_idx % len(ns_cycle)]
                                ns_iter_idx += 1
                                cell["namespace"] = ns
                                cell["cmd"] = build_launch_cmd(model, MEDRAG_SCRIPT, task_envs, ns)
                                cells.append(cell)
                    elif task_key == "babilong":
                        for subset in babilong_subsets:
                            task_envs = {
                                **seed_envs,
                                "SUBSETS": subset,
                                "SPLIT": babilong_split,
                            }
                            task_for_key = f"babilong_{subset}_{babilong_split}"
                            ek = _build_experiment_key(
                                task_for_key=task_for_key,
                                model=model,
                                dec_config=dc,
                                heads_rel_path=heads_rel_path,
                                sampling_seed=seed,
                            )
                            cell = {
                                "model": model,
                                "task": task_for_key,
                                "decoding": dec_name,
                                "seed": seed,
                                "key": ek.key,
                            }
                            if ek.key in complete_keys:
                                already_done.append(cell)
                                continue
                            if ek.key in running_keys:
                                in_progress.append(cell)
                                continue
                            ns = ns_cycle[ns_iter_idx % len(ns_cycle)]
                            ns_iter_idx += 1
                            cell["namespace"] = ns
                            cell["cmd"] = build_launch_cmd(model, BABILONG_SCRIPT, task_envs, ns)
                            cells.append(cell)
                    elif task_key in TASK_DEFS:
                        td = TASK_DEFS[task_key]
                        task_envs = {**seed_envs, **td["envs"]}
                        # longbench_v2 uses LENGTH-suffixed task name in ExperimentKey
                        task_for_key = task_key
                        if task_key == "longbench_v2":
                            length = td["envs"].get("LENGTH", "all")
                            if length and length != "all":
                                task_for_key = f"longbench_v2_{length}"
                        elif task_key == "musique":
                            subset = td["envs"].get("SUBSET", "answerable")
                            task_for_key = f"musique_{subset}"
                        ek = _build_experiment_key(
                            task_for_key=task_for_key,
                            model=model,
                            dec_config=dc,
                            heads_rel_path=heads_rel_path,
                            sampling_seed=seed,
                        )
                        cell = {
                            "model": model,
                            "task": task_key,
                            "decoding": dec_name,
                            "seed": seed,
                            "key": ek.key,
                        }
                        if ek.key in complete_keys:
                            already_done.append(cell)
                            continue
                        if ek.key in running_keys:
                            in_progress.append(cell)
                            continue
                        ns = ns_cycle[ns_iter_idx % len(ns_cycle)]
                        ns_iter_idx += 1
                        cell["namespace"] = ns
                        cell["cmd"] = build_launch_cmd(model, td["script"], task_envs, ns)
                        cells.append(cell)
                    else:
                        print(
                            f"WARNING: unknown task {task_key!r}, skipping",
                            file=sys.stderr,
                        )

    return cells, skipped, already_done, in_progress


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_dry_run(
    cells: list[dict],
    skipped: list[dict],
    already_done: list[dict] | None = None,
    in_progress: list[dict] | None = None,
) -> None:
    """Print experiment summary."""
    print(f"Total experiment cells (remaining): {len(cells)}")
    if already_done:
        print(f"Already complete on HF (filtered out): {len(already_done)}")
    if in_progress:
        print(f"Currently running/pending in cluster (filtered out): {len(in_progress)}")
    print()

    by_model = Counter(c["model"] for c in cells)
    print("By model:")
    for m in ALL_MODELS:
        if m in by_model:
            print(f"  {m}: {by_model[m]}")
    print()

    by_decoding = Counter(c["decoding"] for c in cells)
    print("By decoding:")
    for d, n in sorted(by_decoding.items()):
        print(f"  {d}: {n}")
    print()

    by_task = Counter(c["task"] for c in cells)
    print("By task:")
    for t, n in sorted(by_task.items()):
        print(f"  {t}: {n}")

    by_seed = Counter(c["seed"] for c in cells)
    print()
    print("By seed (remaining):")
    for s, n in sorted(by_seed.items()):
        print(f"  seed={s}: {n}")

    if in_progress:
        by_seed_running = Counter(c["seed"] for c in in_progress)
        print()
        print("By seed (running/pending in cluster):")
        for s, n in sorted(by_seed_running.items()):
            print(f"  seed={s}: {n}")

    by_ns = Counter(c.get("namespace") or "(round-robin)" for c in cells)
    print()
    print("By namespace:")
    for ns, n in sorted(by_ns.items()):
        print(f"  {ns}: {n}")

    if skipped:
        print(f"\nSkipped {len(skipped)} model x decoding combinations (missing heads):")
        for s in skipped:
            print(f"  {s['model']} / {s['decoding']}: {s['reason']}")


def print_commands(cells: list[dict], parallel: int = 1) -> None:
    """Print executable shell commands.

    If ``parallel > 1``, emit a script that pipes the launch lines through
    ``xargs -P`` for concurrent submission. Each cell is pre-assigned a
    namespace, so no launcher-side RR state is contended.
    """
    print("#!/usr/bin/env bash")
    print("# Auto-generated experiment launch commands")
    print(f"# {len(cells)} experiment cells (parallel={parallel})")
    print("# Generated by: python deploy/generate_experiments.py")
    print("set -uo pipefail")
    print()

    if parallel <= 1:
        for cell in cells:
            label = f"{cell['task']} | {model_short_name(cell['model'])} | {cell['decoding']} | s{cell['seed']}"
            print(f"# {label}")
            print(cell["cmd"])
            print()
        return

    # Parallel mode: pipe launch lines through xargs -P N.
    # Use ``tr '\n' '\0' | xargs -0`` for portability (BSD/macOS xargs has
    # no ``-d``). ``-n 1`` passes each NUL-delimited line as a single arg
    # to ``bash -c 'eval "$0"'`` — ``$0`` receives the whole line.
    # This avoids both the ``-I`` replacement-length limit and the GNU-only
    # ``-d`` flag.
    print(f"# Parallel launches via xargs -P {parallel}")
    print("# Each cell is pinned to a namespace in the generator, so xargs-level")
    print("# concurrency does not race the launcher's round-robin state.")
    print("(cat <<'LAUNCHES'")
    for cell in cells:
        # Each line MUST be a complete self-contained shell command.
        print(cell["cmd"])
    print("LAUNCHES")
    print(f") | tr '\\n' '\\0' | xargs -0 -P {parallel} -n 1 bash -c 'eval \"$0\"'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate experiment launch commands for the full matrix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["nq_swap", "aci_bench", "medrag", "longbench_v2", "musique", "babilong"],
        help="Tasks to include. Use 'medrag' for all MedRAG sub-datasets and "
        "'babilong' for all BABILong subsets. "
        "(default: nq_swap aci_bench medrag longbench_v2 musique babilong)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to include (default: all 10).",
    )
    parser.add_argument(
        "--decodings",
        nargs="+",
        default=None,
        help=f"Decoding configs. Choices: {list(DECODING_CONFIGS.keys())} " f"(default: all)",
    )
    parser.add_argument(
        "--medrag-datasets",
        nargs="+",
        default=DEFAULT_MEDRAG_DATASETS,
        help=f"MedRAG sub-datasets (default: {DEFAULT_MEDRAG_DATASETS})",
    )
    parser.add_argument(
        "--medrag-topk",
        nargs="+",
        type=int,
        default=DEFAULT_MEDRAG_TOPKS,
        help=f"MedRAG top-k values (default: {DEFAULT_MEDRAG_TOPKS})",
    )
    parser.add_argument(
        "--babilong-subsets",
        nargs="+",
        default=DEFAULT_BABILONG_SUBSETS,
        help=f"BABILong subsets (default: {DEFAULT_BABILONG_SUBSETS})",
    )
    parser.add_argument(
        "--babilong-split",
        default=DEFAULT_BABILONG_SPLIT,
        help=f"BABILong context split (default: {DEFAULT_BABILONG_SPLIT})",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Seeds for stochastic runs (default: 1 2 3)",
    )
    parser.add_argument(
        "--namespaces",
        nargs="+",
        default=None,
        help=(
            "Namespaces to round-robin cells across (default: all from "
            "deploy/namespaces.conf). Pass a single value to pin all cells."
        ),
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        default=True,
        help="Skip cells already complete on the HF results repo (default: on)",
    )
    parser.add_argument(
        "--no-skip-done",
        dest="skip_done",
        action="store_false",
        help="Disable HF completion check (emit launch commands for everything)",
    )
    parser.add_argument(
        "--skip-running",
        action="store_true",
        default=True,
        help="Skip cells currently running/pending in the cluster namespaces (default: on)",
    )
    parser.add_argument(
        "--no-skip-running",
        dest="skip_running",
        action="store_false",
        help="Disable cluster running-job check",
    )
    parser.add_argument(
        "--hf-repo",
        default="aryopg/locos_downstream_results",
        help="HF dataset repo to query for completed downstream eval experiments "
        "(default: aryopg/locos_downstream_results). Heads / detection / ablation "
        "analyses live on aryopg/decore-results — that split is enforced by the "
        "deploy job scripts via HF_DOWNSTREAM_REPO vs HF_RESULTS_REPO.",
    )
    parser.add_argument(
        "--hf-prefix",
        default="downstream_results",
        help="Prefix in the HF repo where eval results live (default: downstream_results)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="If >1, emit a script that runs launches concurrently via xargs -P N",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary counts instead of launch commands",
    )
    args = parser.parse_args()

    models = args.models or ALL_MODELS
    decodings = args.decodings or list(DECODING_CONFIGS.keys())

    for d in decodings:
        if d not in DECODING_CONFIGS:
            parser.error(f"Unknown decoding config: {d!r}. " f"Choices: {list(DECODING_CONFIGS.keys())}")

    namespaces = args.namespaces if args.namespaces is not None else load_namespaces()
    assert args.parallel >= 1, f"--parallel must be >= 1, got {args.parallel}"

    complete_keys: set[str] = set()
    if args.skip_done:
        complete_keys = fetch_complete_keys(repo_id=args.hf_repo, hf_prefix=args.hf_prefix)

    running_keys: set[str] = set()
    if args.skip_running:
        running_keys = fetch_running_keys()

    cells, skipped, already_done, in_progress = generate_cells(
        tasks=args.tasks,
        models=models,
        decodings=decodings,
        medrag_datasets=args.medrag_datasets,
        medrag_topks=args.medrag_topk,
        seeds=args.seeds,
        babilong_subsets=args.babilong_subsets,
        babilong_split=args.babilong_split,
        namespaces=namespaces,
        complete_keys=complete_keys,
        running_keys=running_keys,
    )

    if args.dry_run:
        print_dry_run(cells, skipped, already_done, in_progress)
    else:
        print_commands(cells, parallel=args.parallel)


if __name__ == "__main__":
    main()
