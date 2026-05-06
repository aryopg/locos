#!/usr/bin/env bash
# Launch jobs across multiple EIDF Kubernetes namespaces.
# Compatible with bash 3.2+ (macOS default) — no associative arrays.
#
# Usage:
#   ./deploy/launch_multi_ns.sh <script> [options]
#   ./deploy/launch_multi_ns.sh deploy/jobs/detect_logit_contrib.sh
#   ./deploy/launch_multi_ns.sh deploy/jobs/eval_nq_swap.sh --dry-run
#   ./deploy/launch_multi_ns.sh deploy/jobs/ablation_nolima.sh --env LIMIT=50
#   ./deploy/launch_multi_ns.sh deploy/jobs/eval_medrag.sh --namespace eidf106ns
#   ./deploy/launch_multi_ns.sh deploy/jobs/detect_cri.sh --env DATASET=niah --env NUM_EXAMPLES=100
#   HEADS_METHOD=wu_niah,wu_nolima,logit_contrib ./deploy/launch_multi_ns.sh deploy/jobs/eval_nq_swap.sh
#   ./deploy/launch_multi_ns.sh deploy/jobs/eval_nq_swap.sh --verbose  # show kblaunch output
#   ./deploy/launch_multi_ns.sh --connect             # pre-establish SSH connections only
#   ./deploy/launch_multi_ns.sh --disconnect           # tear down SSH control sockets
#
# Requires:
#   - SSH config entries for each cluster's login node (see ~/.ssh/config)
#   - kblaunch installed on each login node
#   - deploy/namespaces.conf configured with namespace -> SSH host mappings
#   - Job scripts in deploy/jobs/ (see deploy/jobs/README.md)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SSH_CONTROL_DIR="${TMPDIR:-/tmp}/decore-ssh-controls"

# Use project venv Python for local pre-checks (check_experiment.py needs locos_eval)
if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
    LOCAL_PYTHON="${REPO_DIR}/.venv/bin/python"
else
    LOCAL_PYTHON="python"
fi

# Source shared job configuration
source "${SCRIPT_DIR}/job_config.sh"

# ---------------------------------------------------------------------------
# Parse namespace config into parallel arrays
# ---------------------------------------------------------------------------

# Parallel arrays: index i gives ns name, ssh host, queue, gpu product, secrets, username, email, max_days, gpu_quota
NS_NAMES=()
NS_SSH_HOSTS=()
NS_QUEUES=()
NS_GPU_PRODUCTS=()
NS_SECRETS=()
NS_USERNAMES=()
NS_EMAILS=()
NS_MAX_DAYS=()
NS_GPU_QUOTAS=()

parse_namespaces() {
    local conf="${SCRIPT_DIR}/namespaces.conf"
    if [[ ! -f "$conf" ]]; then
        echo "Error: ${conf} not found" >&2
        exit 1
    fi

    while IFS='|' read -r ns host queue gpu secrets username email max_days gpu_quota; do
        ns=$(echo "$ns" | xargs)
        [[ -z "$ns" || "$ns" == \#* ]] && continue

        NS_NAMES+=("$ns")
        NS_SSH_HOSTS+=("$(echo "$host" | xargs)")
        NS_QUEUES+=("$(echo "$queue" | xargs)")
        NS_GPU_PRODUCTS+=("$(echo "$gpu" | xargs)")
        NS_SECRETS+=("$(echo "$secrets" | xargs)")
        NS_USERNAMES+=("$(echo "${username:-}" | xargs)")
        NS_EMAILS+=("$(echo "${email:-}" | xargs)")
        NS_MAX_DAYS+=("$(echo "${max_days:-0}" | xargs)")
        NS_GPU_QUOTAS+=("$(echo "${gpu_quota:-0}" | xargs)")
    done < <(cat "$conf"; echo)

    if [[ ${#NS_NAMES[@]} -eq 0 ]]; then
        echo "Error: no namespaces configured in ${conf}" >&2
        exit 1
    fi
}

# Look up a namespace's index in the parallel arrays.
ns_index() {
    local target="$1"
    local i
    for i in $(seq 0 $(( ${#NS_NAMES[@]} - 1 ))); do
        if [[ "${NS_NAMES[$i]}" == "$target" ]]; then
            echo "$i"
            return 0
        fi
    done
    return 1
}

# Look up namespace properties by name.
ns_ssh_host()    { local i; i=$(ns_index "$1") && echo "${NS_SSH_HOSTS[$i]}"; }
ns_gpu_product() { local i; i=$(ns_index "$1") && echo "${NS_GPU_PRODUCTS[$i]}"; }
ns_secrets()     { local i; i=$(ns_index "$1") && echo "${NS_SECRETS[$i]}"; }
ns_username()    { local i; i=$(ns_index "$1") && echo "${NS_USERNAMES[$i]}"; }
ns_email()       { local i; i=$(ns_index "$1") && echo "${NS_EMAILS[$i]}"; }
ns_gpu_quota()   { local i; i=$(ns_index "$1") && echo "${NS_GPU_QUOTAS[$i]}"; }

# ---------------------------------------------------------------------------
# SSH ControlMaster management
# ---------------------------------------------------------------------------

ssh_control_path() {
    echo "${SSH_CONTROL_DIR}/$1"
}

connect_host() {
    local host="$1"
    local ctl
    ctl=$(ssh_control_path "$host")

    if ssh -O check -S "$ctl" "$host" 2>/dev/null; then
        echo "  [ok] ${host}: already connected"
        return 0
    fi

    echo "  --> Connecting to ${host} (TOTP may be required)..."
    ssh -f -N -M \
        -S "$ctl" \
        -o ControlPersist=3600 \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        "$host"
    echo "  [ok] ${host}: connected"
}

connect_all() {
    mkdir -p "$SSH_CONTROL_DIR"
    echo "Establishing SSH connections..."

    local connected_hosts=""
    local i
    for i in $(seq 0 $(( ${#NS_NAMES[@]} - 1 ))); do
        local host="${NS_SSH_HOSTS[$i]}"
        # Skip if already connected this host
        if echo "$connected_hosts" | grep -qF "|${host}|"; then
            continue
        fi
        connected_hosts="${connected_hosts}|${host}|"
        connect_host "$host"
    done
    echo ""
}

disconnect_all() {
    echo "Closing SSH connections..."
    if [[ -d "$SSH_CONTROL_DIR" ]]; then
        for ctl in "${SSH_CONTROL_DIR}"/*; do
            [[ -e "$ctl" ]] || continue
            local host
            host=$(basename "$ctl")
            ssh -O exit -S "$ctl" "$host" 2>/dev/null && echo "  [ok] ${host}: disconnected" || true
        done
        rm -rf "$SSH_CONTROL_DIR"
    fi
    echo "Done."
}

# Run a command on a remote host via the ControlMaster socket.
remote_exec() {
    local host="$1"
    shift
    local ctl
    ctl=$(ssh_control_path "$host")

    if ! ssh -O check -S "$ctl" "$host" 2>/dev/null; then
        echo "Error: no active SSH connection to ${host}. Run with --connect first." >&2
        return 1
    fi

    # Use login shell so PATH from .bash_profile/.profile is available
    # (non-login SSH commands skip these, so tools like kblaunch aren't found)
    ssh -S "$ctl" "$host" "bash -l -c $(printf '%q' "$*")"
}

# ---------------------------------------------------------------------------
# Job distribution (pinning via newline-separated list, bash 3.2 compatible)
# ---------------------------------------------------------------------------

PINNED_MODELS=""   # newline-separated "model|namespace" pairs

parse_pins() {
    local pin_str="$1"
    local IFS=','
    local pair
    for pair in $pin_str; do
        local model ns
        model=$(echo "$pair" | cut -d'=' -f1 | xargs)
        ns=$(echo "$pair" | cut -d'=' -f2 | xargs)
        PINNED_MODELS="${PINNED_MODELS}${model}|${ns}"$'\n'
    done
}

# Per-namespace GPU allocation tracking (populated after ACTIVE_NS is set)
# NS_GPU_ALLOCATED[i] tracks GPUs assigned so far in this launch session.
NS_GPU_ALLOCATED=()

init_gpu_tracking() {
    NS_GPU_ALLOCATED=()
    local i
    for i in $(seq 0 $(( ${#ACTIVE_NS[@]} - 1 ))); do
        NS_GPU_ALLOCATED+=(0)
    done
}

# Assign a namespace for a job, respecting GPU quotas.
# Round-robins across ALL namespaces; limited namespaces are skipped when full.
# Priority: pinned > round-robin (all namespaces) > overflow.
assign_namespace() {
    local model="$1" gpu_count="$2"

    # Check if model is pinned
    local pinned_ns
    pinned_ns=$(echo "$PINNED_MODELS" | grep "^${model}|" | head -1 | cut -d'|' -f2)
    if [[ -n "$pinned_ns" ]]; then
        echo "$pinned_ns"
        return
    fi

    local ns_count=${#ACTIVE_NS[@]}
    RR_IDX=${RR_IDX:-0}

    # Try each namespace starting from the round-robin index
    local tried=0
    while [[ $tried -lt $ns_count ]]; do
        local ai=$(( (RR_IDX + tried) % ns_count ))
        local ns_name="${ACTIVE_NS[$ai]}"
        local quota
        quota=$(ns_gpu_quota "$ns_name")
        local allocated=${NS_GPU_ALLOCATED[$ai]}

        # Unlimited (quota=0): always accept; Limited: accept if room
        if [[ "$quota" -eq 0 ]] || [[ $((allocated + gpu_count)) -le $quota ]]; then
            NS_GPU_ALLOCATED[$ai]=$((allocated + gpu_count))
            RR_IDX=$(( (RR_IDX + tried + 1) % ns_count ))
            echo "$ns_name"
            return
        fi

        tried=$((tried + 1))
    done

    # All limited namespaces full, no unlimited available — overflow to first
    echo "WARNING: all namespaces at GPU quota; overflowing" >&2
    NS_GPU_ALLOCATED[0]=$(( ${NS_GPU_ALLOCATED[0]} + gpu_count ))
    echo "${ACTIVE_NS[0]}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DRY_RUN=false
CONNECT_ONLY=false
DISCONNECT=false
VERBOSE=false
TARGET_NS=""
PIN_STR=""
JOB_SCRIPT=""
EXTRA_ENVS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=true; shift ;;
        --connect)      CONNECT_ONLY=true; shift ;;
        --disconnect)   DISCONNECT=true; shift ;;
        --verbose|-v)   VERBOSE=true; shift ;;
        --namespace)    TARGET_NS="$2"; shift 2 ;;
        --pin)          PIN_STR="$2"; shift 2 ;;
        --env)          EXTRA_ENVS+=("$2"); shift 2 ;;
        -*)             echo "Unknown flag: $1" >&2; exit 1 ;;
        *)
            if [[ -z "$JOB_SCRIPT" ]]; then
                JOB_SCRIPT="$1"
            else
                echo "Error: unexpected argument '$1' (script already set to '${JOB_SCRIPT}')" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

# Parse namespaces config
parse_namespaces

# Handle --disconnect
if [[ "$DISCONNECT" == true ]]; then
    disconnect_all
    exit 0
fi

# Determine active namespaces
ACTIVE_NS=()
if [[ -n "$TARGET_NS" ]]; then
    if ! ns_index "$TARGET_NS" > /dev/null 2>&1; then
        echo "Error: namespace '${TARGET_NS}' not found in namespaces.conf" >&2
        exit 1
    fi
    ACTIVE_NS=("$TARGET_NS")
else
    ACTIVE_NS=("${NS_NAMES[@]}")
fi

# Parse pins
[[ -n "$PIN_STR" ]] && parse_pins "$PIN_STR"

# Initialize GPU allocation tracking
init_gpu_tracking
RR_IDX=0

# Establish SSH connections (unless dry-run)
if [[ "$DRY_RUN" != true ]]; then
    connect_all
fi

# Handle --connect only
if [[ "$CONNECT_ONLY" == true ]]; then
    echo "SSH connections established. Run jobs with:"
    echo "  ./deploy/launch_multi_ns.sh deploy/jobs/<script>.sh"
    exit 0
fi

# Validate job script
if [[ -z "$JOB_SCRIPT" ]]; then
    echo "Error: no job script specified" >&2
    echo "" >&2
    echo "Usage: ./deploy/launch_multi_ns.sh <script> [options]" >&2
    echo "  e.g.: ./deploy/launch_multi_ns.sh deploy/jobs/detect_logit_contrib.sh" >&2
    echo "" >&2
    echo "  --dry-run         Preview job distribution without launching" >&2
    echo "  --verbose, -v     Show kblaunch output (YAML spec, etc.)" >&2
    echo "  --namespace NS    Target a single namespace" >&2
    echo "  --pin M=NS,...    Pin models to specific namespaces" >&2
    echo "  --env KEY=VALUE   Pass extra environment variable (repeatable)" >&2
    echo "  --connect         Pre-establish SSH connections only" >&2
    echo "  --disconnect      Tear down SSH control sockets" >&2
    exit 1
fi

if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "Error: job script not found: ${JOB_SCRIPT}" >&2
    exit 1
fi

# Derive script slug for job naming
SCRIPT_SLUG=$(basename "$JOB_SCRIPT" .sh | tr '_' '-')

# Resolve model list
if [[ -n "${MODELS:-}" ]]; then
    IFS=' ' read -ra MODEL_LIST <<< "$MODELS"
else
    MODEL_LIST=("${DEFAULT_MODELS[@]}")
fi

# Resolve heads method list (comma-separated, default: wu_niah)
HEADS_METHOD="${HEADS_METHOD:-wu_niah}"
IFS=',' read -ra METHOD_LIST <<< "$HEADS_METHOD"

# ---------------------------------------------------------------------------
# Build and distribute jobs
# ---------------------------------------------------------------------------

# Per-namespace job counts (parallel array matching ACTIVE_NS)
NS_JOB_COUNTS=()
for ns in "${ACTIVE_NS[@]}"; do
    NS_JOB_COUNTS+=(0)
done

job_index=0
total_jobs=0

echo "============================================"
echo " DeCoRe Multi-Namespace Job Launcher"
echo "============================================"
echo ""
echo "Job script: ${JOB_SCRIPT}"
echo "Active namespaces: ${ACTIVE_NS[*]}"
echo "Models: ${#MODEL_LIST[@]}"
echo "Heads methods: ${METHOD_LIST[*]}"
if [[ ${#EXTRA_ENVS[@]} -gt 0 ]]; then
    echo "Extra env: ${EXTRA_ENVS[*]}"
fi
echo ""

for method in "${METHOD_LIST[@]}"; do
export HEADS_METHOD="$method"
mslug=$(method_job_slug "$method")

for model in "${MODEL_LIST[@]}"; do
    slug=$(job_slug "$model")
    heads=$(heads_path "$model")
    gpus=$(gpu_count_for_model "$model")

    # EXTRA_ENVS may override the default HEADS_METHOD-derived heads path.
    # Parse them before naming/skipping so generated downstream eval commands
    # don't all look like the default wu_niah ("wn") run.
    _decoding_mode="decore"
    _heads_label=""
    for _e in "${EXTRA_ENVS[@]+"${EXTRA_ENVS[@]}"}"; do
        case "$_e" in
            DECODING=*)    _decoding_mode="${_e#DECODING=}" ;;
            HEADS=*)       heads="${_e#HEADS=}" ;;
            HEADS_LABEL=*) _heads_label="${_e#HEADS_LABEL=}" ;;
        esac
    done
    if [[ "$_decoding_mode" == "greedy" ]]; then
        heads=""
        mslug=$(method_job_slug "greedy")
    elif [[ "$heads" == "random" ]]; then
        mslug=$(method_job_slug "random")
    elif [[ -n "$_heads_label" ]]; then
        mslug=$(method_job_slug "$_heads_label")
    else
        mslug=$(method_job_slug "$method")
    fi
    job_name="decore-${slug}-${mslug}-${SCRIPT_SLUG}"

    # Assign namespace respecting GPU quotas (inline to avoid subshell)
    ns=""

    # Check if model is pinned
    pinned_ns=$(echo "$PINNED_MODELS" | grep "^${model}|" | head -1 | cut -d'|' -f2 || true)
    if [[ -n "$pinned_ns" ]]; then
        ns="$pinned_ns"
    fi

    # Round-robin across all namespaces; skip limited ones that are full
    if [[ -z "$ns" ]]; then
        _ns_count=${#ACTIVE_NS[@]}
        _tried=0
        while [[ $_tried -lt $_ns_count ]]; do
            _ai=$(( (RR_IDX + _tried) % _ns_count ))
            _q=$(ns_gpu_quota "${ACTIVE_NS[$_ai]}")
            _alloc=${NS_GPU_ALLOCATED[$_ai]}

            # Unlimited (quota=0): always accept; Limited: accept if room
            if [[ "$_q" -eq 0 ]] || [[ $((_alloc + gpus)) -le $_q ]]; then
                ns="${ACTIVE_NS[$_ai]}"
                NS_GPU_ALLOCATED[$_ai]=$((_alloc + gpus))
                RR_IDX=$(( (_ai + 1) % _ns_count ))
                break
            fi

            _tried=$((_tried + 1))
        done
    fi

    # Overflow — all limited namespaces full, no unlimited available
    if [[ -z "$ns" ]]; then
        echo "WARNING: all namespaces at GPU quota; overflowing to first available" >&2
        ns="${ACTIVE_NS[0]}"
        NS_GPU_ALLOCATED[0]=$(( ${NS_GPU_ALLOCATED[0]} + gpus ))
    fi

    host=$(ns_ssh_host "$ns")
    gpu_product=$(ns_gpu_product "$ns")
    secrets=$(ns_secrets "$ns")
    username=$(ns_username "$ns")
    email=$(ns_email "$ns")

    # Find active NS index for job count tracking
    local_ns_idx=0
    for ai in $(seq 0 $(( ${#ACTIVE_NS[@]} - 1 ))); do
        if [[ "${ACTIVE_NS[$ai]}" == "$ns" ]]; then
            local_ns_idx=$ai
            break
        fi
    done

    # --- Local pre-check: skip if all tasks for this job are already complete ---
    task_list=$(task_names_for_script "$JOB_SCRIPT" "${EXTRA_ENVS[@]+"${EXTRA_ENVS[@]}"}")
    if [[ -n "$task_list" && "${FORCE:-}" != "true" ]]; then
        # Extract DECODING from EXTRA_ENVS (default: decore)
        _decoding="decore"
        _heads_label=""
        _sampling_seed=""
        for _e in "${EXTRA_ENVS[@]+"${EXTRA_ENVS[@]}"}"; do
            case "$_e" in
                DECODING=*)      _decoding="${_e#DECODING=}" ;;
                HEADS_LABEL=*)   _heads_label="${_e#HEADS_LABEL=}" ;;
                SAMPLING_SEED=*) _sampling_seed="${_e#SAMPLING_SEED=}" ;;
            esac
        done

        all_complete=true
        while IFS= read -r task_name; do
            check_exit=0
            "$LOCAL_PYTHON" scripts/check_experiment.py \
                --repo-id "${HF_RESULTS_REPO}" \
                --task "$task_name" \
                --model "$model" \
                --decoding "$_decoding" \
                --heads "$heads" \
                --heads-label "$_heads_label" \
                --sampling-seed "$_sampling_seed" \
                --quiet 2>/dev/null || check_exit=$?
            if [[ $check_exit -ne 0 ]]; then
                all_complete=false
                break
            fi
        done <<< "$task_list"
        if [[ "$all_complete" == true ]]; then
            echo "  SKIP: ${job_name} — all tasks already complete on HF"
            continue
        fi
    fi

    if [[ "$DRY_RUN" == true ]]; then
        printf "  %-55s -> %s (%s, %dx GPU)\n" "$job_name" "$ns" "$host" "$gpus"
    else
        # Scale CPU and RAM with GPU count (8 CPUs and 64Gi per GPU)
        cpus=$(( gpus * 8 ))
        ram=$(( gpus * 64 ))

        echo "Launching: ${job_name} -> ${ns} (${host}, ${gpus}x GPU, ${cpus} CPU, ${ram}Gi RAM)"

        run_cmd=$(build_generic_command "$model" "$heads" "$gpus" "$JOB_SCRIPT" "HEADS_METHOD=${method}" "${EXTRA_ENVS[@]+"${EXTRA_ENVS[@]}"}")

        # Write run_cmd to a temp file, copy to remote, execute kblaunch
        local_tmp=$(mktemp)
        echo "$run_cmd" > "$local_tmp"
        remote_tmp="/tmp/decore-cmd-${job_name}-$$.sh"

        # Copy command file to remote host
        scp -o "ControlPath=${SSH_CONTROL_DIR}/${host}" "$local_tmp" "${host}:${remote_tmp}" > /dev/null

        # Build email flag for kblaunch (annotations are set internally via config)
        email_flag=""
        [[ -n "$email" ]] && email_flag="--email '${email}'"

        # Launch via kblaunch on the remote host, reading command from file
        # Capture output; show only with --verbose or on failure
        launch_output=""
        launch_exit=0
        launch_output=$(remote_exec "$host" \
            "kblaunch launch \
                --job-name '${job_name}' \
                --namespace '${ns}' \
                --docker-image '${DOCKER_IMAGE}' \
                --gpu-limit '${gpus}' \
                --gpu-product '${gpu_product}' \
                --cpu-request ${cpus} \
                --ram-request ${ram}Gi \
                --command \"\$(cat ${remote_tmp})\" \
                --secrets-env-vars '${secrets}' \
                --secret-env-mapping 'GIT_TOKEN=${secrets}:aryo-git-token' \
                --secret-env-mapping 'HF_TOKEN=${secrets}:aryo-hf-token' \
                --secret-env-mapping 'HF_USERNAME=${secrets}:aryo-hf-username' \
                --secret-env-mapping 'WANDB_API_KEY=${secrets}:aryo-wandb-api-key' \
                --secret-env-mapping 'WANDB_ENTITY=${secrets}:aryo-wandb-entity' \
                ${email_flag} && rm -f '${remote_tmp}'" 2>&1) || launch_exit=$?

        if [[ "$VERBOSE" == true && -n "$launch_output" ]]; then
            echo "$launch_output"
        fi

        if [[ $launch_exit -ne 0 ]]; then
            echo "  ERROR: failed to launch ${job_name}" >&2
            if [[ "$VERBOSE" != true && -n "$launch_output" ]]; then
                echo "$launch_output" >&2
            fi
        fi

        rm -f "$local_tmp"
    fi

    NS_JOB_COUNTS[$local_ns_idx]=$(( ${NS_JOB_COUNTS[$local_ns_idx]} + 1 ))
    job_index=$((job_index + 1))
    total_jobs=$((total_jobs + 1))
done
done  # method loop

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "--------------------------------------------"
echo " Summary"
echo "--------------------------------------------"
echo "Total jobs: ${total_jobs}"
for ai in $(seq 0 $(( ${#ACTIVE_NS[@]} - 1 ))); do
    local_quota=$(ns_gpu_quota "${ACTIVE_NS[$ai]}")
    local_alloc=${NS_GPU_ALLOCATED[$ai]}
    if [[ "$local_quota" -gt 0 ]]; then
        echo "  ${ACTIVE_NS[$ai]}: ${NS_JOB_COUNTS[$ai]} jobs, ${local_alloc}/${local_quota} GPUs"
    else
        echo "  ${ACTIVE_NS[$ai]}: ${NS_JOB_COUNTS[$ai]} jobs, ${local_alloc} GPUs (no limit)"
    fi
done
[[ "$DRY_RUN" == true ]] && echo "(dry run -- no jobs were launched)"
echo ""

if [[ "$DRY_RUN" != true ]]; then
    echo "Tip: tear down SSH connections when done:"
    echo "  ./deploy/launch_multi_ns.sh --disconnect"
fi
