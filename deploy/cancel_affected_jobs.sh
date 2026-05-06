#!/usr/bin/env bash
# Cancel Kubernetes jobs affected by the find_needle_idx / tokenizer-drift bug.
#
# Designed to run INSIDE a cluster shell (kubectl context already points at the
# namespace). Requires: kubectl, python3. No jq, no sudo.
#
# By default, cancels:
#   - Detection jobs running Wu (behavioral) or logit_contrib detectors.
#   - Downstream (eval + ablation) jobs whose HEADS_LABEL or HEADS filename
#     indicates wu_niah / wu_nolima / logit_contrib_nolima heads.
#
# Includes suspended, pending, and active jobs. Skips jobs already Complete
# or Failed.
#
# Usage:
#   ./cancel_affected_jobs.sh                 # dry-run: diagnostic breakdown
#   ./cancel_affected_jobs.sh --list          # dry-run: print affected names only
#   ./cancel_affected_jobs.sh --delete        # actually delete (prompts y/N)
#   ./cancel_affected_jobs.sh --delete -y     # delete without prompt
#   ./cancel_affected_jobs.sh --include-contrastive  # also match contrastive detection
#   ./cancel_affected_jobs.sh -n eidf186ns    # run against specific namespace
set -euo pipefail

MODE="diagnostic"         # diagnostic | list | delete
ASSUME_YES=false
NAMESPACE=""
INCLUDE_CONTRASTIVE=false

usage() {
    sed -n '2,20p' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)                  MODE="list" ;;
        --delete)                MODE="delete" ;;
        -y|--yes)                ASSUME_YES=true ;;
        --include-contrastive)   INCLUDE_CONTRASTIVE=true ;;
        -n|--namespace)          NAMESPACE="$2"; shift ;;
        -h|--help)               usage 0 ;;
        *) echo "Unknown arg: $1" >&2; usage 1 ;;
    esac
    shift
done

KUBECTL=(kubectl)
[[ -n "$NAMESPACE" ]] && KUBECTL+=(-n "$NAMESPACE")

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

echo "Fetching jobs..." >&2
"${KUBECTL[@]}" get jobs -o json > "$TMP"

export MODE INCLUDE_CONTRASTIVE

python3 - "$TMP" <<'PYEOF'
import json, os, re, sys
from collections import defaultdict

MODE = os.environ["MODE"]
INCLUDE_CONTRASTIVE = os.environ["INCLUDE_CONTRASTIVE"] == "true"

DETECTORS_TO_CANCEL = {"detectors/behavioral.py", "detectors/logit_contrib.py"}
if INCLUDE_CONTRASTIVE:
    DETECTORS_TO_CANCEL.add("detectors/contrastive.py")

LABELS_TO_CANCEL = {"wu_niah", "wu_nolima", "logitcontrib_nolima"}
SUFFIXES_TO_CANCEL = ("_wu_niah", "_wu_nolima", "_logit_contrib_nolima")

LABEL_PAT  = re.compile(r"export HEADS_LABEL=['\"]?([A-Za-z0-9_]+)")
HEADS_PAT  = re.compile(r"export HEADS=['\"]?([^'\" \n]+)")
SCRIPT_PAT = re.compile(r"# --- Job script: ([^\n]+)")
TUNED_PAT  = re.compile(r"--tuned-lens(?:-url)?\b")

DETECTOR_KIND = {
    "detectors/behavioral.py":    "detect_wu",
    "detectors/contrastive.py":   "detect_contrastive",
    "detectors/logit_contrib.py": "detect_logit_contrib",
    "detectors/cri.py":           "detect_cri",
}

def classify(blob):
    for path, kind in DETECTOR_KIND.items():
        if path in blob:
            if kind == "detect_logit_contrib" and TUNED_PAT.search(blob):
                return "DETECT", "detect_logit_contrib_tuned_lens"
            return "DETECT", kind
    m_l = LABEL_PAT.search(blob)
    label = m_l.group(1) if m_l else None
    m_h = HEADS_PAT.search(blob)
    heads = m_h.group(1) if m_h else None
    suffix = None
    if heads:
        for s in SUFFIXES_TO_CANCEL:
            if heads.endswith(s + ".json"):
                suffix = s.lstrip("_")
                break
    if label or suffix:
        return "DOWNSTREAM", label or suffix
    return "OTHER", "<none>"

def is_affected(blob):
    if any(d in blob for d in DETECTORS_TO_CANCEL):
        return True
    m_l = LABEL_PAT.search(blob)
    if m_l and m_l.group(1) in LABELS_TO_CANCEL:
        return True
    m_h = HEADS_PAT.search(blob)
    if m_h and any(m_h.group(1).endswith(s + ".json") for s in SUFFIXES_TO_CANCEL):
        return True
    return False

def job_phase(j):
    st = j.get("status") or {}
    spec = j.get("spec") or {}
    if spec.get("suspend"):
        return "suspended"
    if st.get("active"):
        return f"active={st['active']}"
    if st.get("ready"):
        return f"ready={st['ready']}"
    return "pending"

data = json.load(open(sys.argv[1]))
buckets = defaultdict(list)     # (kind, method, script) -> [(name, phase)]
affected_names = []             # for list/delete modes

for j in data["items"]:
    st = j.get("status") or {}
    conds = st.get("conditions") or []
    if any(c.get("type") in ("Complete", "Failed") and c.get("status") == "True" for c in conds):
        continue  # nothing to cancel

    containers = j["spec"]["template"]["spec"].get("containers") or []
    if not containers:
        continue
    blob = " ".join(
        str(x)
        for x in (containers[0].get("command") or []) + (containers[0].get("args") or [])
    )

    name = j["metadata"]["name"]
    kind, method = classify(blob)
    m_script = SCRIPT_PAT.search(blob)
    script = m_script.group(1) if m_script else "?"
    buckets[(kind, method, script)].append((name, job_phase(j)))

    if is_affected(blob):
        affected_names.append(name)

if MODE == "diagnostic":
    print(f"{'KIND':<11} {'METHOD':<34} {'SCRIPT':<34} {'COUNT':>5}  SAMPLE")
    print("-" * 110)
    for key in sorted(buckets):
        kind, method, script = key
        rows = buckets[key]
        marker = "★" if kind == "DETECT" or method in LABELS_TO_CANCEL else " "
        print(f"{marker}{kind:<10} {method:<34} {script:<34} {len(rows):>5}  {rows[0][0]} [{rows[0][1]}]")
    print()
    print(f"Total affected jobs that would be cancelled: {len(affected_names)}")
    print("Re-run with --list to see names, --delete to cancel.")
elif MODE == "list":
    for name in sorted(affected_names):
        print(name)
    print(f"# {len(affected_names)} affected jobs", file=sys.stderr)
elif MODE == "delete":
    # Print to stdout so bash can capture for kubectl delete.
    for name in sorted(affected_names):
        print(name)
    print(f"# {len(affected_names)} affected jobs", file=sys.stderr)
PYEOF

PYEXIT=$?
[[ $PYEXIT -ne 0 ]] && exit $PYEXIT

if [[ "$MODE" == "delete" ]]; then
    # Re-run Python to get plain job-name list (piping from heredoc capture is
    # painful; just re-invoke with MODE=list).
    MODE=list python3 - "$TMP" <<'PYEOF2' > /tmp/cancel_jobs_list.$$
import json, os, re, sys
INCLUDE_CONTRASTIVE = os.environ["INCLUDE_CONTRASTIVE"] == "true"
DETECTORS_TO_CANCEL = {"detectors/behavioral.py", "detectors/logit_contrib.py"}
if INCLUDE_CONTRASTIVE:
    DETECTORS_TO_CANCEL.add("detectors/contrastive.py")
LABELS_TO_CANCEL = {"wu_niah", "wu_nolima", "logitcontrib_nolima"}
SUFFIXES_TO_CANCEL = ("_wu_niah", "_wu_nolima", "_logit_contrib_nolima")
LABEL_PAT = re.compile(r"export HEADS_LABEL=['\"]?([A-Za-z0-9_]+)")
HEADS_PAT = re.compile(r"export HEADS=['\"]?([^'\" \n]+)")

data = json.load(open(sys.argv[1]))
for j in data["items"]:
    st = j.get("status") or {}
    conds = st.get("conditions") or []
    if any(c.get("type") in ("Complete","Failed") and c.get("status")=="True" for c in conds):
        continue
    containers = j["spec"]["template"]["spec"].get("containers") or []
    if not containers: continue
    blob = " ".join(str(x) for x in (containers[0].get("command") or []) + (containers[0].get("args") or []))
    hit = any(d in blob for d in DETECTORS_TO_CANCEL)
    if not hit:
        m_l = LABEL_PAT.search(blob)
        if m_l and m_l.group(1) in LABELS_TO_CANCEL: hit = True
    if not hit:
        m_h = HEADS_PAT.search(blob)
        if m_h and any(m_h.group(1).endswith(s + ".json") for s in SUFFIXES_TO_CANCEL): hit = True
    if hit:
        print(j["metadata"]["name"])
PYEOF2
    LIST_FILE="/tmp/cancel_jobs_list.$$"
    trap 'rm -f "$TMP" "$LIST_FILE"' EXIT

    COUNT=$(wc -l < "$LIST_FILE" | tr -d ' ')
    if [[ "$COUNT" -eq 0 ]]; then
        echo "No affected jobs to delete."
        exit 0
    fi

    echo "About to delete $COUNT jobs:" >&2
    head -5 "$LIST_FILE" >&2
    [[ "$COUNT" -gt 5 ]] && echo "  ... (+$((COUNT - 5)) more)" >&2

    if [[ "$ASSUME_YES" != true ]]; then
        read -r -p "Proceed with deletion? [y/N] " reply
        [[ "$reply" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
    fi

    xargs -a "$LIST_FILE" -r "${KUBECTL[@]}" delete job --cascade=foreground
fi
