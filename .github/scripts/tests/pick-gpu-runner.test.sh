#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Behavior tests for pick-gpu-runner.sh. A fake `gh api` serves canned
# workflow-run/job JSON from MOCK_DIR so the picker's selection logic runs
# end to end with no network and no GitHub token.
#
# Run: bash .github/scripts/tests/pick-gpu-runner.test.sh

set -euo pipefail

PICK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/pick-gpu-runner.sh"
MOCK_DIR="$(mktemp -d)"
OUT="$MOCK_DIR/github_output"
RUN_ID=100
trap 'rm -rf "$MOCK_DIR"' EXIT

LINUX_LABELS='["self-hosted","Linux","X64","gpu"]'
WINDOWS_LABELS='["self-hosted","Windows","X64","liquidcortex"]'

EP_CANARY="repos/test/repo/actions/workflows/gpu-runner-canary.yml/runs?per_page=40"
EP_INPROG="repos/test/repo/actions/runs?status=in_progress&per_page=50"
EP_QUEUED="repos/test/repo/actions/runs?status=queued&per_page=50"
ep_jobs() { printf 'repos/test/repo/actions/runs/%s/jobs?per_page=100' "$1"; }

key() { printf '%s' "$1" | tr -c 'A-Za-z0-9' '_'; }
mock() { printf '%s' "$2" >"$MOCK_DIR/$(key "$1").json"; }

gh() {
    if [[ "$1" != "api" ]]; then
        echo "fake gh: unexpected args: $*" >&2
        return 1
    fi
    local f
    f="$MOCK_DIR/$(printf '%s' "$2" | tr -c 'A-Za-z0-9' '_').json"
    if [[ ! -f "$f" ]]; then
        echo "fake gh: no fixture for $2" >&2
        return 1
    fi
    cat "$f"
}
export -f gh
export MOCK_DIR

ts() { date -u -d "-$1 seconds" +%Y-%m-%dT%H:%M:%SZ; }

reset_mocks() {
    find "$MOCK_DIR" -name '*.json' -delete
    mock "$EP_INPROG" '{"workflow_runs":[]}'
    mock "$EP_QUEUED" '{"workflow_runs":[]}'
    mock "$EP_CANARY" '{"workflow_runs":[]}'
}

run_pick() {
    : >"$OUT"
    GITHUB_OUTPUT="$OUT" \
        GITHUB_REPOSITORY="test/repo" \
        GITHUB_RUN_ID="$RUN_ID" \
        GH_TOKEN="fake" \
        bash "$PICK" >/dev/null
    sed -n 's/^target=//p' "$OUT"
}

check() { # expected actual label
    if [[ "$2" != "$1" ]]; then
        echo "FAIL: $3 — expected target=$1, got $2" >&2
        exit 1
    fi
    echo "ok: $3 -> $2"
}

success_job() { # completed_at labels
    printf '{"status":"completed","conclusion":"success","completed_at":"%s","labels":%s}' "$1" "$2"
}

# --- Fresh Linux canary, idle host -> linux ----------------------------------
reset_mocks
mock "$EP_CANARY" "$(printf '{"workflow_runs":[{"id":1,"created_at":"%s"}]}' "$(ts 300)")"
mock "$(ep_jobs 1)" "$(printf '{"jobs":[%s]}' "$(success_job "$(ts 300)" "$LINUX_LABELS")")"
check linux "$(run_pick)" "fresh linux canary"

# --- Fresh Windows canary, idle host -> windows ------------------------------
reset_mocks
mock "$EP_CANARY" "$(printf '{"workflow_runs":[{"id":1,"created_at":"%s"}]}' "$(ts 300)")"
mock "$(ep_jobs 1)" "$(printf '{"jobs":[%s]}' "$(success_job "$(ts 300)" "$WINDOWS_LABELS")")"
check windows "$(run_pick)" "fresh windows canary"

# --- Cancelled newest canary falls back to older fresh success ---------------
reset_mocks
mock "$EP_CANARY" "$(printf '{"workflow_runs":[{"id":2,"created_at":"%s"},{"id":1,"created_at":"%s"}]}' "$(ts 120)" "$(ts 900)")"
mock "$(ep_jobs 2)" "$(printf '{"jobs":[{"status":"completed","conclusion":"cancelled","labels":%s}]}' "$LINUX_LABELS")"
mock "$(ep_jobs 1)" "$(printf '{"jobs":[%s]}' "$(success_job "$(ts 900)" "$LINUX_LABELS")")"
check linux "$(run_pick)" "cancelled canary falls back to success"

# --- Failed newest canary still masks older success (fail closed) ------------
reset_mocks
mock "$EP_CANARY" "$(printf '{"workflow_runs":[{"id":2,"created_at":"%s"},{"id":1,"created_at":"%s"}]}' "$(ts 120)" "$(ts 900)")"
mock "$(ep_jobs 2)" "$(printf '{"jobs":[{"status":"completed","conclusion":"failure","labels":%s}]}' "$LINUX_LABELS")"
mock "$(ep_jobs 1)" "$(printf '{"jobs":[%s]}' "$(success_job "$(ts 900)" "$LINUX_LABELS")")"
check skip "$(run_pick)" "failed canary masks success"

# --- Stale canary -> skip ----------------------------------------------------
reset_mocks
mock "$EP_CANARY" "$(printf '{"workflow_runs":[{"id":1,"created_at":"%s"}]}' "$(ts 3600)")"
mock "$(ep_jobs 1)" "$(printf '{"jobs":[%s]}' "$(success_job "$(ts 3600)" "$LINUX_LABELS")")"
check skip "$(run_pick)" "stale canary"

# --- In-progress GPU job on matching labels -> busy -> skip ------------------
reset_mocks
mock "$EP_CANARY" "$(printf '{"workflow_runs":[{"id":1,"created_at":"%s"}]}' "$(ts 300)")"
mock "$(ep_jobs 1)" "$(printf '{"jobs":[%s]}' "$(success_job "$(ts 300)" "$LINUX_LABELS")")"
mock "$EP_INPROG" "$(printf '{"workflow_runs":[{"id":9,"path":".github/workflows/gpu-ci.yml","created_at":"%s"}]}' "$(ts 120)")"
mock "$(ep_jobs 9)" "$(printf '{"jobs":[{"status":"in_progress","labels":%s}]}' "$LINUX_LABELS")"
check skip "$(run_pick)" "in-progress gpu job marks busy"

# --- Queued GPU job without runner_id -> busy -> skip ------------------------
reset_mocks
mock "$EP_CANARY" "$(printf '{"workflow_runs":[{"id":1,"created_at":"%s"}]}' "$(ts 300)")"
mock "$(ep_jobs 1)" "$(printf '{"jobs":[%s]}' "$(success_job "$(ts 300)" "$LINUX_LABELS")")"
mock "$EP_INPROG" "$(printf '{"workflow_runs":[{"id":9,"path":".github/workflows/gpu-ci.yml","created_at":"%s"}]}' "$(ts 120)")"
mock "$(ep_jobs 9)" "$(printf '{"jobs":[{"status":"queued","runner_id":null,"labels":%s}]}' "$LINUX_LABELS")"
check skip "$(run_pick)" "queued gpu job without runner_id marks busy"

# --- Queued run of another workflow -> busy -> skip --------------------------
reset_mocks
mock "$EP_CANARY" "$(printf '{"workflow_runs":[{"id":1,"created_at":"%s"}]}' "$(ts 300)")"
mock "$(ep_jobs 1)" "$(printf '{"jobs":[%s]}' "$(success_job "$(ts 300)" "$LINUX_LABELS")")"
mock "$EP_QUEUED" "$(printf '{"workflow_runs":[{"id":11,"path":".github/workflows/gpu-ci.yml","created_at":"%s"}]}' "$(ts 60)")"
mock "$(ep_jobs 11)" "$(printf '{"jobs":[{"status":"queued","labels":%s}]}' "$LINUX_LABELS")"
check skip "$(run_pick)" "queued run gpu job marks busy"

# --- Own run's queued GPU jobs must not self-block ---------------------------
reset_mocks
mock "$EP_CANARY" "$(printf '{"workflow_runs":[{"id":1,"created_at":"%s"}]}' "$(ts 300)")"
mock "$(ep_jobs 1)" "$(printf '{"jobs":[%s]}' "$(success_job "$(ts 300)" "$LINUX_LABELS")")"
mock "$EP_INPROG" "$(printf '{"workflow_runs":[{"id":100,"path":".github/workflows/gpu-ci.yml","created_at":"%s"}]}' "$(ts 60)")"
mock "$(ep_jobs 100)" "$(printf '{"jobs":[{"status":"queued","labels":%s}]}' "$LINUX_LABELS")"
check linux "$(run_pick)" "own run queued jobs ignored"

# --- Young queued canary does not mark busy ----------------------------------
reset_mocks
mock "$EP_CANARY" "$(printf '{"workflow_runs":[{"id":3,"created_at":"%s"},{"id":1,"created_at":"%s"}]}' "$(ts 240)" "$(ts 900)")"
mock "$(ep_jobs 3)" "$(printf '{"jobs":[{"status":"queued","labels":%s}]}' "$LINUX_LABELS")"
mock "$(ep_jobs 1)" "$(printf '{"jobs":[%s]}' "$(success_job "$(ts 900)" "$LINUX_LABELS")")"
check linux "$(run_pick)" "young queued canary ignored"

# --- Stale queued canary marks host not picking up work -> skip --------------
reset_mocks
mock "$EP_CANARY" "$(printf '{"workflow_runs":[{"id":3,"created_at":"%s"},{"id":1,"created_at":"%s"}]}' "$(ts 1200)" "$(ts 900)")"
mock "$(ep_jobs 3)" "$(printf '{"jobs":[{"status":"queued","labels":%s}]}' "$LINUX_LABELS")"
mock "$(ep_jobs 1)" "$(printf '{"jobs":[%s]}' "$(success_job "$(ts 900)" "$LINUX_LABELS")")"
check skip "$(run_pick)" "stale queued canary marks busy"

# --- Missing canary data -> skip ---------------------------------------------
reset_mocks
check skip "$(run_pick)" "no canaries"

echo "all pick-gpu-runner tests passed"
