#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Choose an idle self-hosted GPU runner without the admin-only runners API.
# Availability is inferred from the gpu-runner-canary workflow: a fresh
# successful canary job means the runner was recently online, while a canary
# queued past its schedule interval, queued repo GPU jobs, and in-progress
# jobs rule out busy or offline hosts. GITHUB_TOKEN only needs actions:read.

set -euo pipefail

: "${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID is required}"
: "${GH_TOKEN:?GH_TOKEN is required}"

repo="${GITHUB_REPOSITORY}"
canary_workflow="gpu-runner-canary.yml"
heartbeat_minutes="${RUNNER_HEARTBEAT_MINUTES:-30}"
canary_runs="${RUNNER_CANARY_RUNS:-40}"
queue_minutes="${RUNNER_QUEUE_MINUTES:-10}"
now="$(date -u +%s)"

epoch_or_zero() {
    date -u -d "$1" +%s 2>/dev/null || echo 0
}

has_all_labels() {
    local job_json="$1"
    shift
    local label wanted
    for label in "$@"; do
        wanted="$label"
        if ! jq -e --arg wanted "$wanted" \
            '[.labels[]? | select(. == $wanted)] | length > 0' \
            <<<"$job_json" >/dev/null; then
            return 1
        fi
    done
}

is_linux_canary() {
    has_all_labels "$1" self-hosted Linux X64 gpu
}

is_windows_canary() {
    has_all_labels "$1" self-hosted Windows X64 liquidcortex
}

target_labels() {
    case "$1" in
        linux) printf '%s\n' self-hosted Linux X64 gpu ;;
        windows) printf '%s\n' self-hosted Windows X64 liquidcortex ;;
        *) return 1 ;;
    esac
}

job_matches_target() {
    local job_json="$1" target="$2" label wanted
    while IFS= read -r label; do
        [ -n "$label" ] || continue
        wanted="$label"
        if ! jq -e --arg wanted "$wanted" \
            '[.labels[]? | select(. == $wanted)] | length > 0' \
            <<<"$job_json" >/dev/null; then
            return 1
        fi
    done < <(target_labels "$target")
}

job_field() {
    jq -r --arg field "$2" '.[$field] // empty' <<<"$1"
}

classify_canary() {
    local job_json="$1"
    local status conclusion completed_at completed_epoch age
    status="$(job_field "$job_json" status)"
    conclusion="$(job_field "$job_json" conclusion)"
    completed_at="$(job_field "$job_json" completed_at)"

    if [[ "$status" == "completed" && "$conclusion" == "success" && -n "$completed_at" ]]; then
        completed_epoch="$(epoch_or_zero "$completed_at")"
        age=$((now - completed_epoch))
        if (( age <= heartbeat_minutes * 60 )); then
            echo "fresh"
        else
            echo "stale"
        fi
    elif [[ "$status" == "in_progress" ]]; then
        echo "busy"
    elif [[ "$status" =~ ^(queued|waiting|requested|pending)$ ]]; then
        echo "queued"
    else
        echo "unavailable"
    fi
}

runner_busy() {
    local target="$1"
    local run_status runs_json run_id run_path jobs_json job_json status

    # Scan in-progress and queued runs. A queued GPU job has no runner_id yet,
    # but on these single-host label sets it already claims the only matching
    # runner, so another ref must not pick the same host. Fail closed on API
    # errors: an unreadable run might be holding this runner.
    for run_status in in_progress queued; do
        if ! runs_json="$(gh api "repos/${repo}/actions/runs?status=${run_status}&per_page=50")"; then
            return 0
        fi
        while IFS=' ' read -r run_id run_path; do
            [ -n "$run_id" ] || continue
            # This run's own GPU jobs sit queued behind the picker and must
            # not mark the host busy.
            [[ "$run_id" == "$GITHUB_RUN_ID" ]] && continue
            if ! jobs_json="$(gh api "repos/${repo}/actions/runs/${run_id}/jobs?per_page=100")"; then
                return 0
            fi
            while IFS= read -r job_json; do
                [ -n "$job_json" ] || continue
                status="$(job_field "$job_json" status)"
                if [[ "$status" == "in_progress" ]] &&
                    job_matches_target "$job_json" "$target"; then
                    return 0
                fi
                # Queued canary jobs are handled by the queue-age check in
                # availability(), where only a stale queue marks the host.
                if [[ "$status" =~ ^(queued|waiting|requested|pending)$ &&
                    "$run_path" != */"${canary_workflow}" ]] &&
                    job_matches_target "$job_json" "$target"; then
                    return 0
                fi
            done < <(jq -c '.jobs[]?' <<<"$jobs_json")
        done < <(jq -r '.workflow_runs[]? | "\(.id) \(.path)"' <<<"$runs_json")
    done
    return 1
}

# LATEST_CANARY maps host -> newest conclusive canary job JSON. QUEUED_CANARY_AGE
# maps host -> seconds the newest still-queued canary run has been waiting.
declare -A LATEST_CANARY=()
declare -A QUEUED_CANARY_AGE=()

latest_canaries() {
    local runs_json run_id run_created jobs_json job_json
    local status conclusion host

    # A brand-new workflow file can 404 until its first run lands on main.
    # Treat that (and any API failure) as "no heartbeat" so GPU CI skips.
    if ! runs_json="$(gh api "repos/${repo}/actions/workflows/${canary_workflow}/runs?per_page=${canary_runs}")"; then
        runs_json='{"workflow_runs":[]}'
    fi
    while IFS=' ' read -r run_id run_created; do
        [ -n "$run_id" ] || continue
        # Runs arrive newest-first. Once both hosts have a conclusive canary,
        # only still-queued canaries from runs inside the queue window can
        # mark a host busy, so stop when the remaining runs predate it.
        if [[ -n "${LATEST_CANARY[linux]:-}" && -n "${LATEST_CANARY[windows]:-}" ]] &&
            (( now - $(epoch_or_zero "$run_created") > queue_minutes * 60 )); then
            break
        fi
        if ! jobs_json="$(gh api "repos/${repo}/actions/runs/${run_id}/jobs?per_page=100")"; then
            # Stop rather than fall back to older runs: a newer canary we
            # cannot read might be a failure that should hide the heartbeat.
            break
        fi
        while IFS= read -r job_json; do
            [ -n "$job_json" ] || continue
            host=
            if is_linux_canary "$job_json"; then
                host=linux
            elif is_windows_canary "$job_json"; then
                host=windows
            else
                continue
            fi
            status="$(job_field "$job_json" status)"
            if [[ "$status" == "completed" ]]; then
                conclusion="$(job_field "$job_json" conclusion)"
                # Cancelled/skipped canaries prove nothing: the canary's own
                # cancel-in-progress concurrency produces them constantly, so
                # fall through to the newest conclusive job instead of letting
                # one mask an older fresh success. Failure conclusions still
                # count — they fail closed.
                [[ "$conclusion" =~ ^(cancelled|skipped)$ ]] && continue
                [[ -z "${LATEST_CANARY[$host]:-}" ]] && LATEST_CANARY[$host]="$job_json"
            elif [[ "$status" =~ ^(queued|waiting|requested|pending)$ &&
                -z "${QUEUED_CANARY_AGE[$host]:-}" ]]; then
                QUEUED_CANARY_AGE[$host]=$((now - $(epoch_or_zero "$run_created")))
            fi
        done < <(jq -c '.jobs[]?' <<<"$jobs_json")
    done < <(jq -r '.workflow_runs[]? | "\(.id) \(.created_at)"' <<<"$runs_json")
    # The last read/jq in the loops above can exit non-zero; this function is
    # called under `set -e`, so return success explicitly.
    return 0
}

latest_canaries
linux_canary="${LATEST_CANARY[linux]:-}"
windows_canary="${LATEST_CANARY[windows]:-}"

availability() {
    local target="$1" canary_json="$2" state queued_age
    if [[ -z "$canary_json" ]]; then
        echo "missing"
        return
    fi
    state="$(classify_canary "$canary_json")"
    case "$state" in
        fresh)
            # A canary queued past its schedule interval means the runner is
            # not picking up new work, so its fresh heartbeat is already stale
            # in practice — do not trust it for the full heartbeat window.
            queued_age="${QUEUED_CANARY_AGE[$target]:-0}"
            if (( queued_age > queue_minutes * 60 )); then
                echo "busy"
            elif runner_busy "$target"; then
                echo "busy"
            else
                echo "available"
            fi
            ;;
        *)
            echo "$state"
            ;;
    esac
}

linux_state="$(availability linux "$linux_canary")"
windows_state="$(availability windows "$windows_canary")"

target="skip"
reason="no_runner_available"
if [[ "$linux_state" == "available" ]]; then
    target="linux"
    reason="linux_recent_canary"
elif [[ "$windows_state" == "available" ]]; then
    target="windows"
    reason="windows_recent_canary"
fi

{
    echo "target=${target}"
    echo "reason=${reason}"
} >>"$GITHUB_OUTPUT"

echo "pick-gpu-runner: target=${target} reason=${reason} (linux=${linux_state} windows=${windows_state})"
