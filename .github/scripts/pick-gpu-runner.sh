#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Choose an idle self-hosted GPU runner without the admin-only runners API.
# Availability is inferred from the gpu-runner-canary workflow: a fresh
# successful canary job means the runner was recently online, while the latest
# queued/in-progress canary job and repo in-progress jobs rule out busy or
# offline hosts. GITHUB_TOKEN only needs actions:read.

set -euo pipefail

: "${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GH_TOKEN:?GH_TOKEN is required}"

repo="${GITHUB_REPOSITORY}"
canary_workflow="gpu-runner-canary.yml"
heartbeat_minutes="${RUNNER_HEARTBEAT_MINUTES:-30}"
canary_runs="${RUNNER_CANARY_RUNS:-40}"
now="$(date -u +%s)"

epoch_or_zero() {
    date -u -d "$1" +%s 2>/dev/null || echo 0
}

has_all_labels() {
    local job_json="$1"
    shift
    local label
    for label in "$@"; do
        if ! jq -e --arg label "$label" \
            '[.labels[]? | select(. == $label)] | length > 0' \
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
    local job_json="$1" target="$2" label
    while IFS= read -r label; do
        [ -n "$label" ] || continue
        if ! jq -e --arg label "$label" \
            '[.labels[]? | select(. == $label)] | length > 0' \
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
    local runs_json run_id jobs_json job_json status runner_id

    # Fail closed: if busy state cannot be read, do not risk queueing GPU work.
    if ! runs_json="$(gh api "repos/${repo}/actions/runs?status=in_progress&per_page=50")"; then
        return 0
    fi
    while IFS= read -r run_id; do
        [ -n "$run_id" ] || continue
        if ! jobs_json="$(gh api "repos/${repo}/actions/runs/${run_id}/jobs?per_page=100")"; then
            continue
        fi
        while IFS= read -r job_json; do
            [ -n "$job_json" ] || continue
            status="$(job_field "$job_json" status)"
            runner_id="$(job_field "$job_json" runner_id)"
            if [[ "$status" == "in_progress" || ( "$status" == "queued" && -n "$runner_id" ) ]] &&
                job_matches_target "$job_json" "$target"; then
                return 0
            fi
        done < <(jq -c '.jobs[]?' <<<"$jobs_json")
    done < <(jq -r '.workflow_runs[]?.id' <<<"$runs_json")
    return 1
}

latest_canaries() {
    local runs_json run_id jobs_json job_json
    local -A seen=()

    # A brand-new workflow file can 404 until its first run lands on main.
    # Treat that (and any API failure) as "no heartbeat" so GPU CI skips.
    if ! runs_json="$(gh api "repos/${repo}/actions/workflows/${canary_workflow}/runs?per_page=${canary_runs}")"; then
        runs_json='{"workflow_runs":[]}'
    fi
    while IFS= read -r run_id; do
        [ -n "$run_id" ] || continue
        if ! jobs_json="$(gh api "repos/${repo}/actions/runs/${run_id}/jobs?per_page=100")"; then
            continue
        fi
        while IFS= read -r job_json; do
            [ -n "$job_json" ] || continue
            if [[ -z "${seen[linux]:-}" ]] && is_linux_canary "$job_json"; then
                seen[linux]="$job_json"
            fi
            if [[ -z "${seen[windows]:-}" ]] && is_windows_canary "$job_json"; then
                seen[windows]="$job_json"
            fi
            if [[ -n "${seen[linux]:-}" && -n "${seen[windows]:-}" ]]; then
                break
            fi
        done < <(jq -c '.jobs[]?' <<<"$jobs_json")
        if [[ -n "${seen[linux]:-}" && -n "${seen[windows]:-}" ]]; then
            break
        fi
    done < <(jq -r '.workflow_runs[]?.id' <<<"$runs_json")

    printf '%s\n%s\n' "${seen[linux]:-}" "${seen[windows]:-}"
}

mapfile -t canaries < <(latest_canaries)
linux_canary="${canaries[0]:-}"
windows_canary="${canaries[1]:-}"

availability() {
    local target="$1" canary_json="$2" state
    if [[ -z "$canary_json" ]]; then
        echo "missing"
        return
    fi
    state="$(classify_canary "$canary_json")"
    case "$state" in
        fresh)
            if runner_busy "$target"; then
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
