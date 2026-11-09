#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Emit pick-gpu-runner outputs: target=linux|windows|skip
# Requires: gh, jq, GITHUB_OUTPUT, GITHUB_REPOSITORY, and a token that can list
# self-hosted runners (workflow permissions: actions: read).

set -euo pipefail

if [[ -z "${GITHUB_OUTPUT:-}" ]]; then
    echo "GITHUB_OUTPUT is not set" >&2
    exit 1
fi

runner_json='{"runners":[]}'
if runner_json="$(gh api "repos/${GITHUB_REPOSITORY}/actions/runners?per_page=100" --paginate 2>/dev/null)"; then
    :
else
    echo "Could not list self-hosted runners; treating as skip to avoid queuing." >&2
    {
        echo "target=skip"
        echo "reason=api_error"
    } >>"$GITHUB_OUTPUT"
    exit 0
fi

match_linux() {
    jq -e '
        [.runners[]
         | select(.status == "online" and .busy == false)
         | select([.labels[].name] | index("gpu") != null)
         | select([.labels[].name] | index("Linux") != null)]
        | length > 0
    ' <<<"$runner_json" >/dev/null
}

match_windows() {
    jq -e '
        [.runners[]
         | select(.status == "online" and .busy == false)
         | select([.labels[].name] | index("liquidcortex") != null)
         | select([.labels[].name] | index("Windows") != null)]
        | length > 0
    ' <<<"$runner_json" >/dev/null
}

target="skip"
reason="none_online"
if match_linux; then
    target="linux"
    reason="linux_gpu_online"
elif match_windows; then
    target="windows"
    reason="windows_liquidcortex_online"
fi

{
    echo "target=${target}"
    echo "reason=${reason}"
} >>"$GITHUB_OUTPUT"

echo "pick-gpu-runner: target=${target} (${reason})"
