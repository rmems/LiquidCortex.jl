#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Behavior tests for install-juliaup.ps1. A fake `juliaup` and `julia.exe` on
# PATH serve canned status/list output so the installer runs end to end under
# pwsh with no network and no real Julia.
#
# Run: bash .github/scripts/tests/install-juliaup.test.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOCK_DIR="$(mktemp -d)"
trap 'rm -rf "$MOCK_DIR"' EXIT

BIN="$MOCK_DIR/bin"
CALLS="$MOCK_DIR/calls"
INSTALLED="$MOCK_DIR/installed"
DEFAULT="$MOCK_DIR/default_channel"
CATALOG="$MOCK_DIR/catalog"
GITHUB_PATH_FILE="$MOCK_DIR/github_path"
mkdir -p "$BIN" "$MOCK_DIR/home" "$MOCK_DIR/runner-temp"
export FAKE_CALLS="$CALLS" FAKE_CATALOG="$CATALOG" \
    FAKE_INSTALLED="$INSTALLED" FAKE_DEFAULT="$DEFAULT"

# `juliaup list` prints the downloadable channel catalog whether or not the
# channel is installed locally; this catalog always contains 1.13.2.
cat >"$CATALOG" <<'EOF'
 Channel                 Version
-------------------------------------------------------------
 release                 1.13.4+0.x64.w64.mingw32
 1.13                    1.13.4+0.x64.w64.mingw32
 1.13.2                  1.13.2+0.x64.w64.mingw32
 1.13.2~x64              1.13.2+0.x64.w64.mingw32
EOF

# `juliaup status` mirrors the real installed-channels table; `add` is a
# successful no-op on an installed channel (juliaup 1.22.7 prints a notice to
# stderr and exits 0) and `default` fails on a missing one, matching juliaup.
cat >"$BIN/juliaup" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$FAKE_CALLS"
case "${1:-}" in
    list)
        cat "$FAKE_CATALOG"
        ;;
    status)
        printf ' Default  Channel  Version                   Update \n'
        printf -- '----------------------------------------------------\n'
        while IFS= read -r ch; do
            [ -n "$ch" ] || continue
            mark=' '
            [ "$ch" = "$(cat "$FAKE_DEFAULT" 2>/dev/null || true)" ] && mark='*'
            printf '       %s  %-9s 1.13.2+0.x64.w64.mingw32         \n' "$mark" "$ch"
        done <"$FAKE_INSTALLED"
        ;;
    add)
        ch="${2:-}"
        if grep -qxF "$ch" "$FAKE_INSTALLED"; then
            echo "'$ch' is already installed." >&2
            exit 0
        fi
        printf '%s\n' "$ch" >>"$FAKE_INSTALLED"
        ;;
    default)
        ch="${2:-}"
        if ! grep -qxF "$ch" "$FAKE_INSTALLED"; then
            echo "channel '$ch' is not installed; run 'juliaup add $ch' first" >&2
            exit 1
        fi
        printf '%s\n' "$ch" >"$FAKE_DEFAULT"
        ;;
    *)
        echo "fake juliaup: unexpected args: $*" >&2
        exit 1
        ;;
esac
EOF

cat >"$BIN/julia.exe" <<'EOF'
#!/usr/bin/env bash
if [ "${1:-}" = '--version' ]; then
    echo 'julia version 1.13.2'
fi
exit 0
EOF
chmod +x "$BIN/juliaup" "$BIN/julia.exe"

fail() { echo "FAIL: $1" >&2; exit 1; }
ok() { echo "ok: $1"; }
called() { grep -qxF "$1" "$CALLS"; }

run_install() {
    : >"$CALLS"
    : >"$GITHUB_PATH_FILE"
    env \
        PATH="$BIN:$PATH" \
        HOME="$MOCK_DIR/home" \
        USERPROFILE="$MOCK_DIR/home" \
        RUNNER_TEMP="$MOCK_DIR/runner-temp" \
        GITHUB_PATH="$GITHUB_PATH_FILE" \
        pwsh -NoProfile -File "$ROOT/install-juliaup.ps1" >"$MOCK_DIR/install.log" 2>&1
}

expect_install_ok() {
    if ! run_install; then
        cat "$MOCK_DIR/install.log" >&2
        fail 'install-juliaup.ps1 exited nonzero'
    fi
}

# --- Sanity: the fake reproduces the juliaup contract ------------------------
printf 'release\n' >"$INSTALLED"
"$BIN/juliaup" list | grep -q '1\.13\.2' ||
    fail 'test premise broken: catalog is missing 1.13.2'
if "$BIN/juliaup" default 1.13.2 2>/dev/null; then
    fail 'test premise broken: fake juliaup accepted default for a missing channel'
fi
ok 'fake juliaup: list shows catalog channels, default rejects missing ones'

# --- Catalog-only 1.13.2 (listed, not installed) -> add must run -------------
expect_install_ok
called 'status' || fail 'installer did not inspect juliaup status'
if ! called 'add 1.13.2'; then
    fail 'catalog-only channel skipped juliaup add'
fi
called 'default 1.13.2' || fail 'juliaup default 1.13.2 was not invoked'
ok 'catalog-only channel still gets juliaup add and default'

# --- 1.13.2 installed but not the default (unmarked row) -> add is skipped ---
printf '1.13.2\nrelease\n' >"$INSTALLED"
printf 'release\n' >"$DEFAULT"
expect_install_ok
if called 'add 1.13.2'; then
    fail 'juliaup add ran for an already-installed channel'
fi
called 'default 1.13.2' || fail 'juliaup default 1.13.2 was not invoked'
ok 'unmarked installed channel skips juliaup add'

# --- Default-marked installed channel is still detected -----------------------
printf 'release\n1.13.2\n' >"$INSTALLED"
printf '1.13.2\n' >"$DEFAULT"
expect_install_ok
if called 'add 1.13.2'; then
    fail 'juliaup add ran for the default channel'
fi
ok 'default-marked channel (*) still counts as installed'

echo 'all install-juliaup tests passed'
