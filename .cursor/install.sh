#!/usr/bin/env bash
# Cursor Cloud Agent install script for LiquidCortex.jl (`install` in .cursor/environment.json).
#
# Cursor runs this from the repository root during every Build, on its default
# Ubuntu base image (CPU only: cloud agents have no GPU), then snapshots the disk.
# It must be idempotent. Shell exports don't survive into agent runs, so the tools
# it installs are exposed through /etc/profile.d and /usr/local/bin.
# See https://cursor.com/docs/cloud-agent/setup
#
# Installs only what this repo's CI and manifests need:
#   - apt: curl, ca-certificates
#   - juliaup 1.13 [default]
#   - Pkg.instantiate() + Pkg.precompile() (as in ci.yml)
#   - tool directories exposed to later shells (/etc/profile.d + /usr/local/bin links)
#
# It ends with a dependency fetch/prebuild, not a test run.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"; rm -f Manifest.toml' EXIT

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# Install apt packages that are not already present.
apt_install() {
  local missing=() pkg
  for pkg in "$@"; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
      missing+=("$pkg")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    $SUDO apt-get -o Acquire::Retries=5 update -qq
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=5 install -y --no-install-recommends "${missing[@]}"
  fi
}

# --- System packages (curl + CA certs for the installers below) ---
apt_install curl ca-certificates

# --- Julia (juliaup) ---
# ci.yml uses Julia 1.13 (gpu-ci.yml asserts 1.13).
JULIA_CHANNEL="1.13"
export PATH="$HOME/.juliaup/bin:$PATH"
if ! command -v juliaup >/dev/null 2>&1; then
  installer="$tmp_dir/juliainstaller"
  curl -fsSL https://julialang-s3.julialang.org/juliaup/bin/juliainstaller-1.22.7-x86_64-unknown-linux-musl -o "$installer"
  echo "2f6cb3beff3e14b78890fb52dc6d44ec2d2a5bf2c703d2d478d6e519c863eae0  $installer" | sha256sum --check
  chmod +x "$installer"
  "$installer" --yes --default-channel "$JULIA_CHANNEL"
fi
if ! juliaup status | awk -v ch="$JULIA_CHANNEL" '{for (i = 1; i <= NF; i++) if ($i == ch) found = 1} END {exit !found}'; then
  juliaup add "$JULIA_CHANNEL"
fi
juliaup default "$JULIA_CHANNEL"

# --- Julia dependencies ---
# Library consumers resolve freely; do not persist an ignored setup-time lockfile.
rm -f Manifest.toml
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# --- Expose the tools to later shells ---
# The PATH exports above last only for this script; Cursor starts the agent's shells
# separately. Login shells get these directories from /etc/profile.d, and every other
# shell finds the entry points through symlinks in /usr/local/bin (on the default PATH).
tool_dirs=("$HOME/.juliaup/bin")
# shellcheck disable=SC2016 # $PATH must expand when the profile is sourced, not now.
printf 'export PATH="%s:$PATH"\n' "$(IFS=:; echo "${tool_dirs[*]}")" |
  $SUDO tee /etc/profile.d/cursor-env-LiquidCortex.jl.sh >/dev/null
for dir in "${tool_dirs[@]}"; do
  [ -d "$dir" ] || continue
  for tool in "$dir"/*; do
    name="${tool##*/}"
    case "$name" in
      python* | pip* | activate* | deactivate | Activate.ps1) continue ;;
    esac
    if [ -f "$tool" ] && [ -x "$tool" ]; then
      $SUDO ln -sfn "$tool" "/usr/local/bin/$name"
    fi
  done
done

echo "Cursor install for LiquidCortex.jl finished."
