# SPDX-License-Identifier: MIT OR Apache-2.0
# Prepend a directory to PATH for the rest of this GitHub Actions job only.
# Unlike the runner service PATH, GITHUB_PATH is not persisted on the host.

function Add-GithubPath {
    param([Parameter(Mandatory = $true)][string] $Directory)

    if (-not $env:GITHUB_PATH) {
        throw 'GITHUB_PATH is not set; this script is for a GitHub Actions job.'
    }
    $full = [System.IO.Path]::GetFullPath($Directory).TrimEnd('\')
    if (-not (Test-Path -LiteralPath $full)) {
        throw "PATH directory does not exist: $full"
    }

    $env:Path = "$full;$env:Path"
    Add-Content -Path $env:GITHUB_PATH -Value $full
    Write-Host "Prepended to job PATH: $full"
}
