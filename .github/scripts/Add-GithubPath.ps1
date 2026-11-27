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
    # Append as UTF-8 without BOM: under Windows PowerShell 5.1 Add-Content
    # would write ANSI and corrupt non-ASCII paths for the runner's reader.
    [System.IO.File]::AppendAllText(
        $env:GITHUB_PATH,
        "$full`r`n",
        [System.Text.UTF8Encoding]::new($false)
    )
    Write-Host "Prepended to job PATH: $full"
}
