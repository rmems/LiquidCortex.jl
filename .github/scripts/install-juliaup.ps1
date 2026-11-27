# SPDX-License-Identifier: MIT OR Apache-2.0
# Install Julia 1.13.2 with the official juliaup portable build for this job.
# A Windows Store install only publishes execution aliases for the interactive
# user, which the runner service does not see. juliaup's bin directory is added
# through GITHUB_PATH for this job only.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

. (Join-Path $PSScriptRoot 'Add-GithubPath.ps1')

$JuliaVersion = '1.13.2'
$JuliaupVersion = '1.22.7'

$juliaupExe = $null
$cmd = Get-Command juliaup -ErrorAction SilentlyContinue
if ($cmd) {
    $juliaupExe = $cmd.Source
}

$defaultBin = Join-Path $env:USERPROFILE '.juliaup\bin'
if (-not $juliaupExe) {
    $bundled = Join-Path $defaultBin 'juliaup.exe'
    if (Test-Path -LiteralPath $bundled) {
        $juliaupExe = $bundled
    }
}

if (-not $juliaupExe) {
    if (-not $env:RUNNER_TEMP) {
        throw 'RUNNER_TEMP is not set; this script must run inside a GitHub Actions job.'
    }
    New-Item -ItemType Directory -Force -Path $defaultBin | Out-Null
    $archive = Join-Path $env:RUNNER_TEMP "juliaup-$JuliaupVersion-windows-portable.tar.gz"
    $asset = "juliaup-$JuliaupVersion-x86_64-pc-windows-gnu-portable.tar.gz"
    $uri = "https://github.com/JuliaLang/juliaup/releases/download/v$JuliaupVersion/$asset"
    # GitHub release asset digest for the Windows portable juliaup build.
    $expectedHash = '040b688fc39e2f48a7c7c7ee83cbda88131fa6230ed68db24b2a19a40eb39849'
    Write-Host "Downloading $uri"
    Invoke-WebRequest -Uri $uri -OutFile $archive -UseBasicParsing
    $actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLower()
    if ($actualHash -ne $expectedHash) {
        throw "SHA256 mismatch for ${archive}: ${actualHash}"
    }
    & tar.exe -xzf $archive -C $defaultBin
    if ($LASTEXITCODE -ne 0) {
        throw "tar exited $LASTEXITCODE extracting juliaup"
    }
    $juliaupExe = Join-Path $defaultBin 'juliaup.exe'
}

if (-not (Test-Path -LiteralPath $juliaupExe)) {
    throw "juliaup.exe not found at $juliaupExe"
}

$bin = Split-Path -Parent $juliaupExe
Add-GithubPath $bin

# juliaup persists between jobs on this self-hosted host, and `juliaup add`
# fails on a channel that is already installed, so check first.
$channelPattern = "^\s*\*?\s*$([regex]::Escape($JuliaVersion))(\s|\*|$)"
$alreadyInstalled = & $juliaupExe list | Select-String -Pattern $channelPattern
if (-not $alreadyInstalled) {
    & $juliaupExe add $JuliaVersion
    if ($LASTEXITCODE -ne 0) {
        throw "juliaup add $JuliaVersion exited $LASTEXITCODE"
    }
}
& $juliaupExe default $JuliaVersion
if ($LASTEXITCODE -ne 0) {
    throw "juliaup default $JuliaVersion exited $LASTEXITCODE"
}

$juliaExe = Join-Path $bin 'julia.exe'
& $juliaExe --version
if ($LASTEXITCODE -ne 0) {
    throw "julia --version exited $LASTEXITCODE"
}
& $juliaupExe status

# Run the version check from a file: Windows PowerShell 5.1 mangles the
# embedded quotes that `julia -e 'v"1.13.2" ...'` needs.
$versionCheck = Join-Path ([System.IO.Path]::GetTempPath()) 'check-julia-version.jl'
$versionCheckCode = @'
(VERSION.major == 1 && VERSION.minor == 13 && VERSION.patch >= 2) ||
    error("expected Julia 1.13.2 or a newer 1.13 patch, got $VERSION")
'@
[System.IO.File]::WriteAllText(
    $versionCheck,
    $versionCheckCode,
    [System.Text.UTF8Encoding]::new($false)
)
& $juliaExe --pkgimages=no $versionCheck
if ($LASTEXITCODE -ne 0) {
    throw "Julia version check failed"
}
