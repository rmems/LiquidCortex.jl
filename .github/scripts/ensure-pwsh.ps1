# SPDX-License-Identifier: MIT OR Apache-2.0
# Install PowerShell 7 when the Windows runner does not already have pwsh.
# Later workflow steps set `shell: pwsh`. Windows PowerShell 5.1 can run this
# script; it records the pwsh directory on the runner service PATH.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

. (Join-Path $PSScriptRoot 'Add-RunnerServicePath.ps1')

$pwshExe = $null
$cmd = Get-Command pwsh -ErrorAction SilentlyContinue
if ($cmd) {
    $pwshExe = $cmd.Source
}

if (-not $pwshExe) {
    $candidates = @(
        (Join-Path $env:USERPROFILE '.local\pwsh\pwsh.exe'),
        'C:\Program Files\PowerShell\7\pwsh.exe'
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            $pwshExe = $candidate
            break
        }
    }
}

if (-not $pwshExe) {
    $dest = Join-Path $env:USERPROFILE '.local\pwsh'
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $zip = Join-Path $env:RUNNER_TEMP 'PowerShell-7.6.7-win-x64.zip'
    $uri = 'https://github.com/PowerShell/PowerShell/releases/download/v7.6.7/PowerShell-7.6.7-win-x64.zip'
    Write-Host "Downloading $uri"
    Invoke-WebRequest -Uri $uri -OutFile $zip -UseBasicParsing
    Expand-Archive -LiteralPath $zip -DestinationPath $dest -Force
    $pwshExe = Join-Path $dest 'pwsh.exe'
}

if (-not (Test-Path -LiteralPath $pwshExe)) {
    throw "pwsh.exe not found at $pwshExe"
}

Add-RunnerServicePath (Split-Path -Parent $pwshExe)
& $pwshExe -NoProfile -Command 'Write-Output $PSVersionTable.PSVersion.ToString()'
if ($LASTEXITCODE -ne 0) {
    throw "pwsh exited $LASTEXITCODE"
}
