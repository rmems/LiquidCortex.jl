# Install Julia 1.13.2 with the official juliaup portable build and put its
# bin directory on the runner service PATH. A Windows Store install only
# publishes execution aliases for the interactive user, which this service
# does not see.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

. (Join-Path $PSScriptRoot 'Add-RunnerServicePath.ps1')

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
    New-Item -ItemType Directory -Force -Path $defaultBin | Out-Null
    $archive = Join-Path $env:RUNNER_TEMP "juliaup-$JuliaupVersion-windows-portable.tar.gz"
    $asset = "juliaup-$JuliaupVersion-x86_64-pc-windows-gnu-portable.tar.gz"
    $uri = "https://github.com/JuliaLang/juliaup/releases/download/v$JuliaupVersion/$asset"
    Write-Host "Downloading $uri"
    Invoke-WebRequest -Uri $uri -OutFile $archive -UseBasicParsing
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
Add-RunnerServicePath $bin

& $juliaupExe add $JuliaVersion
if ($LASTEXITCODE -ne 0) {
    throw "juliaup add $JuliaVersion exited $LASTEXITCODE"
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
& $juliaExe -e 'VERSION == v"1.13.2" || error("expected Julia 1.13.2, got $VERSION")'
if ($LASTEXITCODE -ne 0) {
    throw "Julia version check failed"
}
