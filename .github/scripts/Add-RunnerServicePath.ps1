# SPDX-License-Identifier: MIT OR Apache-2.0
# Prepend a directory to the current step, later steps in this job, and the
# Windows user/machine PATH. The GitHub Actions runner service reads the
# machine PATH when it starts; user PATH covers a runner launched as the
# desktop account. GITHUB_PATH is what this job can see without a service restart.
# Machine PATH updates need elevation and are best-effort.

function Add-RunnerServicePath {
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

    $persisted = $false
    foreach ($scope in @('User', 'Machine')) {
        try {
            $current = [Environment]::GetEnvironmentVariable('Path', $scope)
            $parts = @()
            if ($current) {
                $parts = @($current -split ';' | Where-Object { $_ })
            }
            $known = $false
            foreach ($part in $parts) {
                if ($part.Trim().TrimEnd('\') -ieq $full) {
                    $known = $true
                    break
                }
            }
            if ($known) {
                Write-Host "$full already on $scope PATH"
                $persisted = $true
                continue
            }
            $updated = if ([string]::IsNullOrEmpty($current)) { $full } else { "$full;$current" }
            [Environment]::SetEnvironmentVariable('Path', $updated, $scope)
            Write-Host "Added $full to $scope PATH"
            $persisted = $true
        } catch {
            Write-Host "Could not update $scope PATH: $($_.Exception.Message)"
        }
    }
    if (-not $persisted) {
        throw "Could not record $full on the user or machine PATH for the runner service."
    }
}
