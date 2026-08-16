[CmdletBinding()]
param(
    [string]$EnvironmentName = "na-lmscnet"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$EnvironmentFile = Join-Path $ProjectRoot "code\environment.yml"
$VerificationScript = Join-Path $ProjectRoot "code\scripts\verify_environment.py"
$CondaPackages = @(
    "python=3.11.15",
    "pip=26.2.1",
    "numpy=2.4.6",
    "scipy=1.17.1",
    "scikit-learn=1.8.0",
    "pandas=3.0.5",
    "h5py=3.15.1",
    "matplotlib=3.10.9",
    "seaborn=0.13.2",
    "pyyaml=6.0.3",
    "tqdm=4.67.3",
    "pytest=9.0.3",
    "pytest-cov=7.0.0",
    "ruff=0.15.22",
    "pip-audit=2.10.1"
)

function Find-CondaExecutable {
    $command = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "miniconda3\Scripts\conda.exe"),
        "D:\Tool\Miniconda\Scripts\conda.exe"
    )

    $registryPaths = @(
        "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*"
    )
    $entries = Get-ItemProperty $registryPaths -ErrorAction SilentlyContinue |
        Where-Object { $_.DisplayName -like "Miniconda3*" }
    foreach ($entry in $entries) {
        if ($entry.InstallLocation) {
            $candidates += Join-Path $entry.InstallLocation "Scripts\conda.exe"
        }
        if ($entry.UninstallString -match '^"?([^"]+Uninstall-Miniconda3\.exe)') {
            $root = Split-Path $Matches[1] -Parent
            $candidates += Join-Path $root "Scripts\conda.exe"
        }
    }

    return $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
}

$Conda = Find-CondaExecutable
if (-not $Conda) {
    $Winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $Winget) {
        throw "Miniconda and winget were not found. Install Miniconda and retry."
    }

    Write-Host "Miniconda was not found. Installing it with winget..."
    & $Winget.Source install --id Anaconda.Miniconda3 --exact --scope user --silent `
        --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "Miniconda installation failed with winget exit code $LASTEXITCODE."
    }
    $Conda = Find-CondaExecutable
    if (-not $Conda) {
        throw "Miniconda was installed, but conda.exe was not found. Restart the terminal and retry."
    }
}

$environmentInfo = & $Conda env list --json | ConvertFrom-Json
$environmentExists = $environmentInfo.envs | Where-Object {
    (Split-Path $_ -Leaf) -eq $EnvironmentName
}

if (-not $environmentExists) {
    Write-Host "Creating Conda environment $EnvironmentName from conda-forge..."
    & $Conda create --name $EnvironmentName --yes --override-channels --channel conda-forge @CondaPackages
    if ($LASTEXITCODE -ne 0) {
        throw "Conda environment creation failed with exit code $LASTEXITCODE."
    }
}

Write-Host "Synchronizing Conda packages from conda-forge..."
& $Conda install --name $EnvironmentName --yes --override-channels --channel conda-forge @CondaPackages
if ($LASTEXITCODE -ne 0) {
    throw "Conda package synchronization failed with exit code $LASTEXITCODE."
}

Write-Host "Installing the pinned PyTorch CUDA wheel..."
& $Conda run --no-capture-output --name $EnvironmentName `
    python -m pip install --disable-pip-version-check --index-url https://download.pytorch.org/whl/cu130 `
    "torch==2.13.0+cu130"
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch installation failed with exit code $LASTEXITCODE."
}

Write-Host "Verifying Python, dependencies, and CUDA..."
& $Conda run --no-capture-output --name $EnvironmentName `
    python $VerificationScript --require-cuda
if ($LASTEXITCODE -ne 0) {
    throw "Environment verification failed with exit code $LASTEXITCODE."
}

Write-Host "Environment is ready: $EnvironmentName"
Write-Host "Run project scripts with: conda run -n $EnvironmentName python <script.py>"
