param(
    [string]$InstallDir = "$HOME\.axonpg-app",
    [string]$RepoUrl = "https://github.com/ricardo2591/axonpg.git",
    [string]$Extras = "local",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$InitArgs
)

$ErrorActionPreference = "Stop"
Write-Host "=== AxonPG Installer (Windows) ===" -ForegroundColor Cyan

$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pyCmd) {
    throw "Python 3.11+ is required and was not found on PATH."
}

& python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.11+ is required."
}

if (-not (Test-Path (Join-Path $InstallDir ".git"))) {
    git clone --depth 1 $RepoUrl $InstallDir
} else {
    git -C $InstallDir pull --ff-only
}

Set-Location $InstallDir
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$venvPy = Join-Path $InstallDir ".venv\Scripts\python.exe"
$venvAxon = Join-Path $InstallDir ".venv\Scripts\axonpg.exe"

& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -e ".[$Extras]"
& $venvAxon init @InitArgs
