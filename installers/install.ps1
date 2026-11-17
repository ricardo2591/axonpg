$ErrorActionPreference = "Stop"
Write-Host "[AxonPG] Bootstrapping Windows installer..."
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    irm https://astral.sh/uv/install.ps1 | iex
}
Write-Host "[AxonPG] Ready. Run 'uv run axonpg init' inside the AxonPG repository."
