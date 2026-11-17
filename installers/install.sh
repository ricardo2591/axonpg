#!/usr/bin/env bash
set -euo pipefail
echo "[AxonPG] Bootstrapping installer..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[AxonPG] Ready. Run 'uvx axonpg init' once published, or 'uv run axonpg init' locally."
