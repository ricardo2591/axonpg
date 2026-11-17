#!/usr/bin/env bash
set -euo pipefail

AXONPG_HOME="${AXONPG_HOME:-$HOME/.axonpg-app}"
AXONPG_REPO="${AXONPG_REPO:-https://github.com/ricardo2591/axonpg.git}"
AXONPG_EXTRAS="${AXONPG_EXTRAS:-local}"

echo "=== AxonPG Installer (Linux / macOS) ==="

if command -v python3 >/dev/null 2>&1; then
  PY_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="python"
else
  echo "Error: Python 3.11+ is required." >&2
  exit 1
fi

"$PY_BIN" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" || {
  echo "Error: Python 3.11+ is required." >&2
  exit 1
}

if [[ ! -d "$AXONPG_HOME/.git" ]]; then
  git clone --depth 1 "$AXONPG_REPO" "$AXONPG_HOME"
else
  git -C "$AXONPG_HOME" pull --ff-only || true
fi

cd "$AXONPG_HOME"

if command -v uv >/dev/null 2>&1; then
  uv venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -e ".[$AXONPG_EXTRAS]"
else
  "$PY_BIN" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -e ".[$AXONPG_EXTRAS]"
fi

exec "$AXONPG_HOME/.venv/bin/axonpg" init "$@"
