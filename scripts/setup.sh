#!/usr/bin/env bash
# Python environment in ./.venv, exactly as uv.lock pins it (installs uv itself if missing). Safe to rerun.
#   scripts/setup.sh            runtime + tests
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$(uname -s)-$(uname -m)" = Darwin-arm64 ] || { echo "error: this needs a Mac with Apple Silicon" >&2; exit 1; }
major=$(sw_vers -productVersion | cut -d. -f1)
[ "$major" -ge 14 ] || { echo "error: MLX needs macOS >= 14 (this is $(sw_vers -productVersion))" >&2; exit 1; }
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --frozen
.venv/bin/python -c "import mlx.core as mx; print('MLX', mx.__version__, 'on', mx.default_device())"
