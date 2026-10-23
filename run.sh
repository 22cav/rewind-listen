#!/usr/bin/env bash
# On Apple Silicon. Start with  ./run.sh setup , then  ./run.sh help  lists every command.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${1:-help}" = setup ] && [ ! -x "$ROOT/.venv/bin/python" ]; then
  "$ROOT/scripts/setup.sh"
fi
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "Not installed yet: run  ./run.sh setup" >&2; exit 1
fi
exec "$ROOT/.venv/bin/python" "$ROOT/listen/cli.py" "$@"
