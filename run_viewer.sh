#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_DIR="$ROOT_DIR/.tools/node-v22.11.0-linux-x64/bin"
export PATH="$NODE_DIR:$PATH"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if ! command -v node >/dev/null 2>&1; then
  echo "Node is not available. Expected local toolchain at $NODE_DIR" >&2
  exit 1
fi

cd "$ROOT_DIR/viewer/frontend"
npm install
npm run build

cd "$ROOT_DIR"
~/.local/bin/uv run python -m viewer.backend.main --host "$HOST" --port "$PORT"
