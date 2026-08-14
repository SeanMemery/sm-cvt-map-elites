#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_DIR="$ROOT_DIR/.tools/node-v22.11.0-linux-x64/bin"
export PATH="$NODE_DIR:$PATH"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8001}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-5173}"

if ! command -v node >/dev/null 2>&1; then
  echo "Node is not available. Expected local toolchain at $NODE_DIR" >&2
  exit 1
fi

cleanup() {
  jobs -p | xargs -r kill
}
trap cleanup EXIT

cd "$ROOT_DIR/viewer/frontend"
npm install

cd "$ROOT_DIR"
~/.local/bin/uv run python -m viewer.backend.main --host "$API_HOST" --port "$API_PORT" &

cd "$ROOT_DIR/viewer/frontend"
npm run dev -- --host "$WEB_HOST" --port "$WEB_PORT"
