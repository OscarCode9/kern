#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
API_STARTED_BY_SCRIPT=0

cleanup() {
  if [[ "${API_STARTED_BY_SCRIPT}" == "1" && -n "${API_PID:-}" ]]; then
    kill "${API_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

cd "${ROOT_DIR}"

if curl -fsS "http://127.0.0.1:${API_PORT}/api/health" >/dev/null 2>&1; then
  echo "Using existing API on http://127.0.0.1:${API_PORT}"
else
  if lsof -nP -iTCP:"${API_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port ${API_PORT} is already in use by a non-responsive process."
    lsof -nP -iTCP:"${API_PORT}" -sTCP:LISTEN || true
    echo "Stop that process or set API_PORT to another value."
    exit 1
  fi

  python3 -m uvicorn backend.main:app --reload --port "${API_PORT}" >/tmp/kern_api.log 2>&1 &
  API_PID=$!
  API_STARTED_BY_SCRIPT=1

  echo "Starting API on http://127.0.0.1:${API_PORT} (pid ${API_PID})"

  for _ in {1..50}; do
    if curl -fsS "http://127.0.0.1:${API_PORT}/api/health" >/dev/null 2>&1; then
      break
    fi
    sleep 0.2
  done

  if ! curl -fsS "http://127.0.0.1:${API_PORT}/api/health" >/dev/null 2>&1; then
    echo "API failed to start. Check /tmp/kern_api.log"
    tail -n 80 /tmp/kern_api.log || true
    exit 1
  fi
fi

if lsof -nP -iTCP:"${WEB_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port ${WEB_PORT} is already in use. Stop the current frontend process or set WEB_PORT."
  lsof -nP -iTCP:"${WEB_PORT}" -sTCP:LISTEN || true
  exit 1
fi

echo "API ready. Starting frontend on http://127.0.0.1:${WEB_PORT}"
cd "${ROOT_DIR}/web"
npm run dev:web -- --host 127.0.0.1 --port "${WEB_PORT}" --strictPort
