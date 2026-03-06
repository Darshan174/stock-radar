#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

echo "[devrel-validate] mode=$MODE"
echo "[devrel-validate] repo=$ROOT_DIR"

run_quick() {
  echo "[1/4] Collect tests"
  pytest --collect-only -q

  echo "[2/4] Core routing/risk/training test slice"
  pytest tests/test_llm_routing.py tests/test_training.py tests/test_phase10.py -q

  echo "[3/4] Static quality checks (non-blocking in quick mode)"
  if ! ruff check src/ tests/ backend/; then
    echo "WARNING: Ruff reported issues in existing repository files."
    echo "WARNING: Treat this as a review exercise during the DevRel track."
  fi

  echo "[4/4] Done"
  echo "Quick validation completed."
}

run_full() {
  echo "[1/3] Full test suite"
  pytest tests/ -v

  echo "[2/3] Ruff checks"
  ruff check src/ tests/ backend/

  echo "[3/3] Done"
  echo "Full validation passed."
}

if ! command -v pytest >/dev/null 2>&1; then
  echo "ERROR: pytest not found in PATH" >&2
  exit 1
fi

if ! command -v ruff >/dev/null 2>&1; then
  echo "ERROR: ruff not found in PATH" >&2
  exit 1
fi

case "$MODE" in
  quick)
    run_quick
    ;;
  full)
    run_full
    ;;
  *)
    echo "Usage: $0 [quick|full]" >&2
    exit 1
    ;;
esac
