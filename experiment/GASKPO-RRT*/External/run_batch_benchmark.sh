#!/usr/bin/env bash
set -euo pipefail

# Run the benchmark from this script's directory so it works from any cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" -u "$SCRIPT_DIR/batch_benchmark.py" "$@"
