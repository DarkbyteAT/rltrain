#!/usr/bin/env bash
# Run any python script under the MLX sidecar venv (Apple Silicon GPU via jax-mlx-plugin).
#
# Usage:
#   ./scripts/run_mlx.sh <python_script> [args...]
#   ./scripts/run_mlx.sh -m rltrain.cli --agent ... --env ...
#
# First run bootstraps .venv-mlx/ with Python 3.13 + jax-mlx-plugin + rltrain (editable).
# Subsequent runs just exec into the sidecar interpreter.
#
# See docs/mlx_setup.md for the install rationale, smoke output, and
# the workload-shape caveat (MLX wins on large matmul / large batch;
# it loses on scan-over-single-step rollouts).

set -euo pipefail

# Locate the worktree root (parent of scripts/)
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
VENV="$ROOT/.venv-mlx"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "[run_mlx] No sidecar at $VENV — bootstrapping." >&2

  # Locate Python 3.13. Prefer uv-managed install, fall back to system.
  if PY313="$(uv python find 3.13 2>/dev/null)"; then
    :
  elif command -v python3.13 >/dev/null 2>&1; then
    PY313="$(command -v python3.13)"
  else
    echo "[run_mlx] No Python 3.13 found. Install via: uv python install 3.13" >&2
    exit 1
  fi

  echo "[run_mlx] Using $PY313 to create venv." >&2
  "$PY313" -m venv "$VENV"
  "$PY/../pip" install --upgrade pip
  "$PY/../pip" install jax-mlx-plugin
  "$PY/../pip" install -e "$ROOT"
  echo "[run_mlx] Bootstrap complete." >&2
fi

exec "$PY" "$@"
