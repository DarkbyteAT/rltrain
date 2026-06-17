#!/usr/bin/env bash
# Run the offline plasticity probe under the MLX sidecar venv.
#
# NB: per docs/mlx_setup.md, MLX is currently SLOWER than CPU for this
# specific workload shape (scan over outer steps with B=128). This shim
# exists so the comparison is a single command flip, not because MLX is
# the default recommendation for this sweep.
#
# Usage:
#   ./examples/offline_plasticity_probe/run_offline_probe_mlx.sh [args...]

set -euo pipefail
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
exec "$ROOT/scripts/run_mlx.sh" "$ROOT/examples/offline_plasticity_probe/run_offline_probe.py" "$@"
