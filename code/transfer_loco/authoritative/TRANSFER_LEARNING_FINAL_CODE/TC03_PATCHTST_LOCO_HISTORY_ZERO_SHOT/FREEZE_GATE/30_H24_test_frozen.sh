#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-/workspace/lstm}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$WORKSPACE/patchtst_loco_history_zero_shot_v1"
python "$HERE/verify_selection_freeze.py" --workspace "$WORKSPACE"
exec "$PKG/30_H24_test.sh" "$WORKSPACE"
