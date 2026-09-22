#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-/workspace/lstm}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$WORKSPACE/patchtst_loco_v1_25_selection_freeze.log"
PACK="$WORKSPACE/patchtst_loco_v1_25_selection_freeze_audit.tar.gz"

python "$HERE/freeze_selection.py" --workspace "$WORKSPACE" 2>&1 | tee "$LOG"
python "$HERE/verify_selection_freeze.py" --workspace "$WORKSPACE" | tee -a "$LOG"

FREEZE_DIR="$WORKSPACE/output/patchtst_loco_v1/selection_freeze_v1"
tar -czf "$PACK" -C "$FREEZE_DIR" freeze_manifest.json freeze_report.json

echo "PASS: PatchTST LOCO H24/H48 selection freeze complete. STOP FOR REVIEW." | tee -a "$LOG"
echo "Upload: $LOG" | tee -a "$LOG"
echo "Upload: $PACK" | tee -a "$LOG"
