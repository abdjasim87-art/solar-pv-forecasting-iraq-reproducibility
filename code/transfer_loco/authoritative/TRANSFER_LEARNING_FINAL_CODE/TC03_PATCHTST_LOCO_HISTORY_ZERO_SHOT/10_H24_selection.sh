#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-/workspace/lstm}"
PKG="$WORKSPACE/patchtst_loco_history_zero_shot_v1"
PREF="$WORKSPACE/output/patchtst_loco_v1/preflight/preflight_lock.json"
OUT="$WORKSPACE/output/patchtst_loco_v1/H24/selection"
LOG="$WORKSPACE/patchtst_loco_v1_10_H24_selection.log"
PACK="$WORKSPACE/patchtst_loco_v1_10_H24_selection_audit.tar.gz"
cd "$WORKSPACE"; python "$PKG/verify_package.py"; test -s "$PREF" || { echo "ERROR run/review preflight first"; exit 1; }
python "$PKG/solar_patchtst_loco_protocol_v1.py" select-source --preflight-lock "$PREF" --horizon 24 --out-dir "$OUT" --resume 2>&1 | tee "$LOG"
test -s "$OUT/stage_lock.json" || { echo "ERROR H24 stage lock missing"; exit 1; }
python "$PKG/pack_audit.py" --root "$OUT" --archive "$PACK" --exclude-raw
echo "PASS: H24 source selection complete. Test was not read. STOP FOR REVIEW."
echo "Upload: $LOG"; echo "Upload: $PACK"
