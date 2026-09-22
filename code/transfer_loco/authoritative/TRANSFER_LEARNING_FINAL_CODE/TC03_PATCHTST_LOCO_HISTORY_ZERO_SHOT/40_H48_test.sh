#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-/workspace/lstm}"
PKG="$WORKSPACE/patchtst_loco_history_zero_shot_v1"
ROOT="$WORKSPACE/output/patchtst_loco_v1"
OUT="$ROOT/H48/test"
LOG="$WORKSPACE/patchtst_loco_v1_40_H48_test.log"
PACK="$WORKSPACE/patchtst_loco_v1_40_H48_test_audit.tar.gz"
RAW="$WORKSPACE/patchtst_loco_v1_40_H48_raw_predictions.tar.gz"
cd "$WORKSPACE"; python "$PKG/verify_package.py"
for f in "$ROOT/H24/selection/stage_lock.json" "$ROOT/H48/selection/stage_lock.json"; do test -s "$f" || { echo "ERROR all H24/H48 selections must be frozen before Test: $f"; exit 1; }; done
python "$PKG/solar_patchtst_loco_protocol_v1.py" evaluate-zero-shot --selection-root "$ROOT" --test-data "$WORKSPACE/data2/test.csv" --horizon 48 --out-dir "$OUT" 2>&1 | tee "$LOG"
test -s "$OUT/test_audit.json" || { echo "ERROR H48 Test audit missing"; exit 1; }
python "$PKG/pack_audit.py" --root "$OUT" --archive "$PACK" --exclude-raw
tar -czf "$RAW" -C "$OUT" raw_predictions
echo "PASS: H48 PatchTST LOCO Test complete. STOP FOR REVIEW."
echo "Upload: $LOG"; echo "Upload: $PACK"; echo "Upload: $RAW"
