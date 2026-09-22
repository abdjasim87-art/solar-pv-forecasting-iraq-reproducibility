#!/usr/bin/env bash
set -euo pipefail
W="${LSTM_WORKSPACE:-/workspace/lstm}"
PKG="$W/patchtst_lstm_loco_paired_stats_v1"
OUT="$W/output/patchtst_lstm_loco_stats_freeze_v1"
LOG="$W/patchtst_lstm_loco_stats_B00_freeze.log"
PACK="$W/patchtst_lstm_loco_stats_B00_freeze_audit.tar.gz"
rm -rf "$OUT"
mkdir -p "$OUT"
python "$PKG/freeze_B00_v1.py" 2>&1 | tee "$LOG"
python "$PKG/pack_audit.py" "$OUT" "$PACK" | tee -a "$LOG"
echo "PASS: B00 frozen. STOP FOR REVIEW. DO NOT RUN B10 YET." | tee -a "$LOG"
echo "Upload: $LOG" | tee -a "$LOG"
echo "Upload: $PACK" | tee -a "$LOG"
