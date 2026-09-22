#!/usr/bin/env bash
set -euo pipefail
W="${LSTM_WORKSPACE:-/workspace/lstm}"
PKG="$W/patchtst_lstm_loco_paired_stats_v1"
OUT="$W/output/patchtst_lstm_loco_paired_stats_v1"
LOG="$W/patchtst_lstm_loco_stats_B10_paired.log"
PACK="$W/patchtst_lstm_loco_stats_B10_paired_audit.tar.gz"
python "$PKG/paired_stats_v1.py" 2>&1 | tee "$LOG"
python "$PKG/pack_audit.py" "$OUT" "$PACK" | tee -a "$LOG"
echo "PASS: B10 paired statistics complete." | tee -a "$LOG"
echo "Upload: $LOG" | tee -a "$LOG"
echo "Upload: $PACK" | tee -a "$LOG"
