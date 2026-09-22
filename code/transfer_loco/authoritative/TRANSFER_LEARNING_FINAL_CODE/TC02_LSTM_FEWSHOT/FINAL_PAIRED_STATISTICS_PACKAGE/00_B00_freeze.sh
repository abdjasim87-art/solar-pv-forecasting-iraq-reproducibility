#!/usr/bin/env bash
set -u
W=/workspace/lstm
PKG="$W/loco_fewshot_paired_stats_v1"
LOG="$W/loco_fewshot_stats_B00_freeze.log"
ARC="$W/loco_fewshot_stats_B00_freeze_audit.tar.gz"

# Freeze exact protocol + exact B10 analysis code in workspace before B00 runs.
cp "$PKG/FEWSHOT_PAIRED_STATS_PROTOCOL_LOCK_v1.json" "$W/FEWSHOT_PAIRED_STATS_PROTOCOL_LOCK_v1.json"
cp "$PKG/paired_fewshot_stats_v1.py" "$W/paired_fewshot_stats_v1.py"

set +e
python "$PKG/freeze_fewshot_stats_B00_v1.py" 2>&1 | tee "$LOG"
S=${PIPESTATUS[0]}
set -e
python "$PKG/pack_B00_audit.py" || true
echo "B00 exit status: $S"
if [[ -f "$ARC" ]]; then sha256sum "$ARC"; fi
exit "$S"
