#!/usr/bin/env bash
set -u
W=/workspace/lstm
PKG="$W/loco_zero_shot_stats_v1"
cp "$PKG/paired_zero_shot_stats_v1.py" "$W/paired_zero_shot_stats_v1.py"
LOG="$W/loco_zero_shot_paired_stats_v1.log"
ARC="$W/loco_zero_shot_paired_stats_v1_audit.tar.gz"
OUT="$W/output/loco_zero_shot_paired_stats_v1"

set +e
python "$W/paired_zero_shot_stats_v1.py" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e
rm -f "$ARC"
if [ -d "$OUT" ]; then
  tar -czf "$ARC" -C "$W/output" loco_zero_shot_paired_stats_v1
  echo "Created $ARC"
fi
echo "Stats exit status: $STATUS"
exit "$STATUS"
