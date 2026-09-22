#!/usr/bin/env bash
set -u
W=/workspace/lstm
PKG="$W/loco_fewshot_A30_freeze_v1"
LOG="$W/loco_fewshot_A30_freeze.log"
ARC="$W/loco_fewshot_A30_freeze_audit.tar.gz"

PKG_LOCK_SHA="$(sha256sum "$PKG/FEWSHOT_PROTOCOL_LOCK_FINAL_v1_1.json" | awk '{print $1}')"
LIVE_LOCK_SHA="$(sha256sum "$W/FEWSHOT_PROTOCOL_LOCK_FINAL_v1_1.json" | awk '{print $1}')"
if [[ "$PKG_LOCK_SHA" != "$LIVE_LOCK_SHA" ]]; then
  echo "ERROR: package/live few-shot protocol lock differs" | tee "$LOG"
  exit 2
fi

set +e
python "$PKG/fewshot_A30_freeze.py" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

python "$PKG/pack_A30_freeze_audit.py" || true
echo "A30 exit status: $STATUS"
if [[ -f "$ARC" ]]; then
  sha256sum "$ARC"
fi
exit "$STATUS"
