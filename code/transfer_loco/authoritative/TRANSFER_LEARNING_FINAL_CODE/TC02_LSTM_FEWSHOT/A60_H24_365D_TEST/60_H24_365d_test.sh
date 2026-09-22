#!/usr/bin/env bash
set -u
W=/workspace/lstm
PKG="$W/loco_fewshot_A60_H24_365d_test_v1"
LOG="$W/loco_fewshot_A60_H24_365d_test.log"
ARC="$W/loco_fewshot_A60_H24_365d_audit.tar.gz"

PKG_LOCK="$(sha256sum "$PKG/FEWSHOT_PROTOCOL_LOCK_FINAL_v1_1.json"|awk '{print $1}')"
LIVE_LOCK="$(sha256sum "$W/FEWSHOT_PROTOCOL_LOCK_FINAL_v1_1.json"|awk '{print $1}')"
if [[ "$PKG_LOCK" != "$LIVE_LOCK" ]]; then echo "ERROR protocol lock mismatch"|tee "$LOG"; exit 2; fi

set +e
python "$PKG/fewshot_A60_H24_365d_test.py" 2>&1 | tee "$LOG"
S1=${PIPESTATUS[0]}
if [[ "$S1" -eq 0 ]]; then
  python "$PKG/audit_A60_H24_365d_raw.py" 2>&1 | tee -a "$LOG"
  S2=${PIPESTATUS[0]}
else
  S2=99
fi
set -e
python "$PKG/pack_A60_audit.py" || true
echo "A60 test status=$S1 raw_audit_status=$S2"
if [[ -f "$ARC" ]]; then sha256sum "$ARC"; fi
if [[ "$S1" -ne 0 || "$S2" -ne 0 ]]; then exit 2; fi
exit 0
