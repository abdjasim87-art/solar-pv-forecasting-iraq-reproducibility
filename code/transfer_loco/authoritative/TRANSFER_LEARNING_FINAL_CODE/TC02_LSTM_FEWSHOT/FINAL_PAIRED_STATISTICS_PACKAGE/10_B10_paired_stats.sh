#!/usr/bin/env bash
set -u
W=/workspace/lstm
PKG="$W/loco_fewshot_paired_stats_v1"
LOG="$W/loco_fewshot_stats_B10_paired.log"
ARC="$W/loco_fewshot_stats_B10_paired_audit.tar.gz"

# Do not replace the code frozen by B00; verify it remains identical to package and protocol.
ROOT_CODE="$W/paired_fewshot_stats_v1.py"
PKG_CODE="$PKG/paired_fewshot_stats_v1.py"
if [[ ! -f "$ROOT_CODE" ]]; then echo "ERROR: frozen B10 code missing; run/review B00 first" | tee "$LOG"; exit 2; fi
if [[ "$(sha256sum "$ROOT_CODE"|awk '{print $1}')" != "$(sha256sum "$PKG_CODE"|awk '{print $1}')" ]]; then
  echo "ERROR: frozen B10 analysis code differs from package" | tee "$LOG"; exit 2
fi
set +e
python "$ROOT_CODE" 2>&1 | tee "$LOG"
S=${PIPESTATUS[0]}
set -e
python "$PKG/pack_B10_audit.py" || true
echo "B10 exit status: $S"
if [[ -f "$ARC" ]]; then sha256sum "$ARC"; fi
exit "$S"
