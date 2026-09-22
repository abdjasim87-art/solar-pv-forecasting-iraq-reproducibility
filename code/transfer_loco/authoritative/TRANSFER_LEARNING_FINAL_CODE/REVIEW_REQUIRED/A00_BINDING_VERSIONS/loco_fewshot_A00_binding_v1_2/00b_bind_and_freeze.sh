#!/usr/bin/env bash
set -u
W=/workspace/lstm
PKG="$W/loco_fewshot_A00_binding_v1_2"
OUT="$W/output/loco_fewshot_A00b_binding"
LOG="$W/loco_fewshot_A00b_binding.log"
ARC="$W/loco_fewshot_A00b_binding_audit.tar.gz"

cp "$PKG/FEWSHOT_PROTOCOL_LOCK_FINAL_v1_1.json" "$W/FEWSHOT_PROTOCOL_LOCK_FINAL_v1_1.json"
cp "$PKG/fewshot_A00b_binding.py" "$W/fewshot_A00b_binding.py"

set +e
python "$W/fewshot_A00b_binding.py" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

rm -f "$ARC"
if [ -d "$OUT" ]; then
  tar -czf "$ARC" -C "$W/output" loco_fewshot_A00b_binding
  echo "Created $ARC"
fi
echo "A00b exit status: $STATUS"
exit "$STATUS"
