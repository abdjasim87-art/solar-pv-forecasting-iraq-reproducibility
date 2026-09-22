#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-/workspace/lstm}"
PKG="$WORKSPACE/patchtst_loco_history_zero_shot_v1"
REPO="$WORKSPACE/vendor/PatchTST_official"
PIN="204c21efe0b39603ad6e2ca640ef5896646ab1a9"
OUT="$WORKSPACE/output/patchtst_loco_v1/preflight"
LOG="$WORKSPACE/patchtst_loco_v1_00_preflight.log"
PACK="$WORKSPACE/patchtst_loco_v1_00_preflight_audit.tar.gz"
mkdir -p "$WORKSPACE/vendor" "$OUT"
cd "$WORKSPACE"
python "$PKG/verify_package.py"
for f in "$WORKSPACE/data2/train.csv" "$WORKSPACE/data2/valid.csv"; do test -s "$f" || { echo "ERROR missing $f"; exit 1; }; done
if [[ ! -d "$REPO/.git" ]]; then git clone https://github.com/yuqinie98/PatchTST.git "$REPO"; fi
git -C "$REPO" fetch --all --tags
git -C "$REPO" checkout --detach "$PIN"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$PIN" ]] || { echo "ERROR PatchTST commit mismatch"; exit 1; }
python -m py_compile "$PKG/solar_patchtst_loco_protocol_v1.py" "$PKG/solar_patchtst_original_protocol_reference.py" "$PKG/solar_lstm_pv_per_kwp_v3.py"
python "$PKG/solar_patchtst_loco_protocol_v1.py" preflight --out-dir "$OUT" 2>&1 | tee "$LOG"
python "$PKG/pack_audit.py" --root "$OUT" --archive "$PACK" --exclude-raw
echo "PASS: PatchTST LOCO preflight complete. STOP HERE FOR REVIEW."
echo "Upload: $LOG"
echo "Upload: $PACK"
