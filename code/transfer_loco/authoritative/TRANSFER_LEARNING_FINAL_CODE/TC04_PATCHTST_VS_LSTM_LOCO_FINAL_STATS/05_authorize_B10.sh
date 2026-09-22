#!/usr/bin/env bash
set -euo pipefail
W="${LSTM_WORKSPACE:-/workspace/lstm}"
R="$W/output/patchtst_lstm_loco_stats_freeze_v1/B00_freeze_report_v1.json"
python - "$R" <<'PY'
import json,sys
p=sys.argv[1]
d=json.load(open(p))
if d.get("status")!="PASS" or not d.get("stage_complete",False):
    raise SystemExit("B00 is not PASS/complete")
d["statistics_authorized_after_review"]=True
open(p,"w").write(json.dumps(d,indent=2)+"\n")
print("B10 statistics authorization recorded.")
PY
