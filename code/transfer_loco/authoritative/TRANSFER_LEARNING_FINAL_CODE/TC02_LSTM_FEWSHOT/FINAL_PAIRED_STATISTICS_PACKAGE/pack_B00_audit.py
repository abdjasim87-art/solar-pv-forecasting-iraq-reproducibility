#!/usr/bin/env python3
import os, json, hashlib, shutil, tarfile
from pathlib import Path
W=Path(os.environ.get("LSTM_WORKSPACE","/workspace/lstm"))
ROOT=W/"output"/"loco_fewshot_stats_freeze_v1"
ARC=W/"loco_fewshot_stats_B00_freeze_audit.tar.gz"
TMP=W/"_B00_stats_pack"
PACK=TMP/"loco_fewshot_stats_B00_freeze_audit"
shutil.rmtree(TMP,ignore_errors=True); PACK.mkdir(parents=True)
for name in ["B00_freeze_report_v1.json","fewshot_stats_freeze_manifest_v1.json","fewshot_stats_pairing_audit_v1.csv","stage_lock_inventory_v1.csv"]:
    p=ROOT/name
    if p.exists(): shutil.copy2(p,PACK/name)
for p in [W/"FEWSHOT_PAIRED_STATS_PROTOCOL_LOCK_v1.json", W/"paired_fewshot_stats_v1.py"]:
    if p.exists(): shutil.copy2(p,PACK/p.name)
# Copy the six small stage locks and source freeze reports as evidence, never NPZ bytes.
stages=[(24,"30d","A40"),(24,"90d","A50"),(24,"365d","A60"),(48,"30d","A70"),(48,"90d","A80"),(48,"365d","A90")]
for H,b,s in stages:
    p=W/"output"/"loco_fewshot"/f"H{H}"/"test"/b/f"{s}_H{H}_{b}_stage_lock.json"
    if p.exists(): shutil.copy2(p,PACK/p.name)
for p in [W/"output"/"loco_zero_shot_freeze_v1"/"zero_shot_freeze_report_v1.json",
          W/"output"/"loco_fewshot_A30_freeze"/"A30_freeze_report_v1.json"]:
    if p.exists(): shutil.copy2(p,PACK/p.name)
inv=[]
def sha(p):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
for p in sorted(PACK.rglob('*')):
    if p.is_file(): inv.append({"file":p.relative_to(PACK).as_posix(),"size_bytes":p.stat().st_size,"sha256":sha(p)})
(PACK/"audit_pack_inventory.json").write_text(json.dumps(inv,indent=2),encoding='utf-8')
if ARC.exists(): ARC.unlink()
with tarfile.open(ARC,'w:gz') as tf: tf.add(PACK,arcname=PACK.name)
print(ARC)
print(sha(ARC))
