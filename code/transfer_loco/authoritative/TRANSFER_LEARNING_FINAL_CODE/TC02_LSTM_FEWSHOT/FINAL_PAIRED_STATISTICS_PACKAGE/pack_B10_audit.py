#!/usr/bin/env python3
import os, json, hashlib, shutil, tarfile
from pathlib import Path
W=Path(os.environ.get("LSTM_WORKSPACE","/workspace/lstm"))
ROOT=W/"output"/"loco_fewshot_paired_stats_v1"
ARC=W/"loco_fewshot_stats_B10_paired_audit.tar.gz"
TMP=W/"_B10_stats_pack"
PACK=TMP/"loco_fewshot_stats_B10_paired_audit"
shutil.rmtree(TMP,ignore_errors=True); PACK.mkdir(parents=True)
for p in ROOT.glob('*'):
    if p.is_file(): shutil.copy2(p,PACK/p.name)
for p in [W/"FEWSHOT_PAIRED_STATS_PROTOCOL_LOCK_v1.json", W/"paired_fewshot_stats_v1.py",
          W/"output"/"loco_fewshot_stats_freeze_v1"/"B00_freeze_report_v1.json"]:
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
