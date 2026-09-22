#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, json, shutil, tarfile

W=Path("/workspace/lstm")
FREEZE=W/"output"/"loco_fewshot_A30_freeze"
A24=W/"output"/"loco_fewshot"/"H24"/"selection_refit"
A48=W/"output"/"loco_fewshot"/"H48"/"selection_refit"
ARC=W/"loco_fewshot_A30_freeze_audit.tar.gz"
TMP=W/"_A30_freeze_audit_pack"
PACK=TMP/"loco_fewshot_A30_freeze_audit"

if TMP.exists(): shutil.rmtree(TMP)
PACK.mkdir(parents=True)

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

# Copy all A30 outputs.
for p in FREEZE.iterdir():
    if p.is_file():
        shutil.copy2(p,PACK/p.name)

# Copy all non-checkpoint evidence from A10/A20, preserving horizon tree.
copied=0
for H,root in ((24,A24),(48,A48)):
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() in {".pt",".pth",".ckpt"}:
            continue
        rel=p.relative_to(root)
        dst=PACK/"stage_evidence"/f"H{H}"/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dst)
        copied+=1

report={
    "stage":"A30",
    "freeze_output_files":sum(1 for p in FREEZE.iterdir() if p.is_file()),
    "non_checkpoint_stage_evidence_files_copied":copied,
    "checkpoint_bytes_in_audit_archive":False,
    "test_predictions_in_audit_archive":False,
}
(PACK/"audit_pack_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")

if ARC.exists(): ARC.unlink()
with tarfile.open(ARC,"w:gz") as tf:
    tf.add(PACK,arcname=PACK.name)
print(ARC)
print("sha256",sha(ARC))
