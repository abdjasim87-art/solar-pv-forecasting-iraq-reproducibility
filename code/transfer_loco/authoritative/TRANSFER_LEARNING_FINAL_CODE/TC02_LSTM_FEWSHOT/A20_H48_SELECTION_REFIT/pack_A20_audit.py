#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, json, tarfile

W=Path("/workspace/lstm")
SRC=W/"output"/"loco_fewshot"/"H48"/"selection_refit"
ARC=W/"loco_fewshot_A20_H48_audit.tar.gz"
STAGE=W/"_A20_audit_pack"
if STAGE.exists():
    import shutil; shutil.rmtree(STAGE)
STAGE.mkdir(parents=True)
PACK=STAGE/"loco_fewshot_A20_H48_audit"
PACK.mkdir()

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

# Full inventory, including checkpoints by path/size/hash, but do not copy heavy checkpoints.
rows=[]
if SRC.exists():
    for p in sorted(SRC.rglob("*")):
        if p.is_file():
            rows.append({
                "relative_path":str(p.relative_to(SRC)),
                "size_bytes":p.stat().st_size,
                "sha256":sha(p),
                "copied_to_audit":p.suffix.lower() not in {".pt",".pth",".ckpt"},
            })
with (PACK/"full_file_inventory.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["relative_path","size_bytes","sha256","copied_to_audit"])
    w.writeheader(); w.writerows(rows)

# Copy all text/tabular evidence, preserving tree.
for r in rows:
    if not r["copied_to_audit"]:
        continue
    src=SRC/r["relative_path"]
    dst=PACK/"files"/r["relative_path"]
    dst.parent.mkdir(parents=True,exist_ok=True)
    import shutil; shutil.copy2(src,dst)

report={
    "stage":"A20",
    "source_dir":str(SRC),
    "files_total":len(rows),
    "heavy_checkpoints_excluded":sum(not r["copied_to_audit"] for r in rows),
    "text_tabular_files_included":sum(r["copied_to_audit"] for r in rows),
    "stage_lock_present":(SRC/"A20_H48_stage_lock.json").is_file(),
    "test_predictions_expected":0,
}
(PACK/"audit_pack_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")

if ARC.exists(): ARC.unlink()
with tarfile.open(ARC,"w:gz") as tf:
    tf.add(PACK,arcname=PACK.name)
print(ARC)
