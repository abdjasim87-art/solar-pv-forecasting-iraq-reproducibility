#!/usr/bin/env python3
from pathlib import Path
import csv,hashlib,json,shutil,tarfile
W=Path("/workspace/lstm"); ROOT=W/"output"/"loco_fewshot"/"H24"/"test"/"365d"
ARC=W/"loco_fewshot_A60_H24_365d_audit.tar.gz"; TMP=W/"_A60_pack"; PACK=TMP/"loco_fewshot_A60_H24_365d_audit"
if TMP.exists(): shutil.rmtree(TMP)
PACK.mkdir(parents=True)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()
rows=[]
for p in sorted(ROOT.rglob("*")):
 if p.is_file():
  rel=p.relative_to(ROOT); heavy=p.suffix.lower()==".npz"
  rows.append({"relative_path":str(rel),"size_bytes":p.stat().st_size,"sha256":sha(p),"copied":not heavy})
  if not heavy:
   d=PACK/"files"/rel; d.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,d)
with (PACK/"full_file_inventory.csv").open("w",newline="",encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=["relative_path","size_bytes","sha256","copied"]);w.writeheader();w.writerows(rows)
rep={"stage":"A60","files_total":len(rows),"npz_excluded":sum(not r["copied"] for r in rows),"non_npz_copied":sum(r["copied"] for r in rows),"raw_npz_bytes_excluded_from_archive":True}
(PACK/"audit_pack_report.json").write_text(json.dumps(rep,indent=2))
if ARC.exists(): ARC.unlink()
with tarfile.open(ARC,"w:gz") as tf: tf.add(PACK,arcname=PACK.name)
print(ARC);print("sha256",sha(ARC))
