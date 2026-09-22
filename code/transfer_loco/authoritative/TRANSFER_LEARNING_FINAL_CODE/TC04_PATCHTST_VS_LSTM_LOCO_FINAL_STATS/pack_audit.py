#!/usr/bin/env python3
from pathlib import Path
import tarfile, hashlib, json, sys
root=Path(sys.argv[1]).resolve(); out=Path(sys.argv[2]).resolve()
files=sorted(p for p in root.rglob("*") if p.is_file())
inv=[]
for p in files:
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    inv.append({"relative_path":str(p.relative_to(root)),"bytes":p.stat().st_size,"sha256":h})
(root/"audit_pack_inventory.json").write_text(json.dumps({"n_files":len(inv),"files":inv},indent=2)+"\n")
files=sorted(p for p in root.rglob("*") if p.is_file())
with tarfile.open(out,"w:gz") as tf:
    for p in files: tf.add(p,arcname=str(root.name/p.relative_to(root)))
print(f"AUDIT PACK PASS: {out}")
