#!/usr/bin/env python3
import argparse, hashlib, json, tarfile
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--archive',required=True); p.add_argument('--exclude-raw',action='store_true'); a=p.parse_args()
root=Path(a.root).resolve(); archive=Path(a.archive).resolve();
if not root.is_dir(): raise SystemExit(f'Missing root: {root}')
files=[]
for x in sorted(root.rglob('*')):
    if not x.is_file(): continue
    if x.name == 'audit_pack_inventory.json': continue
    if x.suffix.lower() in {'.pth','.pt'}: continue
    if a.exclude_raw and '.npz'==x.suffix.lower(): continue
    files.append(x)
def sha(x):
    h=hashlib.sha256();
    with x.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
inv={'root':str(root),'n_files':len(files),'files':[{'relative_path':str(x.relative_to(root)),'bytes':x.stat().st_size,'sha256':sha(x)} for x in files]}
inv_path=root/'audit_pack_inventory.json'; inv_path.write_text(json.dumps(inv,indent=2),encoding='utf-8')
# inventory itself is added but not recursively inventoried
with tarfile.open(archive,'w:gz') as tf:
    for x in files:
        tf.add(x, arcname=f"{root.name}/{x.relative_to(root).as_posix()}")
    tf.add(inv_path,arcname=str(root.name+'/audit_pack_inventory.json'))
print(f'AUDIT PACK PASS: {archive} | evidence_files={len(files)}')
