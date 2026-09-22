#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
fail=[]
for name,meta in manifest.items():
    p=root/name
    if not p.is_file(): fail.append(f'missing:{name}'); continue
    if p.stat().st_size!=int(meta['bytes']): fail.append(f'size:{name}')
    if sha(p)!=meta['sha256']: fail.append(f'sha:{name}')
if fail:
    raise SystemExit('PACKAGE VERIFY FAIL: '+', '.join(fail))
print(f'PACKAGE VERIFY PASS: {len(manifest)} frozen files')
