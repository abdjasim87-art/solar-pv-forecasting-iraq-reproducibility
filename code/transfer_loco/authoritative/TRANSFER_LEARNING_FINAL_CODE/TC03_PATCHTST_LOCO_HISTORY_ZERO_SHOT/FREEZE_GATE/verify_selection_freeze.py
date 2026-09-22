#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="/workspace/lstm")
    args = ap.parse_args()
    ws = Path(args.workspace).resolve()
    manifest_path = ws / "output" / "patchtst_loco_v1" / "selection_freeze_v1" / "freeze_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Freeze manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise RuntimeError("Freeze manifest status is not PASS")
    if manifest.get("test_file_read") is not False:
        raise RuntimeError("Freeze manifest does not certify pre-Test freeze")
    changed = []
    for rec in manifest["files"]:
        p = Path(rec["path"])
        if not p.is_absolute():
            p = ws / p
        if not p.is_file():
            changed.append((rec["path"], "MISSING"))
            continue
        if p.stat().st_size != int(rec["bytes"]):
            changed.append((rec["path"], "SIZE"))
            continue
        h = sha256(p)
        if h != rec["sha256"]:
            changed.append((rec["path"], "SHA256"))
    if changed:
        preview = "\n".join(f"{p}: {why}" for p,why in changed[:20])
        raise RuntimeError(f"Selection changed after freeze ({len(changed)} files):\n{preview}")
    print("PATCHTST LOCO FREEZE VERIFY PASS")
    print(f"Rehashed files : {len(manifest['files'])}")
    print("Changed files  : 0")

if __name__ == "__main__":
    main()
