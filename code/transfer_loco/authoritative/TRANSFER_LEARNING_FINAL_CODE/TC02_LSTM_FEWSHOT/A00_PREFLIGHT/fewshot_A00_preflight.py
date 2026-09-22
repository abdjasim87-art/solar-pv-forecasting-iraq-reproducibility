#!/usr/bin/env python3
from __future__ import annotations
import json, hashlib, os, re, shutil, subprocess, sys
from pathlib import Path
from collections import Counter, defaultdict

W = Path("/workspace/lstm")
OUT = W/"output"/"loco_fewshot_A00_preflight"
LOCO = W/"output"/"loco_transfer_v2"
FREEZE = W/"output"/"loco_zero_shot_freeze_v1"
STATS = W/"output"/"loco_zero_shot_paired_stats_v1"

EXPECTED_FREEZE_DIGEST = "6112890c1e2c3fc6cb0f1e3b6b44c67c71d4955ab1e84d167715624c5f843692"
EXPECTED_VALIDATION_SHA = "86f95e8581b864e2cc6d67640bbcd802fdf7176716b5119aa721d1a64369ca5a"
EXPECTED_TEST_SHA = "b0a3f938359eb1d5da615d0db5f34333b764217d2ce41454f5ff220b6ac9a222"

CITIES = ["Anbar","Babylon","Baghdad","Basra","Dhi_Qar","Diyala","Karbala",
          "Kirkuk","Maysan","Muthanna","Najaf","Nineveh","Qadisiyyah",
          "Salah_al_Din","Wasit"]

def sha256(p: Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def safe_json(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def add_file(report, p, label=None):
    if p and p.exists() and p.is_file():
        report["files"][label or str(p)]={"path":str(p),"sha256":sha256(p),"size_bytes":p.stat().st_size}

def discover_checkpoints():
    roots=[
        LOCO/"H24"/"cold_start",
        LOCO/"H48"/"cold_start",
    ]
    cps=[]
    for r in roots:
        if r.exists():
            for p in r.rglob("*"):
                if p.is_file() and p.suffix.lower() in (".pt",".pth",".ckpt"):
                    # Exclude any adapted artifacts if package is rerun later.
                    if "fewshot" not in str(p).lower() and "adapt" not in str(p).lower():
                        cps.append(p)
    return sorted(set(cps))

def checkpoint_state_dict_inventory(cps):
    rows=[]
    torch_error=None
    try:
        import torch
    except Exception as e:
        return [], f"torch import failed: {e}"
    for p in cps:
        try:
            obj=torch.load(p,map_location="cpu")
            if isinstance(obj,dict) and "state_dict" in obj and isinstance(obj["state_dict"],dict):
                sd=obj["state_dict"]
            elif isinstance(obj,dict) and "model_state_dict" in obj and isinstance(obj["model_state_dict"],dict):
                sd=obj["model_state_dict"]
            elif isinstance(obj,dict):
                # Heuristic: direct state_dict if tensor-like values dominate.
                tensor_items={k:v for k,v in obj.items() if hasattr(v,"shape")}
                sd=tensor_items if tensor_items else {}
            else:
                sd={}
            keys=[]
            for k,v in sd.items():
                shape=list(v.shape) if hasattr(v,"shape") else None
                n=int(v.numel()) if hasattr(v,"numel") else None
                keys.append({"name":k,"shape":shape,"numel":n})
            rows.append({"checkpoint":str(p),"sha256":sha256(p),"state_keys":keys})
        except Exception as e:
            rows.append({"checkpoint":str(p),"sha256":sha256(p),"error":repr(e),"state_keys":[]})
    return rows, torch_error

def scan_json_for_hash_and_paths(root, target_hash):
    hits=[]
    if not root.exists(): return hits
    for p in root.rglob("*.json"):
        d=safe_json(p)
        if d is None: continue
        txt=json.dumps(d,sort_keys=True)
        if target_hash in txt:
            hits.append({"json":str(p),"sha256":sha256(p),"content":d})
    return hits

def extract_candidate_paths(obj):
    out=set()
    def rec(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if isinstance(v,str):
                    kl=k.lower()
                    if any(t in kl for t in ("path","file","csv","parquet","data")) or v.startswith("/workspace/"):
                        out.add(v)
                rec(v)
        elif isinstance(x,list):
            for v in x: rec(v)
    rec(obj)
    return sorted(out)

def source_inventory(report):
    candidates=[
        W/"loco_transfer_v2"/"solar_lstm_loco_transfer_protocol_v2.py",
        W/"loco_transfer_v2"/"solar_lstm_strict_protocol_reference.py",
        W/"solar_lstm_loco_transfer_protocol_v2.py",
        W/"solar_lstm_strict_protocol_reference.py",
    ]
    src=[]
    for p in candidates:
        if p.exists():
            src.append(p)
            add_file(report,p)
    # Also collect shell runners/config/protocol markdown without reading Test data.
    pkg=W/"loco_transfer_v2"
    if pkg.exists():
        for p in pkg.iterdir():
            if p.is_file() and p.suffix.lower() in (".py",".sh",".md",".json"):
                if p not in src:
                    add_file(report,p)
    return src

def text_parameter_hints(srcs):
    hints={}
    pats=("head","fc","linear","lstm","projection","output","Sequential","nn.Linear")
    for p in srcs:
        try:
            lines=p.read_text(errors="replace").splitlines()
        except Exception:
            continue
        chosen=[]
        for i,line in enumerate(lines,1):
            if any(t.lower() in line.lower() for t in pats):
                chosen.append({"line":i,"text":line[:500]})
        hints[str(p)]=chosen[:400]
    return hints

def main():
    shutil.rmtree(OUT,ignore_errors=True)
    OUT.mkdir(parents=True)
    failures=[]
    warnings=[]
    report={
        "status":"PENDING",
        "stage":"A00",
        "training_performed":False,
        "test_dataset_opened":False,
        "files":{},
        "expected_source_checkpoints":90,
        "expected_validation_sha256":EXPECTED_VALIDATION_SHA,
        "forbidden_test_sha256":EXPECTED_TEST_SHA,
    }

    # Protocol lock copied by runner.
    lock=W/"FEWSHOT_PROTOCOL_LOCK_PRETEST_v1.json"
    if not lock.exists():
        failures.append("Missing FEWSHOT_PROTOCOL_LOCK_PRETEST_v1.json")
    else:
        add_file(report,lock,"fewshot_protocol_lock")

    # Verify zero-shot freeze and stats remain PASS.
    fr=FREEZE/"zero_shot_freeze_report_v1.json"
    fm=FREEZE/"zero_shot_freeze_manifest_v1.json"
    sr=STATS/"paired_stats_report_v1.json"
    for p in (fr,fm,sr):
        if not p.exists(): failures.append(f"Missing prerequisite {p}")
        else: add_file(report,p)
    if fr.exists():
        d=safe_json(fr) or {}
        if d.get("status")!="PASS": failures.append("Zero-shot freeze report not PASS")
        if d.get("manifest_digest_sha256")!=EXPECTED_FREEZE_DIGEST:
            failures.append("Zero-shot freeze manifest digest differs from frozen value")
    if sr.exists():
        d=safe_json(sr) or {}
        if d.get("status")!="PASS": failures.append("Paired stats report not PASS")
        if d.get("changed_after_freeze") not in (0,None): failures.append("Frozen files changed before few-shot preflight")

    # Locate source checkpoints.
    cps=discover_checkpoints()
    report["source_checkpoints_found"]=len(cps)
    if len(cps)!=90:
        warnings.append(f"Expected exactly 90 cold-start source checkpoints; discovered {len(cps)}. Review path logic before training.")
    inv,terr=checkpoint_state_dict_inventory(cps)
    (OUT/"checkpoint_state_dict_inventory.json").write_text(json.dumps(inv,indent=2),encoding="utf-8")
    if terr: warnings.append(terr)

    # Aggregate state-key patterns to bind the forecast head after review.
    sig_counter=Counter()
    key_counter=Counter()
    for row in inv:
        names=tuple(x["name"] for x in row.get("state_keys",[]))
        if names: sig_counter[names]+=1
        for k in names: key_counter[k]+=1
    sig_summary=[
        {"count":count,"n_keys":len(sig),"keys":list(sig)}
        for sig,count in sig_counter.most_common()
    ]
    (OUT/"checkpoint_state_key_signatures.json").write_text(json.dumps(sig_summary,indent=2),encoding="utf-8")
    (OUT/"checkpoint_state_key_frequency.json").write_text(json.dumps(dict(key_counter),indent=2),encoding="utf-8")

    # Source-code inventory + parameter hints.
    srcs=source_inventory(report)
    hints=text_parameter_hints(srcs)
    (OUT/"source_parameter_hints.json").write_text(json.dumps(hints,indent=2),encoding="utf-8")

    # Find JSON records containing the known Validation SHA. This does not read Test data.
    val_hits=scan_json_for_hash_and_paths(W/"output"/"loco_transfer_v2",EXPECTED_VALIDATION_SHA)
    compact=[]
    candidate_paths=set()
    for h in val_hits:
        candidate_paths.update(extract_candidate_paths(h["content"]))
        compact.append({"json":h["json"],"sha256":h["sha256"],"candidate_paths":extract_candidate_paths(h["content"])})
    (OUT/"validation_sha_hits.json").write_text(json.dumps(compact,indent=2),encoding="utf-8")
    report["validation_sha_json_hits"]=len(compact)

    existing_candidates=[]
    for s in sorted(candidate_paths):
        p=Path(s)
        if p.exists() and p.is_file():
            # DO NOT hash files whose recorded/content hash is the forbidden Test SHA by brute force here.
            # We only inventory obvious validation-named candidates.
            lname=p.name.lower()
            if any(t in lname for t in ("val","valid","2022","2023")) and "test" not in lname:
                try:
                    existing_candidates.append({"path":str(p),"sha256":sha256(p),"size_bytes":p.stat().st_size})
                except Exception as e:
                    existing_candidates.append({"path":str(p),"error":repr(e)})
    (OUT/"existing_validation_path_candidates.json").write_text(json.dumps(existing_candidates,indent=2),encoding="utf-8")
    exact=[x for x in existing_candidates if x.get("sha256")==EXPECTED_VALIDATION_SHA]
    report["exact_validation_file_matches"]=len(exact)
    if not exact:
        warnings.append("Exact Validation data file path was not bound automatically. Training remains forbidden until it is identified without opening Test.")
    elif len(exact)>1:
        warnings.append("Multiple files match the frozen Validation SHA; bind one canonical path before training.")

    # Scaler/correction artifact inventory.
    artifacts=[]
    for H in (24,48):
        r=LOCO/f"H{H}"/"cold_start"
        if not r.exists(): continue
        for p in r.rglob("*"):
            if not p.is_file(): continue
            n=p.name.lower()
            if any(t in n for t in ("scaler","correction","bias")) and p.suffix.lower() in (".json",".pkl",".joblib",".npz",".npy",".csv"):
                artifacts.append({"horizon":H,"path":str(p),"sha256":sha256(p),"size_bytes":p.stat().st_size})
    (OUT/"source_scaler_correction_inventory.json").write_text(json.dumps(artifacts,indent=2),encoding="utf-8")
    report["scaler_correction_artifacts_found"]=len(artifacts)

    # Frozen budget calendar is emitted for independent review.
    lockdata=safe_json(lock) if lock.exists() else {}
    (OUT/"budget_calendar_from_lock.json").write_text(json.dumps(lockdata.get("budgets_utc",{}),indent=2),encoding="utf-8")

    report["failures"]=failures
    report["warnings"]=warnings
    report["n_failures"]=len(failures)
    report["n_warnings"]=len(warnings)
    report["head_parameter_binding_complete"]=False
    report["adaptation_training_authorized"]=False
    report["status"]="PASS_PRELIMINARY" if not failures else "FAIL"
    (OUT/"fewshot_A00_preflight_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    sys.exit(0 if not failures else 2)

if __name__=="__main__":
    main()
