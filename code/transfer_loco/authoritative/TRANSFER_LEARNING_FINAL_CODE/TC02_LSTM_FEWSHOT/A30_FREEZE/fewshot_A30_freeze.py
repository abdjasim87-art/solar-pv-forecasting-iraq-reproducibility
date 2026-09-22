#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

W = Path("/workspace/lstm")
H24 = W/"output"/"loco_fewshot"/"H24"/"selection_refit"
H48 = W/"output"/"loco_fewshot"/"H48"/"selection_refit"
OUT = W/"output"/"loco_fewshot_A30_freeze"

PROTOCOL = W/"FEWSHOT_PROTOCOL_LOCK_FINAL_v1_1.json"
EXPECTED_PROTOCOL_SHA = "6558a8034efacdbae3b3b4a6002bf7108d9605e89f25bbe661380f1ad07dd351"
EXPECTED_A10_STAGE_LOCK_SHA = "62ae754d49fad5561c5759b9900b5c07c2c31ba545123d0a0531a419efeaf13a"
EXPECTED_A20_STAGE_LOCK_SHA = "16d8103a6be11d40127b578cb09fcf071dfafcfa029d8eae4e02a774c2964204"
EXPECTED_A10_CHECKPOINT_MANIFEST_SHA = "d0d1e945355248d54d5c18cec8f805bbc80c1983764db5af655c7ffdafa10945"
EXPECTED_A20_CHECKPOINT_MANIFEST_SHA = "b8f553833e3fdaf94fa5792ff81c10732ee8d2435ea8247990db816f31ac2916"

CITIES = [
    "Anbar","Babylon","Baghdad","Basra","Dhi_Qar","Diyala","Karbala",
    "Kirkuk","Maysan","Muthanna","Najaf","Nineveh","Qadisiyyah",
    "Salah_al_Din","Wasit",
]
BUDGETS = ["30d","90d","365d"]
SEEDS = [1,2,3]

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def load_json(p: Path) -> Dict[str,Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def dump_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj,indent=2),encoding="utf-8")

def category(rel: str) -> str:
    n=Path(rel).name
    if n=="adapted_checkpoint.pt": return "adapted_checkpoint"
    if n=="selection_best_checkpoint.pt": return "selection_checkpoint"
    if n=="run_lock.json": return "run_lock"
    if n=="selection_history.csv": return "selection_history"
    if n=="refit_history.csv": return "refit_history"
    if n.endswith("_stage_lock.json"): return "stage_lock"
    if n.endswith("_checkpoint_manifest.csv"): return "checkpoint_manifest"
    if n.endswith("_all_runs.csv"): return "all_runs"
    if "selection_summary" in n: return "selection_summary"
    return "other"

def canonical_digest(rows: List[Dict[str,Any]]) -> str:
    payload=json.dumps(
        sorted(rows,key=lambda r:(r["horizon"],r["relative_path"])),
        sort_keys=True,separators=(",",":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def verify_stage(root: Path, H: int, expected_stage_sha: str, expected_manifest_sha: str):
    failures=[]
    stage_name=f"A{10 if H==24 else 20}_H{H}_stage_lock.json"
    manifest_name=f"A{10 if H==24 else 20}_H{H}_checkpoint_manifest.csv"
    stage=root/stage_name
    mani=root/manifest_name
    if not stage.is_file():
        failures.append(f"Missing {stage}")
        return failures
    if sha256(stage)!=expected_stage_sha:
        failures.append(f"H{H} stage-lock SHA changed")
    d=load_json(stage)
    required={
        "status":"PASS",
        "stage_complete":True,
        "horizon":H,
        "n_cities":15,
        "selection_runs":135,
        "adapted_checkpoints":135,
        "lookback":24,
        "feature_count":34,
        "target_history_predictors":0,
        "frozen_backbone":True,
        "source_scalers_frozen":True,
        "source_correction_frozen":True,
        "test_dataset_opened":False,
        "test_inference_performed":False,
        "sample_sd_ddof":1,
    }
    for k,v in required.items():
        if d.get(k)!=v:
            failures.append(f"H{H} stage lock {k}={d.get(k)!r}, expected {v!r}")
    if d.get("trainable_head_parameters") != (42712 if H==24 else 44272):
        failures.append(f"H{H} trainable head parameter count mismatch")
    if d.get("prerequisites",{}).get("final_protocol_lock_sha256")!=EXPECTED_PROTOCOL_SHA:
        failures.append(f"H{H} protocol SHA in stage lock changed")
    if not mani.is_file():
        failures.append(f"Missing {mani}")
    elif sha256(mani)!=expected_manifest_sha:
        failures.append(f"H{H} checkpoint-manifest SHA changed")
    return failures

def main():
    shutil.rmtree(OUT,ignore_errors=True)
    OUT.mkdir(parents=True)

    failures=[]
    if not PROTOCOL.is_file() or sha256(PROTOCOL)!=EXPECTED_PROTOCOL_SHA:
        failures.append("Final few-shot protocol lock missing or changed")

    failures += verify_stage(H24,24,EXPECTED_A10_STAGE_LOCK_SHA,EXPECTED_A10_CHECKPOINT_MANIFEST_SHA)
    failures += verify_stage(H48,48,EXPECTED_A20_STAGE_LOCK_SHA,EXPECTED_A20_CHECKPOINT_MANIFEST_SHA)

    manifest_rows=[]
    adapted_rows=[]
    run_lock_rows=[]

    expected_grid={(H,c,b,s) for H in (24,48) for c in CITIES for b in BUDGETS for s in SEEDS}
    seen_grid=set()
    adapted_shas=[]

    for H,root in ((24,H24),(48,H48)):
        if not root.is_dir():
            failures.append(f"Missing selection/refit directory: {root}")
            continue

        # Freeze every file in the completed stage, including both selection and adapted checkpoints.
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel=str(p.relative_to(root))
            manifest_rows.append({
                "horizon":H,
                "relative_path":rel,
                "size_bytes":p.stat().st_size,
                "sha256":sha256(p),
                "category":category(rel),
            })

        # Independently verify every final run lock and final adapted checkpoint.
        for city in CITIES:
            for budget in BUDGETS:
                for seed in SEEDS:
                    rd=root/"folds"/city/budget/f"seed_{seed}"
                    lk=rd/"run_lock.json"
                    ck=rd/"adapted_checkpoint.pt"
                    sel=rd/"selection_best_checkpoint.pt"
                    sh=rd/"selection_history.csv"
                    rh=rd/"refit_history.csv"
                    for p in (lk,ck,sel,sh,rh):
                        if not p.is_file():
                            failures.append(f"Missing H{H} run artifact: {p}")
                    if not lk.is_file():
                        continue
                    d=load_json(lk)
                    seen_grid.add((H,city,budget,seed))
                    checks={
                        "run_complete":d.get("run_complete") is True,
                        "stage":d.get("stage")==("A10" if H==24 else "A20"),
                        "horizon":d.get("horizon")==H,
                        "city":d.get("city")==city,
                        "budget":d.get("budget")==budget,
                        "seed":d.get("seed")==seed,
                        "test_dataset_opened":d.get("test_dataset_opened") is False,
                        "test_inference_performed":d.get("test_inference_performed") is False,
                        "no_target_history":d.get("adaptation_uses_target_history_predictors") is False,
                        "feature_count":d.get("feature_count")==34,
                        "source_correction_frozen":d.get("source_correction_frozen") is True,
                        "source_scalers_frozen":d.get("source_scalers_frozen") is True,
                        "frozen_backbone_identity_verified":d.get("frozen_backbone_identity_verified") is True,
                    }
                    for k,ok in checks.items():
                        if not ok:
                            failures.append(f"H{H} {city} {budget} seed{seed}: {k} check failed")
                    if ck.is_file():
                        csha=sha256(ck)
                        adapted_shas.append(csha)
                        if csha!=d.get("adapted_checkpoint_sha256"):
                            failures.append(f"H{H} {city} {budget} seed{seed}: adapted checkpoint SHA differs from run lock")
                    if sel.is_file() and sha256(sel)!=d.get("selection_checkpoint_sha256"):
                        failures.append(f"H{H} {city} {budget} seed{seed}: selection checkpoint SHA differs")
                    if sh.is_file() and sha256(sh)!=d.get("selection_history_sha256"):
                        failures.append(f"H{H} {city} {budget} seed{seed}: selection history SHA differs")
                    if rh.is_file() and sha256(rh)!=d.get("refit_history_sha256"):
                        failures.append(f"H{H} {city} {budget} seed{seed}: refit history SHA differs")

                    adapted_rows.append({
                        "horizon":H,"city":city,"budget":budget,"seed":seed,
                        "adapted_checkpoint":str(ck),
                        "adapted_checkpoint_sha256":d.get("adapted_checkpoint_sha256"),
                        "adapted_checkpoint_state_digest":d.get("adapted_checkpoint_state_digest"),
                        "source_checkpoint_sha256":d.get("source_checkpoint_sha256"),
                        "source_correction_sha256":d.get("source_correction_sha256"),
                        "x_scaler_sha256":d.get("x_scaler_sha256"),
                        "y_scaler_sha256":d.get("y_scaler_sha256"),
                        "best_epoch":d.get("best_epoch"),
                        "run_lock_sha256":sha256(lk),
                    })
                    run_lock_rows.append({
                        "horizon":H,"city":city,"budget":budget,"seed":seed,
                        "run_lock_sha256":sha256(lk),
                    })

    if seen_grid!=expected_grid:
        failures.append(f"Run grid mismatch: seen={len(seen_grid)} expected={len(expected_grid)}")
    if len(adapted_rows)!=270:
        failures.append(f"Expected 270 adapted runs, got {len(adapted_rows)}")
    if len(adapted_shas)!=270:
        failures.append(f"Expected 270 adapted checkpoint hashes, got {len(adapted_shas)}")
    if len(set(adapted_shas))!=270:
        failures.append(f"Adapted checkpoint SHA uniqueness failed: {len(set(adapted_shas))}/270")

    # Exact expected complete-stage file count: each horizon has
    # 135*(adapted ckpt + selection ckpt + run lock + two histories) + 4 top-level = 679.
    expected_files=1358
    if len(manifest_rows)!=expected_files:
        failures.append(f"Freeze file count {len(manifest_rows)} != expected {expected_files}")

    counts={}
    for r in manifest_rows:
        counts[r["category"]]=counts.get(r["category"],0)+1
    expected_categories={
        "adapted_checkpoint":270,
        "selection_checkpoint":270,
        "run_lock":270,
        "selection_history":270,
        "refit_history":270,
        "stage_lock":2,
        "checkpoint_manifest":2,
        "all_runs":2,
        "selection_summary":2,
    }
    for k,v in expected_categories.items():
        if counts.get(k,0)!=v:
            failures.append(f"Category count {k}={counts.get(k,0)} expected {v}")

    # Write immutable freeze manifest and compact adapted-checkpoint registry.
    freeze_csv=OUT/"fewshot_adaptation_freeze_manifest_v1.csv"
    with freeze_csv.open("w",newline="",encoding="utf-8") as f:
        fields=["horizon","relative_path","size_bytes","sha256","category"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(
            sorted(manifest_rows,key=lambda r:(r["horizon"],r["relative_path"]))
        )

    adapted_csv=OUT/"adapted_checkpoint_registry_v1.csv"
    with adapted_csv.open("w",newline="",encoding="utf-8") as f:
        fields=[
            "horizon","city","budget","seed","adapted_checkpoint",
            "adapted_checkpoint_sha256","adapted_checkpoint_state_digest",
            "source_checkpoint_sha256","source_correction_sha256",
            "x_scaler_sha256","y_scaler_sha256","best_epoch","run_lock_sha256"
        ]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(
            sorted(adapted_rows,key=lambda r:(r["horizon"],r["city"],r["budget"],r["seed"]))
        )

    digest=canonical_digest(manifest_rows)
    report={
        "status":"PASS" if not failures else "FAIL",
        "stage":"A30",
        "stage_complete":not failures,
        "purpose":"Freeze all completed H24/H48 few-shot selection/refit artifacts before any adapted Test inference.",
        "training_performed":False,
        "test_dataset_opened":False,
        "test_inference_performed":False,
        "adaptation_test_results_available":False,
        "horizons":[24,48],
        "cities":15,
        "budgets":["30d","90d","365d"],
        "seeds":[1,2,3],
        "adapted_checkpoints_frozen":len(adapted_rows),
        "selection_checkpoints_frozen":counts.get("selection_checkpoint",0),
        "run_locks_frozen":counts.get("run_lock",0),
        "all_stage_files_frozen":len(manifest_rows),
        "expected_stage_files":expected_files,
        "adapted_checkpoint_sha_unique":len(set(adapted_shas)),
        "protocol_lock_sha256":sha256(PROTOCOL) if PROTOCOL.is_file() else None,
        "A10_stage_lock_sha256":sha256(H24/"A10_H24_stage_lock.json") if (H24/"A10_H24_stage_lock.json").is_file() else None,
        "A20_stage_lock_sha256":sha256(H48/"A20_H48_stage_lock.json") if (H48/"A20_H48_stage_lock.json").is_file() else None,
        "freeze_manifest_sha256":sha256(freeze_csv),
        "adapted_registry_sha256":sha256(adapted_csv),
        "freeze_canonical_digest_sha256":digest,
        "n_failures":len(failures),
        "failures":failures,
        "adapted_test_authorized":not failures,
    }
    dump_json(OUT/"A30_freeze_report_v1.json",report)

    # Copy frozen protocol and the two already-closed stage locks into A30 evidence.
    shutil.copy2(PROTOCOL,OUT/"FEWSHOT_PROTOCOL_LOCK_FINAL_v1_1.json")
    shutil.copy2(H24/"A10_H24_stage_lock.json",OUT/"A10_H24_stage_lock_snapshot.json")
    shutil.copy2(H48/"A20_H48_stage_lock.json",OUT/"A20_H48_stage_lock_snapshot.json")

    print(json.dumps(report,indent=2))
    if failures:
        raise SystemExit(2)
    print("A30 FEW-SHOT ADAPTATION FREEZE PASS | 270 adapted checkpoints frozen | Test file was not read.")

if __name__=="__main__":
    main()
