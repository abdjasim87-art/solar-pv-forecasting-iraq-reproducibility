#!/usr/bin/env python3
from __future__ import annotations
import os, json, hashlib, shutil
from pathlib import Path
import numpy as np
import pandas as pd

W=Path(os.environ.get("LSTM_WORKSPACE","/workspace/lstm"))
ZERO_BASE=W/"output"/"loco_transfer_v2"
ZERO_FREEZE=W/"output"/"loco_zero_shot_freeze_v1"
ADAPT_BASE=W/"output"/"loco_fewshot"
A30=W/"output"/"loco_fewshot_A30_freeze"
OUT=W/"output"/"loco_fewshot_stats_freeze_v1"
PROTOCOL=W/"FEWSHOT_PAIRED_STATS_PROTOCOL_LOCK_v1.json"
ANALYSIS_CODE=W/"paired_fewshot_stats_v1.py"
CITIES=["Anbar","Babylon","Baghdad","Basra","Dhi_Qar","Diyala","Karbala",
        "Kirkuk","Maysan","Muthanna","Najaf","Nineveh","Qadisiyyah","Salah_al_Din","Wasit"]
SEEDS=(1,2,3)
BUDGETS=("30d","90d","365d")
STAGES={(24,"30d"):"A40",(24,"90d"):"A50",(24,"365d"):"A60",
        (48,"30d"):"A70",(48,"90d"):"A80",(48,"365d"):"A90"}
EXPECTED={24:{"windows_city":17185,"pooled_windows":257775,"residuals":6186600,"daylight":3103207},
          48:{"windows_city":17161,"pooled_windows":257415,"residuals":12355920,"daylight":6199242}}


def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def np_hash(a)->str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def canon_digest(files):
    payload=json.dumps(files,sort_keys=True,separators=(",",":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def add_file(manifest,p:Path,kind:str,fail):
    if not p.exists():
        fail.append(f"Missing file: {p}")
        return None
    h=sha256(p)
    manifest["files"][str(p)]={"sha256":h,"size_bytes":p.stat().st_size,"kind":kind}
    return h


def loadj(p):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    shutil.rmtree(OUT,ignore_errors=True)
    OUT.mkdir(parents=True)
    fail=[]
    manifest={"version":"1.0","files":{},"stage_locks":{},"pairing":{}}
    pairing=[]
    stage_rows=[]

    if not PROTOCOL.exists():
        fail.append(f"Missing statistical protocol lock {PROTOCOL}")
        protocol={}
    else:
        protocol=loadj(PROTOCOL)
        add_file(manifest,PROTOCOL,"statistical_protocol_lock",fail)
    if ANALYSIS_CODE.exists():
        h=add_file(manifest,ANALYSIS_CODE,"frozen_B10_analysis_code",fail)
        exp=protocol.get("analysis_code_sha256")
        if exp and h!=exp:
            fail.append("B10 analysis code SHA does not match protocol lock")
    else:
        fail.append(f"Missing frozen B10 analysis code {ANALYSIS_CODE}")

    # Re-validate the previously frozen zero-shot experiment and all files it froze.
    zrep=ZERO_FREEZE/"zero_shot_freeze_report_v1.json"
    zman=ZERO_FREEZE/"zero_shot_freeze_manifest_v1.json"
    if zrep.exists():
        hz=add_file(manifest,zrep,"zero_shot_freeze_report",fail)
        if protocol.get("source_locks",{}).get("zero_shot_freeze_report_sha256") not in (None,hz):
            fail.append("zero-shot freeze report SHA changed")
        zr=loadj(zrep)
        for k,v in {"status":"PASS","n_failures":0,"n_npz_frozen":180,"history_cold_metadata_exact_match_all":True,
                    "statistics_not_run":True,"adaptation_not_run":True}.items():
            if zr.get(k)!=v:
                fail.append(f"zero-shot freeze report {k}={zr.get(k)!r}, expected {v!r}")
        expected_digest=protocol.get("source_locks",{}).get("zero_shot_manifest_digest_sha256")
        if expected_digest and zr.get("manifest_digest_sha256")!=expected_digest:
            fail.append("zero-shot manifest canonical digest changed")
    else:
        fail.append(f"Missing {zrep}")

    if zman.exists():
        hm=add_file(manifest,zman,"zero_shot_freeze_manifest",fail)
        if protocol.get("source_locks",{}).get("zero_shot_freeze_manifest_sha256") not in (None,hm):
            fail.append("zero-shot freeze manifest file SHA changed")
        zm=loadj(zman)
        for pstr,expected in zm.get("files",{}).items():
            p=Path(pstr)
            if not p.exists():
                fail.append(f"zero-shot frozen file missing: {p}")
                continue
            got=sha256(p)
            if got!=expected:
                fail.append(f"zero-shot frozen file changed: {p}")
            manifest["files"].setdefault(str(p),{"sha256":got,"size_bytes":p.stat().st_size,"kind":"zero_shot_prior_freeze"})
    else:
        fail.append(f"Missing {zman}")

    # A30 pre-Test adaptation freeze key artifacts.
    a30_files={
        "protocol":A30/"FEWSHOT_PROTOCOL_LOCK_FINAL_v1_1.json",
        "report":A30/"A30_freeze_report_v1.json",
        "manifest":A30/"fewshot_adaptation_freeze_manifest_v1.csv",
        "registry":A30/"adapted_checkpoint_registry_v1.csv",
    }
    for name,p in a30_files.items():
        h=add_file(manifest,p,f"A30_{name}",fail)
        exp=protocol.get("source_locks",{}).get(f"A30_{name}_sha256")
        if exp and h!=exp:
            fail.append(f"A30 {name} SHA changed")
    if a30_files["report"].exists():
        ar=loadj(a30_files["report"])
        for k,v in {"status":"PASS","stage_complete":True,"adapted_checkpoints_frozen":270,
                    "adaptation_test_results_available":False,"adapted_test_authorized":True}.items():
            if ar.get(k)!=v:
                fail.append(f"A30 report {k}={ar.get(k)!r}, expected {v!r}")

    # Six closed adapted Test stages + exact pairing against frozen cold-start zero-shot NPZs.
    adapted_npz=0
    baseline_unique=set()
    pair_true=0
    daylight_counts={}
    for H in (24,48):
        for budget in BUDGETS:
            stage=STAGES[(H,budget)]
            root=ADAPT_BASE/f"H{H}"/"test"/budget
            sl=root/f"{stage}_H{H}_{budget}_stage_lock.json"
            sh=add_file(manifest,sl,"adapted_stage_lock",fail)
            expected_stage_sha=protocol.get("adapted_stage_lock_sha256",{}).get(stage)
            if expected_stage_sha and sh!=expected_stage_sha:
                fail.append(f"{stage} stage-lock SHA changed")
            if sl.exists():
                sd=loadj(sl)
                req={"status":"PASS","stage":stage,"stage_complete":True,"horizon":H,"budget":budget,
                     "cities":15,"adapted_checkpoints_evaluated":45,"test_dataset_opened":True,
                     "test_inference_performed":True,"sample_sd_ddof":1}
                for k,v in req.items():
                    if sd.get(k)!=v:
                        fail.append(f"{stage} {k}={sd.get(k)!r}, expected {v!r}")
                if sd.get("test_sha256")!=protocol.get("test_sha256"):
                    fail.append(f"{stage} Test SHA changed")
                for k in [x for x in sd if x.startswith("same_grid_as_") or x.startswith("same_truth_as_") or x.startswith("same_daylight_as_")]:
                    if sd.get(k) is not True:
                        fail.append(f"{stage} {k} is not true")
                if sd.get("windows_per_city")!=EXPECTED[H]["windows_city"]:
                    fail.append(f"{stage} windows_per_city changed")
                if sd.get("pooled_windows_per_seed")!=EXPECTED[H]["pooled_windows"]:
                    fail.append(f"{stage} pooled_windows_per_seed changed")
                if sd.get("residuals_per_seed")!=EXPECTED[H]["residuals"]:
                    fail.append(f"{stage} residuals_per_seed changed")
                if sd.get("daylight_residuals_per_seed")!=EXPECTED[H]["daylight"]:
                    fail.append(f"{stage} daylight count changed")
                manifest["stage_locks"][stage]={"path":str(sl),"sha256":sh,"horizon":H,"budget":budget}
                stage_rows.append({"stage":stage,"horizon":H,"budget":budget,"path":str(sl),"sha256":sh})

            # Freeze top-level evidence used for reporting/audit.
            top_files=[
                "raw_npz_manifest.csv","adapted_pooled_test_seed_results.csv","adapted_pooled_test_summary_ddof1.csv",
                "adapted_aggregate_energy_by_seed.csv","adapted_energy_by_city.csv","adapted_test_by_city_and_seed.csv",
                "adapted_test_per_horizon_by_city.csv","pretest_frozen_checkpoint_checks.csv",
                "stage60_pairing_audit.csv" if H==24 else "stage80_pairing_audit.csv",
            ]
            for fn in top_files:
                p=root/fn
                h=add_file(manifest,p,"adapted_test_evidence",fail)
                if sl.exists():
                    sd=loadj(sl)
                    expected_key={
                        "raw_npz_manifest.csv":"raw_npz_manifest_sha256",
                        "adapted_pooled_test_seed_results.csv":"pooled_results_sha256",
                        "adapted_pooled_test_summary_ddof1.csv":"summary_sha256",
                        "adapted_aggregate_energy_by_seed.csv":"energy_sha256",
                    }.get(fn)
                    if expected_key and sd.get(expected_key) and h!=sd.get(expected_key):
                        fail.append(f"{stage} {fn} SHA does not match stage lock")
            raw_report=root/"raw_audit"/f"{stage}_raw_audit_report.json"
            add_file(manifest,raw_report,"adapted_raw_audit_report",fail)
            if raw_report.exists():
                rr=loadj(raw_report)
                for k,v in {"status":"PASS","n_failures":0,"raw_metrics_recomputed":True,
                            "raw_energy_recomputed":True,"physical_bounds_checked":True}.items():
                    if rr.get(k)!=v:
                        fail.append(f"{stage} raw audit {k}={rr.get(k)!r}, expected {v!r}")
            for fn in ["raw_npz_inventory.csv","raw_recomputed_pooled_metrics.csv","raw_recomputed_energy.csv"]:
                add_file(manifest,root/"raw_audit"/fn,"adapted_raw_audit_evidence",fail)

            rman=root/"raw_npz_manifest.csv"
            if not rman.exists():
                continue
            rm=pd.read_csv(rman)
            if len(rm)!=45:
                fail.append(f"{stage} raw NPZ manifest has {len(rm)} rows, expected 45")
            keys=set(zip(rm["city"].astype(str),rm["seed"].astype(int)))
            expected_keys={(c,s) for c in CITIES for s in SEEDS}
            if keys!=expected_keys:
                fail.append(f"{stage} raw NPZ city×seed grid differs from 45 expected pairs")

            for city in CITIES:
                bm=ZERO_BASE/f"H{H}"/"cold_start"/"test"/"cities"/city/"test_window_metadata.csv.gz"
                am=root/"cities"/city/"test_window_metadata.csv.gz"
                add_file(manifest,bm,"cold_start_metadata",fail)
                add_file(manifest,am,"adapted_metadata",fail)
                if bm.exists() and am.exists():
                    mb=pd.read_csv(bm); ma=pd.read_csv(am)
                    cols=["start_index","forecast_origin","first_valid_time"]
                    if len(mb)!=EXPECTED[H]["windows_city"] or len(ma)!=EXPECTED[H]["windows_city"]:
                        fail.append(f"{stage} {city} metadata row count mismatch")
                    if not mb[cols].equals(ma[cols]):
                        fail.append(f"{stage} {city} adapted/zero metadata mismatch")

                for seed in SEEDS:
                    row=rm[(rm.city==city)&(rm.seed==seed)]
                    if len(row)!=1:
                        fail.append(f"{stage} {city} seed{seed}: raw manifest row count {len(row)}")
                        continue
                    ap=Path(row.iloc[0]["path"])
                    if not ap.exists():
                        fail.append(f"{stage} adapted NPZ missing: {ap}")
                        continue
                    ah=sha256(ap)
                    if ah!=str(row.iloc[0]["sha256"]):
                        fail.append(f"{stage} {city} seed{seed}: adapted NPZ SHA changed")
                    add_file(manifest,ap,"adapted_test_npz",fail)
                    adapted_npz += 1

                    zp=ZERO_BASE/f"H{H}"/"cold_start"/"test"/"cities"/city/f"seed_{seed}"/"test_predictions.npz"
                    if not zp.exists():
                        fail.append(f"H{H} {city} seed{seed}: cold-start zero-shot NPZ missing")
                        continue
                    add_file(manifest,zp,"cold_start_zero_shot_npz",fail)
                    baseline_unique.add(str(zp))

                    za=np.load(ap,allow_pickle=False); zz=np.load(zp,allow_pickle=False)
                    required=("pred","true","daylight","window_start_idx","series_time_ns")
                    if any(k not in za.files for k in required) or any(k not in zz.files for k in required):
                        fail.append(f"{stage} {city} seed{seed}: missing required NPZ keys")
                        continue
                    if za["pred"].shape!=(EXPECTED[H]["windows_city"],H) or zz["pred"].shape!=(EXPECTED[H]["windows_city"],H):
                        fail.append(f"{stage} {city} seed{seed}: NPZ shape mismatch")
                    checks={
                        "start_equal":np.array_equal(za["window_start_idx"],zz["window_start_idx"]),
                        "series_time_equal":np.array_equal(za["series_time_ns"],zz["series_time_ns"]),
                        "truth_equal":np.array_equal(za["true"],zz["true"]),
                        "daylight_equal":np.array_equal(za["daylight"],zz["daylight"]),
                    }
                    if not all(checks.values()):
                        fail.append(f"{stage} {city} seed{seed}: adapted/zero raw pairing mismatch")
                    if not np.isfinite(za["pred"]).all() or za["pred"].min() < -1e-8 or za["pred"].max() > 1.2000001:
                        fail.append(f"{stage} {city} seed{seed}: adapted prediction physical bounds/finite failure")
                    pairing.append({
                        "stage":stage,"horizon":H,"budget":budget,"city":city,"seed":seed,
                        **checks,"windows":int(za["pred"].shape[0]),
                        "adapted_npz_sha256":ah,"zero_npz_sha256":sha256(zp),
                    })
                    pair_true += int(all(checks.values()))
                    daylight_counts[(H,budget,seed)] = daylight_counts.get((H,budget,seed),0) + int(za["daylight"].sum())

    if adapted_npz!=270:
        fail.append(f"Expected 270 adapted Test NPZs, froze {adapted_npz}")
    if len(baseline_unique)!=90:
        fail.append(f"Expected 90 unique cold-start zero-shot NPZs, froze {len(baseline_unique)}")
    if len(pairing)!=270 or pair_true!=270:
        fail.append(f"Expected 270 exact adapted/zero pairings, got rows={len(pairing)}, exact={pair_true}")
    for (H,budget,seed),n in daylight_counts.items():
        if n!=EXPECTED[H]["daylight"]:
            fail.append(f"H{H} {budget} seed{seed}: daylight residual count {n}, expected {EXPECTED[H]['daylight']}")

    pd.DataFrame(pairing).to_csv(OUT/"fewshot_stats_pairing_audit_v1.csv",index=False)
    pd.DataFrame(stage_rows).to_csv(OUT/"stage_lock_inventory_v1.csv",index=False)
    manifest["manifest_digest_sha256"]=canon_digest(manifest["files"])
    (OUT/"fewshot_stats_freeze_manifest_v1.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    report={
        "status":"PASS" if not fail else "FAIL",
        "stage":"B00",
        "stage_complete":not fail,
        "purpose":"Freeze cold-start zero-shot baseline, all six adapted Test outputs, pairing, and B10 analysis code before any adapted-vs-zero inferential test.",
        "n_failures":len(fail),"failures":fail,
        "adapted_test_npz_frozen":adapted_npz,
        "cold_start_zero_shot_npz_unique_frozen":len(baseline_unique),
        "exact_pairing_rows":pair_true,
        "expected_pairing_rows":270,
        "horizons":[24,48],"budgets":["30d","90d","365d"],"cities":15,"seeds":[1,2,3],
        "pairing_key":["city","forecast_origin","horizon_step"],
        "inferential_statistics_run":False,
        "cross_budget_inferential_tests_run":False,
        "training_performed":False,
        "model_inference_performed":False,
        "raw_test_csv_opened":False,
        "statistics_authorized_after_review":not fail,
        "manifest_digest_sha256":manifest["manifest_digest_sha256"],
    }
    (OUT/"B00_freeze_report_v1.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if not fail else 2)

if __name__=="__main__":
    main()
