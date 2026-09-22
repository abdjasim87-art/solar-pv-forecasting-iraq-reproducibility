#!/usr/bin/env python3
from __future__ import annotations
import json, hashlib, importlib.util, sys, shutil
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import joblib

W=Path("/workspace/lstm")
OUT=W/"output"/"loco_fewshot_A00b_binding"
VALID=W/"data2"/"valid.csv"
BASE=W/"loco_transfer_v2"/"solar_lstm_pv_per_kwp_v3.py"
LOCO=W/"loco_transfer_v2"/"solar_lstm_loco_transfer_protocol_v2.py"
A00=W/"output"/"loco_fewshot_A00_preflight"
LOCK=W/"FEWSHOT_PROTOCOL_LOCK_FINAL_v1_1.json"

EXPECTED_VALID_SHA="86f95e8581b864e2cc6d67640bbcd802fdf7176716b5119aa721d1a64369ca5a"
EXPECTED_BASE_SHA="41f7892cbdd296f6ea4d6887a5dd92669cd3a1a42ef0fb4695c29e7d79369438"
EXPECTED_LOCO_SHA="aa3e4f32f161b5d8e2f02fe65ce56c74c11ec30ecbd50a2410e77c3955636a17"
CITIES=["Anbar","Babylon","Baghdad","Basra","Dhi_Qar","Diyala","Karbala",
        "Kirkuk","Maysan","Muthanna","Najaf","Nineveh","Qadisiyyah",
        "Salah_al_Din","Wasit"]
HISTORY=[
"solar_lag1","solar_lag2","solar_lag3","solar_lag6","solar_lag12","solar_lag24","solar_lag48","solar_lag168",
"solar_roll3_mean","solar_roll6_mean","solar_roll12_mean","solar_roll24_mean",
"solar_roll3_std","solar_roll6_std","solar_roll12_std",
"solar_delta1","solar_delta2","solar_delta3","solar_daylight_lag1"]

EXPECTED_WINDOWS={
24:{
"30d":{"selection_train":553,"selection_validation":121,"full_refit":697},
"90d":{"selection_train":1705,"selection_validation":409,"full_refit":2137},
"365d":{"selection_train":6985,"selection_validation":1729,"full_refit":8737}},
48:{
"30d":{"selection_train":529,"selection_validation":97,"full_refit":673},
"90d":{"selection_train":1681,"selection_validation":385,"full_refit":2113},
"365d":{"selection_train":6961,"selection_validation":1705,"full_refit":8713}}}
INTERVALS={
"30d":{
"selection_train":("2023-12-02T00:00:00Z","2023-12-25T23:00:00Z"),
"selection_validation":("2023-12-26T00:00:00Z","2023-12-31T23:00:00Z"),
"full_refit":("2023-12-02T00:00:00Z","2023-12-31T23:00:00Z")},
"90d":{
"selection_train":("2023-10-03T00:00:00Z","2023-12-13T23:00:00Z"),
"selection_validation":("2023-12-14T00:00:00Z","2023-12-31T23:00:00Z"),
"full_refit":("2023-10-03T00:00:00Z","2023-12-31T23:00:00Z")},
"365d":{
"selection_train":("2023-01-01T00:00:00Z","2023-10-19T23:00:00Z"),
"selection_validation":("2023-10-20T00:00:00Z","2023-12-31T23:00:00Z"),
"full_refit":("2023-01-01T00:00:00Z","2023-12-31T23:00:00Z")}}

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def loadmod(name,p):
    spec=importlib.util.spec_from_file_location(name,str(p))
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def window_count(times, start, end, H):
    s=pd.Timestamp(start); e=pd.Timestamp(end)
    # Count forecast targets t..t+H-1 fully inside interval.
    end_first=e-pd.Timedelta(hours=H-1)
    return int(((times>=s)&(times<=end_first)).sum())

def selected_method_from_json(d):
    methods={"none","global_bias","daylight_bias","horizon_bias","horizon_daylight_bias"}
    preferred=[]
    def rec(x,path=""):
        if isinstance(x,dict):
            for k,v in x.items():
                p=f"{path}.{k}" if path else k
                if isinstance(v,str) and v in methods and any(t in k.lower() for t in ("selected","method","correction")):
                    preferred.append((p,v))
                rec(v,p)
        elif isinstance(x,list):
            for i,v in enumerate(x): rec(v,f"{path}[{i}]")
    rec(d)
    vals=[v for _,v in preferred]
    if "horizon_daylight_bias" in vals: return "horizon_daylight_bias", preferred
    if len(set(vals))==1 and vals: return vals[0], preferred
    return None, preferred

def main():
    shutil.rmtree(OUT,ignore_errors=True); OUT.mkdir(parents=True)
    fail=[]; warn=[]
    report={"stage":"A00b","audit_implementation_version":"1.2-joblib-fix","training_performed":False,"test_dataset_opened":False}

    # Prerequisite A00.
    a00p=A00/"fewshot_A00_preflight_report.json"
    if not a00p.exists(): fail.append("Missing A00 report")
    else:
        a=json.loads(a00p.read_text())
        if a.get("status")!="PASS_PRELIMINARY": fail.append("A00 not PASS_PRELIMINARY")
        if a.get("training_performed") is not False: fail.append("A00 training flag not false")
        if a.get("test_dataset_opened") is not False: fail.append("A00 test-open flag not false")
        if a.get("source_checkpoints_found")!=90: fail.append("A00 source checkpoint count !=90")

    for p,exp,label in [(VALID,EXPECTED_VALID_SHA,"validation"),(BASE,EXPECTED_BASE_SHA,"base code"),(LOCO,EXPECTED_LOCO_SHA,"LOCO code")]:
        if not p.exists(): fail.append(f"Missing {label}: {p}")
        else:
            got=sha(p); report[f"{label.replace(' ','_')}_sha256"]=got
            if got!=exp: fail.append(f"{label} SHA mismatch")

    # Bind validation file only. Test file is never opened/read/hashed.
    if VALID.exists():
        raw=pd.read_csv(VALID)
        report["validation_raw_rows"]=len(raw)
        city_col="city_canonical" if "city_canonical" in raw.columns else ("city" if "city" in raw.columns else None)
        if city_col is None: fail.append("No city column in validation")
        if "time" not in raw.columns: fail.append("No time column in validation")
        if city_col and "time" in raw.columns:
            raw["time"]=pd.to_datetime(raw["time"],errors="coerce",utc=True)
            raw[city_col]=raw[city_col].astype(str).str.strip()
            if raw["time"].isna().any(): fail.append("NaT in validation time")
            cities=sorted(raw[city_col].unique().tolist())
            report["cities"]=cities
            if cities!=sorted(CITIES): fail.append(f"City list mismatch: {cities}")
            if len(raw)!=262800: fail.append(f"Validation row count {len(raw)} !=262800")
            grid=[]
            for c in CITIES:
                g=raw[raw[city_col]==c].sort_values("time")
                t=g["time"].reset_index(drop=True)
                dif=t.diff().dropna()
                row={"city":c,"rows":len(g),"rows_2023":int((t.dt.year==2023).sum()),
                     "min_time":str(t.min()),"max_time":str(t.max()),
                     "duplicates":int(t.duplicated().sum()),
                     "nonhourly_gaps":int((dif!=pd.Timedelta(hours=1)).sum())}
                grid.append(row)
                if row["rows"]!=17520 or row["rows_2023"]!=8760 or row["duplicates"] or row["nonhourly_gaps"]:
                    fail.append(f"Validation hourly grid issue for {c}: {row}")
            pd.DataFrame(grid).to_csv(OUT/"validation_city_grid.csv",index=False)

            counts=[]
            for H in (24,48):
                for budget,parts in INTERVALS.items():
                    for part,(s,e) in parts.items():
                        vals=[]
                        for c in CITIES:
                            t=raw.loc[raw[city_col]==c,"time"].sort_values().reset_index(drop=True)
                            n=window_count(t,s,e,H)
                            vals.append(n)
                            counts.append({"horizon":H,"budget":budget,"part":part,"city":c,
                                           "window_count":n,"expected":EXPECTED_WINDOWS[H][budget][part]})
                        if len(set(vals))!=1 or vals[0]!=EXPECTED_WINDOWS[H][budget][part]:
                            fail.append(f"Window count mismatch H{H} {budget} {part}: {sorted(set(vals))}")
            pd.DataFrame(counts).to_csv(OUT/"budget_window_counts_by_city.csv",index=False)

    # Bind exact cold-start feature list through authoritative base code.
    if BASE.exists() and VALID.exists():
        base=loadmod("solar_base_a00b",BASE)
        v,target=base.load_split(str(VALID),"valid","auto")
        features,continuous=base.build_feature_columns(v,"basic",False)
        report["resolved_target_col"]=target
        report["coldstart_feature_count"]=len(features)
        report["coldstart_features"]=features
        report["coldstart_history_overlap"]=sorted(set(features)&set(HISTORY))
        if len(features)!=34: fail.append(f"Cold-start feature count {len(features)} !=34")
        if report["coldstart_history_overlap"]: fail.append("Target-history feature leaked into cold-start feature list")
        (OUT/"coldstart_feature_binding.json").write_text(json.dumps({
            "feature_count":len(features),"features":features,"continuous_features":continuous,
            "excluded_history_features":HISTORY,"history_overlap":report["coldstart_history_overlap"]
        },indent=2))

    # Bind exact head keys from A00 inventory.
    invp=A00/"checkpoint_state_dict_inventory.json"
    if not invp.exists(): fail.append("Missing A00 checkpoint inventory")
    else:
        inv=json.loads(invp.read_text())
        if len(inv)!=90: fail.append(f"Checkpoint inventory rows {len(inv)} !=90")
        expected_names=[
        "lstm.weight_ih_l0","lstm.weight_hh_l0","lstm.bias_ih_l0","lstm.bias_hh_l0",
        "lstm.weight_ih_l1","lstm.weight_hh_l1","lstm.bias_ih_l1","lstm.bias_hh_l1",
        "lstm.weight_ih_l2","lstm.weight_hh_l2","lstm.bias_ih_l2","lstm.bias_hh_l2",
        "head.0.weight","head.0.bias","head.3.weight","head.3.bias","head.6.weight","head.6.bias"]
        sigs=set()
        head_counts={24:set(),48:set()}
        for r in inv:
            names=tuple(x["name"] for x in r.get("state_keys",[]))
            sigs.add(names)
            if list(names)!=expected_names: fail.append(f"Unexpected state key signature: {r.get('checkpoint')}")
            H=48 if "/H48/" in r.get("checkpoint","") else 24
            head_counts[H].add(sum(int(x["numel"]) for x in r.get("state_keys",[]) if x["name"].startswith("head.")))
        if len(sigs)!=1: fail.append(f"Checkpoint state signatures={len(sigs)}")
        if head_counts[24]!={42712}: fail.append(f"H24 head param counts={head_counts[24]}")
        if head_counts[48]!={44272}: fail.append(f"H48 head param counts={head_counts[48]}")
        binding={
            "all_90_same_signature":len(sigs)==1,
            "trainable_keys":["head.0.weight","head.0.bias","head.3.weight","head.3.bias","head.6.weight","head.6.bias"],
            "frozen_prefix":"lstm.",
            "trainable_params_H24":sorted(head_counts[24]),
            "trainable_params_H48":sorted(head_counts[48]),
            "backbone_mode_during_adaptation":"eval",
            "head_mode_during_adaptation":"train"
        }
        (OUT/"head_parameter_binding_final.json").write_text(json.dumps(binding,indent=2))

    # Source scalers/corrections.
    rows=[]
    corr_methods=[]
    for H in (24,48):
        foldroot=W/"output"/"loco_transfer_v2"/f"H{H}"/"cold_start"/"selection"/"folds"
        for c in CITIES:
            fr=foldroot/c
            xs=fr/"x_scaler.pkl"; ys=fr/"y_scaler.pkl"
            for kind,p in (("x_scaler",xs),("y_scaler",ys)):
                if not p.exists(): fail.append(f"Missing {kind} H{H} {c}")
                else:
                    obj=joblib.load(p)
                    nf=int(getattr(obj,"n_features_in_",1))
                    if kind=="x_scaler" and nf!=34: fail.append(f"x_scaler n_features H{H} {c}={nf}")
                    rows.append({"horizon":H,"city":c,"artifact":kind,"path":str(p),"sha256":sha(p),"n_features_in":nf})
            for seed in (1,2,3):
                cp=fr/"L24"/f"seed_{seed}"/"correction.json"
                if not cp.exists():
                    fail.append(f"Missing correction H{H} {c} seed{seed}")
                    continue
                d=json.loads(cp.read_text())
                method,evidence=selected_method_from_json(d)
                corr_methods.append(method)
                rows.append({"horizon":H,"city":c,"seed":seed,"artifact":"correction",
                             "path":str(cp),"sha256":sha(cp),"selected_method":method,
                             "method_evidence":json.dumps(evidence)})
                if method!="horizon_daylight_bias":
                    fail.append(f"Correction binding H{H} {c} seed{seed}: {method}")
    pd.DataFrame(rows).to_csv(OUT/"source_artifact_binding.csv",index=False)
    report["source_scalers_expected"]=60
    report["source_corrections_expected"]=90
    report["correction_method_counts"]=dict(Counter(corr_methods))

    if LOCK.exists():
        report["final_protocol_lock_sha256"]=sha(LOCK)
    else:
        fail.append("Missing final protocol lock")

    report["n_failures"]=len(fail); report["failures"]=fail; report["warnings"]=warn
    report["validation_path_bound"]=VALID.exists() and report.get("validation_sha256")==EXPECTED_VALID_SHA
    report["head_parameter_binding_complete"]=not any("state key" in x.lower() or "head param" in x.lower() for x in fail)
    report["budget_window_binding_complete"]=not any("window count" in x.lower() or "hourly grid" in x.lower() for x in fail)
    report["adaptation_training_authorized"]=len(fail)==0
    report["status"]="PASS" if not fail else "FAIL"
    (OUT/"fewshot_A00b_binding_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if not fail else 2)

if __name__=="__main__":
    main()
