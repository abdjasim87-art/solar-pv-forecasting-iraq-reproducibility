#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, importlib.util, json, math, sys, shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

W=Path("/workspace/lstm")
BASE_CODE=W/"loco_transfer_v2"/"solar_lstm_pv_per_kwp_v3.py"
TEST_PATH=W/"data2"/"test.csv"
A30=W/"output"/"loco_fewshot_A30_freeze"
ADAPT_ROOT=W/"output"/"loco_fewshot"/"H48"/"selection_refit"
SOURCE_ROOT=W/"output"/"loco_transfer_v2"/"H48"/"cold_start"/"selection"
ZERO_ROOT=W/"output"/"loco_transfer_v2"/"H48"/"cold_start"/"test"
OUT=W/"output"/"loco_fewshot"/"H48"/"test"/"365d"

EXPECTED_BASE_SHA="41f7892cbdd296f6ea4d6887a5dd92669cd3a1a42ef0fb4695c29e7d79369438"
EXPECTED_TEST_SHA="b0a3f938359eb1d5da615d0db5f34333b764217d2ce41454f5ff220b6ac9a222"
EXPECTED_PROTOCOL_SHA="6558a8034efacdbae3b3b4a6002bf7108d9605e89f25bbe661380f1ad07dd351"
EXPECTED_A30_REPORT_SHA="facf216fc54e4a0e412101438982c7bd726636052b6b814e0cd728babccd0c31"
EXPECTED_FREEZE_MANIFEST_SHA="487e343cb78a6781c50f7adf56da533466b0f19b2f02a71aa1cb39f0b869016c"
EXPECTED_ADAPTED_REGISTRY_SHA="5208b17399cf974b6faaced9325e451da3699806e2129341c4fef9dc4f39b62f"
EXPECTED_A60_STAGE_LOCK_SHA="008bee242615a10b9cbbb172bf10f2aa4a99069af6b67d66ea0d2b16d3636f4e"
A60_STAGE_LOCK=W/"output"/"loco_fewshot"/"H24"/"test"/"365d"/"A60_H24_365d_stage_lock.json"
EXPECTED_A70_STAGE_LOCK_SHA="94886411c1786f796ff313b0041f9762d056bdbc893a959a4da2aab3be28d1b2"
A70_STAGE_LOCK=W/"output"/"loco_fewshot"/"H48"/"test"/"30d"/"A70_H48_30d_stage_lock.json"
EXPECTED_A80_STAGE_LOCK_SHA="78de99432b7b37ed22ec872105588ac4f7b63319a14520f6f14ad4d0620a4794"
A80_STAGE_LOCK=W/"output"/"loco_fewshot"/"H48"/"test"/"90d"/"A80_H48_90d_stage_lock.json"

H=48
L=24
ALIGN_TRIM=168
MIN_START=168
BUDGET="365d"
REF_KWP=3370.0
MAX_PER_KWP=1.20
BATCH=512
EXPECTED_RAW_ROWS_PER_CITY=17544
EXPECTED_ALIGNED_ROWS_PER_CITY=17376
EXPECTED_WINDOWS_PER_CITY=17161
EXPECTED_WINDOWS_POOLED=257415
EXPECTED_RESIDUALS=12355920
EXPECTED_DAYLIGHT=6199242
EXPECTED_UNIQUE_HOURS_PER_CITY=17208
EXPECTED_UNIQUE_CITY_HOURS=258120
EXPECTED_TRUE_ENERGY_MWH=171314.7648391358
EXPECTED_FIRST_VALID=pd.Timestamp("2024-01-15T00:00:00Z")
EXPECTED_LAST_VALID=pd.Timestamp("2025-12-31T23:00:00Z")
CITIES=[
"Anbar","Babylon","Baghdad","Basra","Dhi_Qar","Diyala","Karbala",
"Kirkuk","Maysan","Muthanna","Najaf","Nineveh","Qadisiyyah","Salah_al_Din","Wasit"]
SEEDS=[1,2,3]

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def load_json(p:Path): return json.loads(p.read_text(encoding="utf-8"))
def dump_json(p:Path,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2),encoding="utf-8")

def import_file(name:str,path:Path):
    spec=importlib.util.spec_from_file_location(name,str(path))
    if spec is None or spec.loader is None: raise ImportError(path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    try: spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name,None); raise
    return mod

def metric_dict(pred,true,day):
    p=np.asarray(pred,np.float64).ravel(); y=np.asarray(true,np.float64).ravel()
    d=np.asarray(day).astype(bool).ravel(); e=p-y
    sse=float(np.sum(e*e)); sst=float(np.sum((y-y.mean())**2))
    out={
      "per_kwp_mae":float(np.mean(np.abs(e))),
      "per_kwp_rmse":float(np.sqrt(np.mean(e*e))),
      "per_kwp_r2":float(1.0-sse/sst),
      "per_kwp_bias":float(np.mean(e)),
      "per_kwp_mean_true":float(y.mean()),
      "per_kwp_mean_pred":float(p.mean()),
      "per_kwp_count":int(y.size),
    }
    for k in ("mae","rmse","bias","mean_true","mean_pred"):
        out["reference_kw_"+k]=out["per_kwp_"+k]*REF_KWP
    out["reference_kw_r2"]=out["per_kwp_r2"]
    out["reference_kw_count"]=out["per_kwp_count"]
    pd_=p[d]; yd=y[d]; ed=pd_-yd
    out.update({
      "daylight_per_kwp_mae":float(np.mean(np.abs(ed))),
      "daylight_per_kwp_rmse":float(np.sqrt(np.mean(ed*ed))),
      "daylight_per_kwp_r2":float(1.0-np.sum(ed*ed)/np.sum((yd-yd.mean())**2)),
      "daylight_per_kwp_bias":float(np.mean(ed)),
      "daylight_per_kwp_mean_true":float(yd.mean()),
      "daylight_per_kwp_mean_pred":float(pd_.mean()),
      "daylight_per_kwp_count":int(yd.size),
    })
    return out

class StrictAlignedTestDataset(Dataset):
    def __init__(self,base,prepared_city:pd.DataFrame,features:Sequence[str]):
        g=prepared_city.sort_values(base.TIME_COL).reset_index(drop=True)
        if len(g)!=EXPECTED_RAW_ROWS_PER_CITY:
            raise RuntimeError(f"raw city rows {len(g)} != {EXPECTED_RAW_ROWS_PER_CITY}")
        g=g.iloc[ALIGN_TRIM:].reset_index(drop=True)
        if len(g)!=EXPECTED_ALIGNED_ROWS_PER_CITY:
            raise RuntimeError(f"aligned rows {len(g)} != {EXPECTED_ALIGNED_ROWS_PER_CITY}")
        self.x=np.ascontiguousarray(g[list(features)].to_numpy(np.float32)).copy()
        self.ys=np.ascontiguousarray(g[base.TARGET_SCALED_COL].to_numpy(np.float32)).copy()
        self.yr=np.ascontiguousarray(g[base.TARGET_VALUE_COL].to_numpy(np.float32)).copy()
        self.day=np.ascontiguousarray(g[base.DAYLIGHT_COL].to_numpy(np.float32)).copy()
        self.times=pd.DatetimeIndex(pd.to_datetime(g[base.TIME_COL],utc=True))
        self.index_map=list(range(MIN_START,len(g)-H+1))
        if len(self.index_map)!=EXPECTED_WINDOWS_PER_CITY:
            raise RuntimeError(f"test windows {len(self.index_map)} != {EXPECTED_WINDOWS_PER_CITY}")
    def __len__(self): return len(self.index_map)
    def __getitem__(self,i):
        t=self.index_map[i]
        return (
          torch.from_numpy(self.x[t-L:t].copy()),
          torch.from_numpy(self.ys[t:t+H].copy()),
          torch.from_numpy(self.yr[t:t+H].copy()),
          torch.from_numpy(self.day[t:t+H].copy()),
        )

def prepare_city(base,test_full,target_col,city,features,continuous,x_scaler,y_scaler):
    df=test_full[test_full[base.CITY_COL].astype(str)==city].copy()
    built,bcont=base.build_feature_columns(df,"basic",False)
    if list(built)!=list(features) or list(bcont)!=list(continuous):
        raise RuntimeError(f"{city}: feature binding changed")
    req=list(features)+[base.TARGET_VALUE_COL,base.RAW_TARGET_COL,base.DAYLIGHT_COL]
    n0=len(df); df=df.dropna(subset=req).copy()
    if len(df)!=n0: raise RuntimeError(f"{city}: unexpected NaN drop {n0-len(df)}")
    df.loc[:,list(continuous)]=x_scaler.transform(df[list(continuous)].to_numpy(np.float32))
    df.loc[:,base.TARGET_SCALED_COL]=y_scaler.transform(
        df[[base.TARGET_VALUE_COL]].to_numpy(np.float32)).reshape(-1)
    return df.sort_values(base.TIME_COL).reset_index(drop=True)

def build_model(base,nf,device):
    return base.SolarLSTM(n_features=nf,hidden_size=256,num_layers=3,dropout=0.20,horizon=H,head_hidden=128).to(device)

def verify_pretest_freeze():
    fail=[]
    prot=A30/"FEWSHOT_PROTOCOL_LOCK_FINAL_v1_1.json"
    rep=A30/"A30_freeze_report_v1.json"
    man=A30/"fewshot_adaptation_freeze_manifest_v1.csv"
    reg=A30/"adapted_checkpoint_registry_v1.csv"
    for p,e,label in [
      (prot,EXPECTED_PROTOCOL_SHA,"protocol"),(rep,EXPECTED_A30_REPORT_SHA,"A30 report"),
      (man,EXPECTED_FREEZE_MANIFEST_SHA,"freeze manifest"),(reg,EXPECTED_ADAPTED_REGISTRY_SHA,"adapted registry"),
      (BASE_CODE,EXPECTED_BASE_SHA,"base code")]:
        if not p.is_file(): fail.append(f"missing {label}: {p}")
        elif sha256(p)!=e: fail.append(f"{label} SHA mismatch")
    if rep.is_file():
        d=load_json(rep)
        for k,v in [("status","PASS"),("stage_complete",True),("adapted_checkpoints_frozen",270),
                    ("adapted_checkpoint_sha_unique",270),("n_failures",0),("adapted_test_authorized",True),
                    ("test_dataset_opened",False),("test_inference_performed",False)]:
            if d.get(k)!=v: fail.append(f"A30 {k}={d.get(k)!r} expected {v!r}")
    if not A60_STAGE_LOCK.is_file():
        fail.append(f"missing closed A60 stage lock: {A60_STAGE_LOCK}")
    elif sha256(A60_STAGE_LOCK)!=EXPECTED_A60_STAGE_LOCK_SHA:
        fail.append("A60 stage-lock SHA mismatch")
    else:
        a60=load_json(A60_STAGE_LOCK)
        for k,v in [("status","PASS"),("stage_complete",True),("stage","A60"),
                    ("horizon",24),("budget","365d"),
                    ("test_dataset_opened",True),("test_inference_performed",True)]:
            if a60.get(k)!=v:
                fail.append(f"A60 prerequisite {k}={a60.get(k)!r}, expected {v!r}")
    if not A70_STAGE_LOCK.is_file():
        fail.append(f"missing closed A70 stage lock: {A70_STAGE_LOCK}")
    elif sha256(A70_STAGE_LOCK)!=EXPECTED_A70_STAGE_LOCK_SHA:
        fail.append("A70 stage-lock SHA mismatch")
    else:
        a70=load_json(A70_STAGE_LOCK)
        for k,v in [("status","PASS"),("stage_complete",True),("stage","A70"),
                    ("horizon",48),("budget","30d"),
                    ("test_dataset_opened",True),("test_inference_performed",True)]:
            if a70.get(k)!=v:
                fail.append(f"A70 prerequisite {k}={a70.get(k)!r}, expected {v!r}")
    if not A80_STAGE_LOCK.is_file():
        fail.append(f"missing closed A80 stage lock: {A80_STAGE_LOCK}")
    elif sha256(A80_STAGE_LOCK)!=EXPECTED_A80_STAGE_LOCK_SHA:
        fail.append("A80 stage-lock SHA mismatch")
    else:
        a80=load_json(A80_STAGE_LOCK)
        for k,v in [("status","PASS"),("stage_complete",True),("stage","A80"),
                    ("horizon",48),("budget","90d"),
                    ("test_dataset_opened",True),("test_inference_performed",True)]:
            if a80.get(k)!=v:
                fail.append(f"A80 prerequisite {k}={a80.get(k)!r}, expected {v!r}")
    if fail: raise RuntimeError("A90 pre-Test freeze failure:\n- "+"\n- ".join(fail))
    registry=pd.read_csv(reg)
    if len(registry)!=270: raise RuntimeError("A30 adapted registry row count changed")
    target=registry[(registry.horizon==48)&(registry.budget=="365d")].copy()
    if len(target)!=45: raise RuntimeError(f"A90 registry grid {len(target)} !=45")
    # Rehash every H24/30d adapted checkpoint BEFORE opening Test.
    checks=[]
    for _,r in target.iterrows():
        p=Path(r.adapted_checkpoint)
        got=sha256(p)
        if got!=r.adapted_checkpoint_sha256:
            raise RuntimeError(f"Frozen adapted checkpoint changed: {p}")
        checks.append({"city":r.city,"seed":int(r.seed),"checkpoint":str(p),"sha256":got})
    return checks

def main():
    # Critical ordering: A30 integrity and all 45 target checkpoints are checked before test.csv is read.
    frozen_checks=verify_pretest_freeze()

    # This is the first operation that reads Test.
    if not TEST_PATH.is_file(): raise RuntimeError(f"missing Test file {TEST_PATH}")
    test_sha=sha256(TEST_PATH)
    if test_sha!=EXPECTED_TEST_SHA: raise RuntimeError(f"Test SHA mismatch {test_sha}")

    base=import_file("fewshot_A40_base",BASE_CODE)
    test_full,target_col=base.load_split(str(TEST_PATH),"test","auto")
    test_full=base.prepare_physical_target(test_full,target_col)
    if len(test_full)!=EXPECTED_RAW_ROWS_PER_CITY*15:
        raise RuntimeError(f"Test rows {len(test_full)} unexpected")
    if sorted(test_full[base.CITY_COL].astype(str).unique())!=sorted(CITIES):
        raise RuntimeError("Test city list changed")

    shutil.rmtree(OUT,ignore_errors=True); OUT.mkdir(parents=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"A90 H48 365d adapted Test | device={device}")
    print(f"Test SHA256={test_sha}")

    city_metric_rows=[]; per_h_rows=[]; pairing_rows=[]; npz_manifest=[]
    for city in CITIES:
        sf=SOURCE_ROOT/"folds"/city
        features=json.loads((sf/"feature_cols.json").read_text())
        continuous=json.loads((sf/"continuous_cols.json").read_text())
        if len(features)!=34 or len(continuous)!=20: raise RuntimeError(f"{city}: feature scope")
        xsc=joblib.load(sf/"x_scaler.pkl"); ysc=joblib.load(sf/"y_scaler.pkl")
        prepared=prepare_city(base,test_full,target_col,city,features,continuous,xsc,ysc)
        ds=StrictAlignedTestDataset(base,prepared,features)
        starts=np.asarray(ds.index_map,np.int64)
        times_ns=np.fromiter((pd.Timestamp(x).value for x in ds.times),dtype=np.int64,count=len(ds.times))
        first_valid=pd.to_datetime(times_ns[starts],utc=True)
        origin=first_valid-pd.Timedelta(hours=1)
        if first_valid[0]!=EXPECTED_FIRST_VALID or pd.to_datetime(times_ns[starts[-1]+H-1],utc=True)!=EXPECTED_LAST_VALID:
            raise RuntimeError(f"{city}: Test valid-time bounds changed")

        # Exact pairing with already-closed Stage-60 zero-shot H24 cold-start grid/truth.
        zmeta=pd.read_csv(ZERO_ROOT/"cities"/city/"test_window_metadata.csv.gz")
        z=np.load(ZERO_ROOT/"cities"/city/"seed_1"/"test_predictions.npz",allow_pickle=False)
        if len(zmeta)!=len(ds) or not np.array_equal(zmeta.start_index.to_numpy(np.int64),starts):
            raise RuntimeError(f"{city}: Stage80 metadata pairing mismatch")
        if not np.array_equal(z["window_start_idx"].astype(np.int64),starts):
            raise RuntimeError(f"{city}: Stage80 start-index pairing mismatch")
        if not np.array_equal(z["series_time_ns"].astype(np.int64),times_ns):
            raise RuntimeError(f"{city}: Stage80 series-time pairing mismatch")

        cdir=OUT/"cities"/city; cdir.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({
          "start_index":starts,
          "forecast_origin":origin.astype(str),
          "first_valid_time":first_valid.astype(str),
        }).to_csv(cdir/"test_window_metadata.csv.gz",index=False,compression="gzip")

        for seed in SEEDS:
            run=ADAPT_ROOT/"folds"/city/BUDGET/f"seed_{seed}"
            rlock=load_json(run/"run_lock.json")
            ck=run/"adapted_checkpoint.pt"
            corrp=sf/"L24"/f"seed_{seed}"/"correction.json"
            if sha256(ck)!=rlock["adapted_checkpoint_sha256"]:
                raise RuntimeError(f"{city} seed{seed}: adapted checkpoint changed after A30")
            if sha256(corrp)!=rlock["source_correction_sha256"]:
                raise RuntimeError(f"{city} seed{seed}: source correction changed")
            corr=load_json(corrp)
            if corr.get("method")!="horizon_daylight_bias": raise RuntimeError("correction method changed")

            model=build_model(base,len(features),device)
            state=torch.load(ck,map_location=device,weights_only=True)
            model.load_state_dict(state)
            loader=DataLoader(ds,batch_size=BATCH,shuffle=False,num_workers=0,
                              pin_memory=(device=="cuda"),drop_last=False)
            pred,true,day=base.collect_predictions(model,loader,ysc,device,False,True,MAX_PER_KWP,True)
            pred=base.apply_bias_correction(pred,day,corr)
            pred=base.final_physical_postprocess(pred,day,True,MAX_PER_KWP,True)
            pred=np.asarray(pred,np.float32); true=np.asarray(true,np.float32); day=np.asarray(day,np.float32)

            if pred.shape!=(EXPECTED_WINDOWS_PER_CITY,H): raise RuntimeError(f"{city} seed{seed}: shape {pred.shape}")
            if not np.array_equal(true,z["true"]): raise RuntimeError(f"{city} seed{seed}: truth differs from Stage80")
            if not np.array_equal(day,z["daylight"]): raise RuntimeError(f"{city} seed{seed}: daylight differs from Stage80")
            if float(pred.min()) < -1e-7 or float(pred.max()) > MAX_PER_KWP+1e-7:
                raise RuntimeError(f"{city} seed{seed}: physical bounds")
            if np.any(~day.astype(bool)) and float(np.max(np.abs(pred[~day.astype(bool)])))>1e-7:
                raise RuntimeError(f"{city} seed{seed}: nighttime nonzero")

            sdir=cdir/f"seed_{seed}"; sdir.mkdir(parents=True,exist_ok=True)
            npz=sdir/"test_predictions.npz"
            np.savez_compressed(npz,
              pred=pred,true=true,daylight=day,
              window_start_idx=starts,series_time_ns=times_ns,
              horizon=np.int64(H),lookback=np.int64(L),seed=np.int64(seed),city=np.asarray(city))
            npz_manifest.append({"city":city,"seed":seed,"path":str(npz),"sha256":sha256(npz),"size_bytes":npz.stat().st_size})

            met=metric_dict(pred,true,day)
            city_metric_rows.append({"target_city":city,"budget":BUDGET,"seed":seed,**met})
            for h in range(H):
                per_h_rows.append({"target_city":city,"budget":BUDGET,"seed":seed,"horizon_step":h+1,
                                   **metric_dict(pred[:,h:h+1],true[:,h:h+1],day[:,h:h+1])})
            pairing_rows.append({"city":city,"seed":seed,"windows":len(ds),
              "starts_equal_stage80":True,"series_times_equal_stage80":True,
              "truth_equal_stage80":True,"daylight_equal_stage80":True})

    city_df=pd.DataFrame(city_metric_rows)
    city_df.to_csv(OUT/"adapted_test_by_city_and_seed.csv",index=False)
    pd.DataFrame(per_h_rows).to_csv(OUT/"adapted_test_per_horizon_by_city.csv",index=False)
    pd.DataFrame(pairing_rows).to_csv(OUT/"stage80_pairing_audit.csv",index=False)
    pd.DataFrame(npz_manifest).to_csv(OUT/"raw_npz_manifest.csv",index=False)
    pd.DataFrame(frozen_checks).to_csv(OUT/"pretest_frozen_checkpoint_checks.csv",index=False)

    pooled_rows=[]; energy_city=[]
    for seed in SEEDS:
        ps=[]; ys=[]; ds_=[]
        for city in CITIES:
            z=np.load(OUT/"cities"/city/f"seed_{seed}"/"test_predictions.npz",allow_pickle=False)
            ps.append(z["pred"]); ys.append(z["true"]); ds_.append(z["daylight"])
            starts=z["window_start_idx"].astype(np.int64); times=z["series_time_ns"].astype(np.int64)
            valid=times[starts[:,None]+np.arange(H,dtype=np.int64)[None,:]]
            f=pd.DataFrame({"t":valid.ravel(),"p":z["pred"].astype(np.float64).ravel(),"y":z["true"].astype(np.float64).ravel()})
            u=f.groupby("t",sort=True,as_index=False)[["p","y"]].mean()
            if len(u)!=EXPECTED_UNIQUE_HOURS_PER_CITY: raise RuntimeError(f"{city} seed{seed}: unique hours {len(u)}")
            act=float(u.y.sum()*REF_KWP/1000.0); pr=float(u.p.sum()*REF_KWP/1000.0)
            energy_city.append({"target_city":city,"seed":seed,"n_unique_target_hours":len(u),
              "actual_reference_energy_mwh":act,"predicted_reference_energy_mwh":pr,
              "reference_energy_error_mwh":pr-act,"energy_error_pct":100*(pr-act)/act})
        pred=np.concatenate(ps); true=np.concatenate(ys); day=np.concatenate(ds_)
        met=metric_dict(pred,true,day)
        if pred.shape!=(EXPECTED_WINDOWS_POOLED,H) or met["per_kwp_count"]!=EXPECTED_RESIDUALS or met["daylight_per_kwp_count"]!=EXPECTED_DAYLIGHT:
            raise RuntimeError(f"seed{seed}: pooled count mismatch")
        pooled_rows.append({"budget":BUDGET,"seed":seed,**met})

    pool=pd.DataFrame(pooled_rows); pool.to_csv(OUT/"adapted_pooled_test_seed_results.csv",index=False)
    ec=pd.DataFrame(energy_city); ec.to_csv(OUT/"adapted_energy_by_city.csv",index=False)
    ea=ec.groupby("seed",as_index=False).agg(
      actual_reference_energy_mwh=("actual_reference_energy_mwh","sum"),
      predicted_reference_energy_mwh=("predicted_reference_energy_mwh","sum"),
      reference_energy_error_mwh=("reference_energy_error_mwh","sum"),
      n_unique_target_hours_total=("n_unique_target_hours","sum"))
    ea["energy_error_pct"]=100*ea.reference_energy_error_mwh/ea.actual_reference_energy_mwh
    if not np.all(ea.n_unique_target_hours_total==EXPECTED_UNIQUE_CITY_HOURS): raise RuntimeError("unique city-hour total mismatch")
    if not np.allclose(ea.actual_reference_energy_mwh,EXPECTED_TRUE_ENERGY_MWH,rtol=0,atol=1e-6):
        raise RuntimeError(f"true energy mismatch {ea.actual_reference_energy_mwh.tolist()}")
    ea.to_csv(OUT/"adapted_aggregate_energy_by_seed.csv",index=False)

    summary={"n_seeds":3,"sd_ddof":1,"budget":BUDGET}
    for c in pool.columns:
        if c not in ("seed","budget") and pd.api.types.is_numeric_dtype(pool[c]):
            summary[c+"_mean"]=float(pool[c].mean()); summary[c+"_std"]=float(pool[c].std(ddof=1))
    for c in ("predicted_reference_energy_mwh","reference_energy_error_mwh","energy_error_pct"):
        summary[c+"_mean"]=float(ea[c].mean()); summary[c+"_std"]=float(ea[c].std(ddof=1))
    pd.DataFrame([summary]).to_csv(OUT/"adapted_pooled_test_summary_ddof1.csv",index=False)

    stage={
      "status":"PASS","stage":"A90","stage_complete":True,"horizon":H,"budget":BUDGET,
      "cities":15,"seeds":SEEDS,"adapted_checkpoints_evaluated":45,
      "test_dataset_opened":True,"test_inference_performed":True,
      "test_sha256":test_sha,"A30_report_sha256":sha256(A30/"A30_freeze_report_v1.json"),
      "A30_freeze_manifest_sha256":sha256(A30/"fewshot_adaptation_freeze_manifest_v1.csv"),
      "A30_adapted_registry_sha256":sha256(A30/"adapted_checkpoint_registry_v1.csv"),
      "A60_stage_lock_sha256":sha256(A60_STAGE_LOCK),
      "A70_stage_lock_sha256":sha256(A70_STAGE_LOCK),
      "A80_stage_lock_sha256":sha256(A80_STAGE_LOCK),
      "same_grid_as_stage80_zero_shot":True,"same_truth_as_stage80_zero_shot":True,
      "same_daylight_as_stage80_zero_shot":True,
      "windows_per_city":EXPECTED_WINDOWS_PER_CITY,"pooled_windows_per_seed":EXPECTED_WINDOWS_POOLED,
      "residuals_per_seed":EXPECTED_RESIDUALS,"daylight_residuals_per_seed":EXPECTED_DAYLIGHT,
      "unique_city_hours_per_seed":EXPECTED_UNIQUE_CITY_HOURS,"true_energy_mwh":EXPECTED_TRUE_ENERGY_MWH,
      "sample_sd_ddof":1,
      "raw_npz_manifest_sha256":sha256(OUT/"raw_npz_manifest.csv"),
      "pooled_results_sha256":sha256(OUT/"adapted_pooled_test_seed_results.csv"),
      "energy_sha256":sha256(OUT/"adapted_aggregate_energy_by_seed.csv"),
      "summary_sha256":sha256(OUT/"adapted_pooled_test_summary_ddof1.csv"),
    }
    dump_json(OUT/"A90_H48_365d_stage_lock.json",stage)
    print(json.dumps(stage,indent=2))
    print("A90 H48 365d ADAPTED TEST PASS")

if __name__=="__main__": main()
