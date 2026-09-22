#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

CITIES = [
    "Anbar","Babylon","Baghdad","Basra","Dhi_Qar","Diyala","Karbala",
    "Kirkuk","Maysan","Muthanna","Najaf","Nineveh","Qadisiyyah",
    "Salah_al_Din","Wasit"
]
SEEDS=(1,2,3)
H=48
L=24
REF_KWP=3370.0
EXPECTED_WINDOWS_PER_CITY=17161
EXPECTED_WINDOWS_POOLED=257415
EXPECTED_RESIDUALS=12355920
EXPECTED_DAYLIGHT=6199242
EXPECTED_UNIQUE_CITY_HOURS=258120
EXPECTED_TRUE_ENERGY_MWH=171314.7648391358

def ts_ns(series):
    # Robust across pandas datetime resolutions (ns/us/ms).
    return np.fromiter((pd.Timestamp(x).value for x in series), dtype=np.int64, count=len(series))

def metric_dict(pred,true,day):
    p=pred.astype(np.float64).ravel()
    y=true.astype(np.float64).ravel()
    d=day.astype(bool).ravel()
    e=p-y
    sse=np.sum(e*e)
    sst=np.sum((y-y.mean())**2)
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

def metric_close(name,a,b):
    if "count" in name:
        return int(a)==int(b)
    # Saved NPZ arrays are float32. Reference-kW metrics can differ by a few
    # 1e-7 kW when recomputed in float64; this threshold is much tighter than
    # any reportable precision.
    if name.startswith("reference_kw_"):
        return bool(np.isclose(float(a),float(b),rtol=2e-7,atol=2e-5))
    return bool(np.isclose(float(a),float(b),rtol=2e-7,atol=2e-8))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--test_dir",default="/workspace/lstm/output/loco_transfer_v2/H48/history_aware/test")
    ap.add_argument("--out_dir",default="/workspace/lstm/output/loco_transfer_v2/H48/history_aware/raw_audit_v1")
    a=ap.parse_args()
    root=Path(a.test_dir)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)

    saved_city=pd.read_csv(root/"zero_shot_test_by_city_and_seed.csv")
    saved_pool=pd.read_csv(root/"zero_shot_pooled_test_seed_results.csv")
    saved_energy=pd.read_csv(root/"zero_shot_aggregate_energy_by_seed.csv")

    failures=[]
    diagnostics=[]
    pooled={s:{"p":[],"y":[],"d":[]} for s in SEEDS}
    energy_rows=[]
    city_rows=[]
    ref_grid=None
    npz_count=0

    for city in CITIES:
        meta=pd.read_csv(root/"cities"/city/"test_window_metadata.csv.gz")
        if len(meta)!=EXPECTED_WINDOWS_PER_CITY:
            failures.append(f"{city}: metadata windows={len(meta)}")
        starts_csv=meta.start_index.to_numpy(np.int64)
        if int(starts_csv[0])!=168 or not np.all(np.diff(starts_csv)==1):
            failures.append(f"{city}: invalid start-index grid")
        fo=pd.to_datetime(meta.forecast_origin,utc=True)
        fv=pd.to_datetime(meta.first_valid_time,utc=True)
        if not np.all((fv-fo)==pd.Timedelta(hours=1)):
            failures.append(f"{city}: first_valid != origin+1h")
        grid=pd.DataFrame({"start":starts_csv,"fo_ns":ts_ns(fo),"fv_ns":ts_ns(fv)})
        if ref_grid is None:
            ref_grid=grid
        elif not grid.equals(ref_grid):
            failures.append(f"{city}: metadata grid differs across cities")

        truth_hash=[]; day_hash=[]; time_hash=[]
        for seed in SEEDS:
            pth=root/"cities"/city/f"seed_{seed}"/"test_predictions.npz"
            if not pth.is_file():
                failures.append(f"missing {pth}")
                continue
            npz_count+=1
            z=np.load(pth,allow_pickle=False)
            pred=z["pred"]; true=z["true"]; day=z["daylight"]
            starts=z["window_start_idx"].astype(np.int64)
            times=z["series_time_ns"].astype(np.int64)

            if pred.shape!=(EXPECTED_WINDOWS_PER_CITY,H) or true.shape!=pred.shape or day.shape!=pred.shape:
                failures.append(f"{city} seed{seed}: shape mismatch {pred.shape}/{true.shape}/{day.shape}")
            if int(z["horizon"])!=H or int(z["lookback"])!=L or int(z["seed"])!=seed or str(z["city"])!=city:
                failures.append(f"{city} seed{seed}: NPZ metadata mismatch")
            if not np.array_equal(starts,starts_csv):
                failures.append(f"{city} seed{seed}: start-index mismatch")

            fv_ns=ts_ns(fv)
            got=times[starts]
            if not np.array_equal(got,fv_ns):
                delta=got.astype(np.int64)-fv_ns.astype(np.int64)
                diagnostics.append({
                    "city":city,"seed":seed,
                    "npz_first_valid_ns":int(got[0]),
                    "csv_first_valid_ns":int(fv_ns[0]),
                    "first_delta_ns":int(delta[0]),
                    "unique_delta_ns_first_100":sorted(set(delta[:100].tolist()))[:20],
                })
                failures.append(f"{city} seed{seed}: NPZ timestamp mismatch")

            valid_idx=starts[:,None]+np.arange(H,dtype=np.int64)[None,:]
            valid_ns=times[valid_idx]
            if int(valid_ns.min())!=pd.Timestamp("2024-01-15T00:00:00Z").value:
                failures.append(f"{city} seed{seed}: valid min mismatch")
            if int(valid_ns.max())!=pd.Timestamp("2025-12-31T23:00:00Z").value:
                failures.append(f"{city} seed{seed}: valid max mismatch")

            if not (np.isfinite(pred).all() and np.isfinite(true).all() and np.isfinite(day).all()):
                failures.append(f"{city} seed{seed}: nonfinite")
            if float(np.min(pred)) < -1e-7:
                failures.append(f"{city} seed{seed}: negative prediction")
            if float(np.max(pred)) > 1.2000001:
                failures.append(f"{city} seed{seed}: cap violation")
            if np.any(~day.astype(bool)) and float(np.max(np.abs(pred[~day.astype(bool)]))) > 1e-7:
                failures.append(f"{city} seed{seed}: nighttime nonzero")

            truth_hash.append(hashlib.sha256(np.ascontiguousarray(true).view(np.uint8)).hexdigest())
            day_hash.append(hashlib.sha256(np.ascontiguousarray(day).view(np.uint8)).hexdigest())
            time_hash.append(hashlib.sha256(np.ascontiguousarray(valid_ns).view(np.uint8)).hexdigest())

            met=metric_dict(pred,true,day)
            saved=saved_city[(saved_city.target_city==city)&(saved_city.seed==seed)]
            if len(saved)!=1:
                failures.append(f"{city} seed{seed}: missing saved city metric row")
            else:
                sr=saved.iloc[0]
                for k,v in met.items():
                    if k in sr and not metric_close(k,v,sr[k]):
                        failures.append(f"{city} seed{seed}: metric {k} mismatch {v} vs {sr[k]}")
            city_rows.append({"city":city,"seed":seed,**met})
            pooled[seed]["p"].append(pred); pooled[seed]["y"].append(true); pooled[seed]["d"].append(day)

            f=pd.DataFrame({
                "t":valid_ns.ravel(),
                "p":pred.astype(np.float64).ravel(),
                "y":true.astype(np.float64).ravel(),
            })
            u=f.groupby("t",sort=True,as_index=False)[["p","y"]].mean()
            if len(u)!=17208:
                failures.append(f"{city} seed{seed}: unique target hours={len(u)}")
            act=float(u.y.sum()*REF_KWP/1000.0)
            prd=float(u.p.sum()*REF_KWP/1000.0)
            energy_rows.append({
                "city":city,"seed":seed,"n_unique_target_hours":len(u),
                "actual_reference_energy_mwh":act,
                "predicted_reference_energy_mwh":prd,
                "reference_energy_error_mwh":prd-act,
            })

        if len(set(truth_hash))!=1:
            failures.append(f"{city}: true differs across seeds")
        if len(set(day_hash))!=1:
            failures.append(f"{city}: daylight differs across seeds")
        if len(set(time_hash))!=1:
            failures.append(f"{city}: valid timestamps differ across seeds")

    pd.DataFrame(city_rows).to_csv(out/"raw_recomputed_city_metrics.csv",index=False)
    er=pd.DataFrame(energy_rows)
    er.to_csv(out/"raw_recomputed_energy_by_city.csv",index=False)

    pool_rows=[]
    for seed in SEEDS:
        pred=np.concatenate(pooled[seed]["p"],axis=0)
        true=np.concatenate(pooled[seed]["y"],axis=0)
        day=np.concatenate(pooled[seed]["d"],axis=0)
        if pred.shape!=(EXPECTED_WINDOWS_POOLED,H):
            failures.append(f"seed{seed}: pooled shape={pred.shape}")
        met=metric_dict(pred,true,day)
        if met["per_kwp_count"]!=EXPECTED_RESIDUALS:
            failures.append(f"seed{seed}: residual count={met['per_kwp_count']}")
        if met["daylight_per_kwp_count"]!=EXPECTED_DAYLIGHT:
            failures.append(f"seed{seed}: daylight count={met['daylight_per_kwp_count']}")
        sr=saved_pool[saved_pool.seed==seed]
        if len(sr)!=1:
            failures.append(f"seed{seed}: missing saved pooled row")
        else:
            row=sr.iloc[0]
            for k,v in met.items():
                if k in row and not metric_close(k,v,row[k]):
                    failures.append(f"seed{seed}: pooled metric {k} mismatch {v} vs {row[k]}")
        pool_rows.append({"seed":seed,**met})
    pool_df=pd.DataFrame(pool_rows)
    pool_df.to_csv(out/"raw_recomputed_pooled_metrics.csv",index=False)

    agg=er.groupby("seed",as_index=False).agg(
        actual_reference_energy_mwh=("actual_reference_energy_mwh","sum"),
        predicted_reference_energy_mwh=("predicted_reference_energy_mwh","sum"),
        reference_energy_error_mwh=("reference_energy_error_mwh","sum"),
        n_unique_target_hours_total=("n_unique_target_hours","sum"),
    )
    agg["energy_error_pct"]=100*agg.reference_energy_error_mwh/agg.actual_reference_energy_mwh
    agg.to_csv(out/"raw_recomputed_aggregate_energy.csv",index=False)

    for _,r in agg.iterrows():
        seed=int(r.seed)
        sr=saved_energy[saved_energy.seed==seed]
        if len(sr)!=1:
            failures.append(f"seed{seed}: missing saved energy row")
        else:
            row=sr.iloc[0]
            # ≤1 Wh absolute tolerance in MWh; far below reportable precision.
            for k in ["actual_reference_energy_mwh","predicted_reference_energy_mwh","reference_energy_error_mwh"]:
                if not np.isclose(float(r[k]),float(row[k]),rtol=1e-8,atol=1e-6):
                    failures.append(f"seed{seed}: energy {k} mismatch {r[k]} vs {row[k]}")
            if int(r.n_unique_target_hours_total)!=int(row.n_unique_target_hours_total):
                failures.append(f"seed{seed}: saved unique-hour mismatch")
        if int(r.n_unique_target_hours_total)!=EXPECTED_UNIQUE_CITY_HOURS:
            failures.append(f"seed{seed}: total unique city-hours={r.n_unique_target_hours_total}")
        if not np.isclose(float(r.actual_reference_energy_mwh),EXPECTED_TRUE_ENERGY_MWH,rtol=0,atol=1e-6):
            failures.append(f"seed{seed}: true energy={r.actual_reference_energy_mwh}")

    summary={"n_seeds":3,"sd_ddof":1}
    for c in pool_df.columns:
        if c!="seed" and pd.api.types.is_numeric_dtype(pool_df[c]):
            summary[c+"_mean"]=float(pool_df[c].mean())
            summary[c+"_std"]=float(pool_df[c].std(ddof=1))
    pd.DataFrame([summary]).to_csv(out/"raw_recomputed_pooled_summary_ddof1.csv",index=False)

    report={
        "status":"PASS" if not failures else "FAIL",
        "n_failures":len(failures),
        "failures":failures,
        "timestamp_diagnostics":diagnostics,
        "checks":{
            "npz_files_found":npz_count,
            "expected_npz_files":45,
            "pooled_windows_per_seed":EXPECTED_WINDOWS_POOLED,
            "pooled_residuals_per_seed":EXPECTED_RESIDUALS,
            "pooled_daylight_residuals_per_seed":EXPECTED_DAYLIGHT,
            "unique_city_hours_per_seed":EXPECTED_UNIQUE_CITY_HOURS,
            "true_energy_mwh_expected":EXPECTED_TRUE_ENERGY_MWH,
            "robust_timestamp_conversion":"pd.Timestamp.value -> nanoseconds",
            "metrics_recomputed_from_raw_npz":True,
            "energy_recomputed_from_raw_npz":True,
            "night_zero_bounds_checked":True,
            "true_daylight_validtime_seed_identity_checked":True,
        }
    }
    (out/"raw_audit_report_v2.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({
        "status":report["status"],
        "n_failures":len(failures),
        "npz_files_found":npz_count,
        "timestamp_diagnostics_first":diagnostics[:3],
        "failures_first":failures[:30]
    },indent=2))
    raise SystemExit(0 if not failures else 2)

if __name__=="__main__":
    main()
