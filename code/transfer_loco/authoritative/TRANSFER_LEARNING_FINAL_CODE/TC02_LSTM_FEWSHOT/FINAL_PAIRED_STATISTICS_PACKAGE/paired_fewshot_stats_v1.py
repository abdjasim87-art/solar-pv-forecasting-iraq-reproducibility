#!/usr/bin/env python3
from __future__ import annotations
import os, json, hashlib, math, shutil
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.stats import t as t_dist
except Exception:
    t_dist = None

W = Path(os.environ.get("LSTM_WORKSPACE", "/workspace/lstm"))
ZERO_BASE = W/"output"/"loco_transfer_v2"
ADAPT_BASE = W/"output"/"loco_fewshot"
FREEZE = W/"output"/"loco_fewshot_stats_freeze_v1"
OUT = W/"output"/"loco_fewshot_paired_stats_v1"
CITIES = ["Anbar","Babylon","Baghdad","Basra","Dhi_Qar","Diyala","Karbala",
          "Kirkuk","Maysan","Muthanna","Najaf","Nineveh","Qadisiyyah",
          "Salah_al_Din","Wasit"]
SEEDS = (1,2,3)
HORIZONS = (24,48)
BUDGETS = ("30d","90d","365d")
N_BOOT = 5000
BLOCK_DAYS = 7
BOOT_SEED_BASE = 20260806
HAC_LAG_DAYS = 7


def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def holm(pvals):
    p=np.asarray(pvals,float)
    n=len(p)
    order=np.argsort(p)
    adj=np.empty(n,float)
    running=0.0
    for rank,idx in enumerate(order):
        val=(n-rank)*p[idx]
        running=max(running,val)
        adj[idx]=min(1.0,running)
    return adj


def nw_hac_p(d, lag, h_days):
    d=np.asarray(d,float)
    d=d[np.isfinite(d)]
    T=len(d)
    if T < 10:
        return np.nan,np.nan,np.nan
    m=float(d.mean())
    x=d-m
    gamma0=float(np.dot(x,x)/T)
    lrv=gamma0
    L=min(int(lag),T-1)
    for l in range(1,L+1):
        gam=float(np.dot(x[l:],x[:-l])/T)
        weight=1.0-l/(L+1.0)
        lrv += 2.0*weight*gam
    if not np.isfinite(lrv) or lrv <= 0:
        return np.nan,np.nan,np.nan
    se=math.sqrt(lrv/T)
    stat=m/se
    h=float(h_days)
    hln=math.sqrt(max((T+1-2*h+h*(h-1)/T)/T,0.0))
    stat_hln=stat*hln
    if t_dist is not None:
        p=2.0*float(t_dist.sf(abs(stat_hln),df=T-1))
    else:
        p=math.erfc(abs(stat_hln)/math.sqrt(2.0))
    return stat_hln,p,se


def make_boot_weights(T, seed):
    rng=np.random.default_rng(seed)
    nblocks=math.ceil(T/BLOCK_DAYS)
    starts=rng.integers(0,T,size=(N_BOOT,nblocks),endpoint=False)
    offsets=np.arange(BLOCK_DAYS)[None,None,:]
    idx=(starts[:,:,None]+offsets)%T
    idx=idx.reshape(N_BOOT,-1)[:,:T]
    weights=np.zeros((N_BOOT,T),dtype=np.uint16)
    rows=np.repeat(np.arange(N_BOOT),T)
    np.add.at(weights,(rows,idx.ravel()),1)
    return weights


def summarize_scope(daily, metric, weights, H):
    sa=daily["sa"].to_numpy(float)
    sz=daily["sz"].to_numpy(float)
    n=daily["n"].to_numpy(float)
    ta=float(sa.sum()/n.sum())
    tz=float(sz.sum()/n.sum())
    if metric == "RMSE":
        ma=math.sqrt(ta); mz=math.sqrt(tz)
        ba=np.sqrt((weights@sa)/(weights@n))
        bz=np.sqrt((weights@sz)/(weights@n))
    else:
        ma=ta; mz=tz
        ba=(weights@sa)/(weights@n)
        bz=(weights@sz)/(weights@n)
    delta=ma-mz
    bd=ba-bz
    lo,hi=np.quantile(bd,[0.025,0.975])
    daily_diff=sa/n-sz/n
    stat,p,se=nw_hac_p(daily_diff,HAC_LAG_DAYS,H//24)
    improve_pct=(-delta/mz*100.0) if mz != 0 else np.nan
    return {
        "adapted_metric":ma,
        "zero_shot_metric":mz,
        "delta_adapted_minus_zero":delta,
        "improvement_pct_vs_zero":improve_pct,
        "bootstrap_ci_low":float(lo),
        "bootstrap_ci_high":float(hi),
        "bootstrap_reps":N_BOOT,
        "bootstrap_block_days":BLOCK_DAYS,
        "dm_hln_stat":stat,
        "dm_p_raw":p,
        "dm_daily_n":len(daily_diff),
        "dm_hac_lag_days":HAC_LAG_DAYS,
        "dm_h_days":H//24,
        "ci_excludes_zero":bool(lo>0 or hi<0),
    }


def load_loss_arrays(path: Path):
    z=np.load(path,allow_pickle=False)
    required=("pred","true","daylight","window_start_idx","series_time_ns")
    missing=[k for k in required if k not in z.files]
    if missing:
        raise RuntimeError(f"{path}: missing NPZ keys {missing}")
    e=z["pred"].astype(np.float64)-z["true"].astype(np.float64)
    return {
        "sq":e*e,
        "ab":np.abs(e),
        "true":z["true"],
        "day":z["daylight"].astype(bool),
        "start":z["window_start_idx"],
        "time":z["series_time_ns"],
    }


def main():
    shutil.rmtree(OUT,ignore_errors=True)
    OUT.mkdir(parents=True)

    freeze_report=FREEZE/"B00_freeze_report_v1.json"
    freeze_manifest=FREEZE/"fewshot_stats_freeze_manifest_v1.json"
    if not freeze_report.exists() or not freeze_manifest.exists():
        raise SystemExit("B00 freeze outputs missing. Run and review 00_B00_freeze.sh first.")
    rep=json.loads(freeze_report.read_text(encoding="utf-8"))
    if rep.get("status") != "PASS" or not rep.get("statistics_authorized_after_review",False):
        raise SystemExit("B00 freeze is not PASS/authorized.")

    manifest=json.loads(freeze_manifest.read_text(encoding="utf-8"))
    changed=[]
    for pstr,meta in manifest.get("files",{}).items():
        p=Path(pstr)
        expected=meta["sha256"] if isinstance(meta,dict) else meta
        if not p.exists() or sha256(p) != expected:
            changed.append(pstr)
    if changed:
        (OUT/"changed_after_B00_freeze.json").write_text(json.dumps(changed,indent=2),encoding="utf-8")
        raise SystemExit(f"{len(changed)} frozen files changed/missing after B00 freeze.")

    results=[]
    lead_rows=[]
    pairing_fail=[]

    for H in HORIZONS:
        m0=pd.read_csv(ZERO_BASE/f"H{H}"/"cold_start"/"test"/"cities"/CITIES[0]/"test_window_metadata.csv.gz")
        dates0=pd.to_datetime(m0["forecast_origin"],utc=True).dt.strftime("%Y-%m-%d").to_numpy()
        udates=np.array(sorted(pd.unique(dates0)))
        date_to_i={d:i for i,d in enumerate(udates)}
        weights=make_boot_weights(len(udates),BOOT_SEED_BASE+H)

        acc={}
        for budget in BUDGETS:
            for daytype in ("daylight","all_hours"):
                for metric in ("RMSE","MAE"):
                    for scope in ["pooled"]+CITIES:
                        acc[(budget,daytype,metric,scope)]={
                            "sa":np.zeros(len(udates),dtype=np.float64),
                            "sz":np.zeros(len(udates),dtype=np.float64),
                            "n":np.zeros(len(udates),dtype=np.int64),
                        }

        lead_acc={(budget,daytype,metric,lead):[0.0,0.0,0]
                  for budget in BUDGETS
                  for daytype in ("daylight","all_hours")
                  for metric in ("RMSE","MAE")
                  for lead in range(1,H+1)}

        for city in CITIES:
            meta=pd.read_csv(ZERO_BASE/f"H{H}"/"cold_start"/"test"/"cities"/city/"test_window_metadata.csv.gz")
            city_dates=pd.to_datetime(meta["forecast_origin"],utc=True).dt.strftime("%Y-%m-%d").to_numpy()
            try:
                day_idx=np.array([date_to_i[d] for d in city_dates],dtype=np.int64)
            except KeyError as e:
                pairing_fail.append(f"H{H} {city}: origin date not in master date grid: {e}")
                continue

            z_sq=[]; z_ab=[]; z_ref=None
            for seed in SEEDS:
                d=load_loss_arrays(ZERO_BASE/f"H{H}"/"cold_start"/"test"/"cities"/city/f"seed_{seed}"/"test_predictions.npz")
                z_sq.append(d["sq"]); z_ab.append(d["ab"])
                if z_ref is None:
                    z_ref=d
                else:
                    for k in ("true","day","start","time"):
                        if not np.array_equal(z_ref[k],d[k]):
                            pairing_fail.append(f"H{H} {city} zero-shot: {k} differs across seeds")
            zloss={"sq":np.mean(np.stack(z_sq),axis=0),"ab":np.mean(np.stack(z_ab),axis=0)}

            for budget in BUDGETS:
                a_sq=[]; a_ab=[]; a_ref=None
                for seed in SEEDS:
                    d=load_loss_arrays(ADAPT_BASE/f"H{H}"/"test"/budget/"cities"/city/f"seed_{seed}"/"test_predictions.npz")
                    a_sq.append(d["sq"]); a_ab.append(d["ab"])
                    if a_ref is None:
                        a_ref=d
                    else:
                        for k in ("true","day","start","time"):
                            if not np.array_equal(a_ref[k],d[k]):
                                pairing_fail.append(f"H{H} {budget} {city} adapted: {k} differs across seeds")
                for k in ("true","day","start","time"):
                    if not np.array_equal(a_ref[k],z_ref[k]):
                        pairing_fail.append(f"H{H} {budget} {city}: adapted/zero {k} mismatch")
                aloss={"sq":np.mean(np.stack(a_sq),axis=0),"ab":np.mean(np.stack(a_ab),axis=0)}

                for daytype in ("daylight","all_hours"):
                    mask=z_ref["day"] if daytype=="daylight" else np.ones_like(z_ref["day"],dtype=bool)
                    for metric,field in (("RMSE","sq"),("MAE","ab")):
                        la=aloss[field]; lz=zloss[field]
                        for lead in range(H):
                            ml=mask[:,lead]
                            A=lead_acc[(budget,daytype,metric,lead+1)]
                            A[0]+=float(la[:,lead][ml].sum())
                            A[1]+=float(lz[:,lead][ml].sum())
                            A[2]+=int(ml.sum())
                        for di in np.unique(day_idx):
                            wm=(day_idx==di)[:,None] & mask
                            n=int(wm.sum())
                            if n==0:
                                continue
                            sa=float(la[wm].sum()); sz=float(lz[wm].sum())
                            for scope in ("pooled",city):
                                A=acc[(budget,daytype,metric,scope)]
                                A["sa"][di]+=sa; A["sz"][di]+=sz; A["n"][di]+=n

        if pairing_fail:
            (OUT/"pairing_failures.json").write_text(json.dumps(pairing_fail,indent=2),encoding="utf-8")
            raise SystemExit(f"Pairing failed: {len(pairing_fail)} issues.")

        for (budget,daytype,metric,scope),A in acc.items():
            keep=A["n"]>0
            daily=pd.DataFrame({"date":udates[keep],"sa":A["sa"][keep],"sz":A["sz"][keep],"n":A["n"][keep]})
            wuse=weights if np.all(keep) else make_boot_weights(int(keep.sum()),BOOT_SEED_BASE+H)
            r=summarize_scope(daily,metric,wuse,H)
            r.update({
                "family":"adapted_vs_cold_start_zero_shot",
                "horizon":H,
                "budget":budget,
                "daytype":daytype,
                "metric":metric,
                "scope_type":"pooled" if scope=="pooled" else "city",
                "scope":scope,
                "effect_sign":"adapted_minus_zero; negative_favors_adapted",
            })
            results.append(r)

        for (budget,daytype,metric,lead),A in lead_acc.items():
            sa,sz,n=A
            if n==0:
                continue
            if metric=="RMSE":
                ma=math.sqrt(sa/n); mz=math.sqrt(sz/n)
            else:
                ma=sa/n; mz=sz/n
            lead_rows.append({
                "horizon":H,"budget":budget,"daytype":daytype,"metric":metric,"lead":lead,
                "adapted_metric":ma,"zero_shot_metric":mz,
                "delta_adapted_minus_zero":ma-mz,
                "improvement_pct_vs_zero":(-(ma-mz)/mz*100.0) if mz else np.nan,
                "n":n,
            })

    df=pd.DataFrame(results)
    df["dm_p_holm"]=np.nan
    df["holm_family_size"]=np.nan
    # Pre-specified families: pooled 3 budgets; city 15 cities x 3 budgets = 45.
    for (H,daytype,metric,stype),g in df.groupby(["horizon","daytype","metric","scope_type"],sort=False):
        idx=g.index.to_numpy()
        pvals=g["dm_p_raw"].to_numpy(float)
        df.loc[idx,"dm_p_holm"]=holm(pvals)
        df.loc[idx,"holm_family_size"]=len(idx)
    df["strong_support"]=df["ci_excludes_zero"] & (df["dm_p_holm"]<0.05)
    df["favored_by_point_estimate"]=np.where(df["delta_adapted_minus_zero"]<0,"adapted",
                                               np.where(df["delta_adapted_minus_zero"]>0,"cold_start_zero_shot","tie"))

    df.to_csv(OUT/"fewshot_paired_stats_all_v1.csv",index=False)
    df[df.scope_type=="pooled"].to_csv(OUT/"fewshot_paired_stats_pooled_v1.csv",index=False)
    df[df.scope_type=="city"].to_csv(OUT/"fewshot_paired_stats_by_city_v1.csv",index=False)
    pd.DataFrame(lead_rows).to_csv(OUT/"fewshot_paired_per_lead_descriptive_v1.csv",index=False)

    pooled=df[df.scope_type=="pooled"].copy()
    pooled.to_csv(OUT/"fewshot_budget_curve_pooled_v1.csv",index=False)

    summary={
        "status":"PASS",
        "analysis":"paired adapted few-shot versus frozen cold-start zero-shot",
        "n_tests":int(len(df)),
        "n_pooled_tests":int((df.scope_type=="pooled").sum()),
        "n_city_tests":int((df.scope_type=="city").sum()),
        "expected_tests":384,
        "bootstrap_reps":N_BOOT,
        "block_days":BLOCK_DAYS,
        "bootstrap_seed_base":BOOT_SEED_BASE,
        "hac_lag_days":HAC_LAG_DAYS,
        "holm_pooled_family_size":3,
        "holm_city_family_size":45,
        "pairing_failures":0,
        "frozen_files_rehashed":len(manifest.get("files",{})),
        "changed_after_freeze":0,
        "cross_budget_inferential_tests_run":False,
        "budget_seasonality_caveat_preserved":True,
    }
    if len(df)!=384:
        summary["status"]="FAIL"
        summary["unexpected_test_count"]=True
    (OUT/"B10_paired_stats_report_v1.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    raise SystemExit(0 if summary["status"]=="PASS" else 2)

if __name__ == "__main__":
    main()
