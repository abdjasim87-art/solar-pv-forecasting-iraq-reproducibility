#!/usr/bin/env python3
from __future__ import annotations
import json, hashlib, math, shutil
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.stats import t as t_dist
except Exception:
    t_dist=None

W=Path("/workspace/lstm")
BASE=W/"output"/"loco_transfer_v2"
FREEZE=W/"output"/"loco_zero_shot_freeze_v1"
OUT=W/"output"/"loco_zero_shot_paired_stats_v1"
CITIES=["Anbar","Babylon","Baghdad","Basra","Dhi_Qar","Diyala","Karbala",
        "Kirkuk","Maysan","Muthanna","Najaf","Nineveh","Qadisiyyah",
        "Salah_al_Din","Wasit"]
SEEDS=(1,2,3)
HORIZONS=(24,48)
TRACKS=("history_aware","cold_start")
N_BOOT=5000
BLOCK=7
BOOT_SEED=20260806
HAC_LAG=7

def sha256(p: Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
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
    if T<10: return np.nan,np.nan,np.nan
    m=d.mean()
    x=d-m
    gamma0=np.dot(x,x)/T
    lrv=gamma0
    L=min(lag,T-1)
    for l in range(1,L+1):
        gam=np.dot(x[l:],x[:-l])/T
        w=1.0-l/(L+1.0)
        lrv += 2*w*gam
    if lrv<=0: return np.nan,np.nan,np.nan
    se=math.sqrt(lrv/T)
    stat=m/se
    h=float(h_days)
    hln=math.sqrt(max((T+1-2*h+h*(h-1)/T)/T,0.0))
    stat_hln=stat*hln
    if t_dist is not None:
        p=2*t_dist.sf(abs(stat_hln),df=T-1)
    else:
        p=math.erfc(abs(stat_hln)/math.sqrt(2.0))
    return stat_hln,p,se

def make_boot_weights(T, seed):
    rng=np.random.default_rng(seed)
    nblocks=math.ceil(T/BLOCK)
    starts=rng.integers(0,T,size=(N_BOOT,nblocks),endpoint=False)
    offsets=np.arange(BLOCK)[None,None,:]
    idx=(starts[:,:,None]+offsets)%T
    idx=idx.reshape(N_BOOT,-1)[:,:T]
    Wt=np.zeros((N_BOOT,T),dtype=np.uint16)
    rows=np.repeat(np.arange(N_BOOT),T)
    np.add.at(Wt,(rows,idx.ravel()),1)
    return Wt

def summarize_scope(daily, metric, Wt, H):
    # daily columns: date, sh, sc, n where sh/sc are daily summed seed-mean losses.
    sh=daily["sh"].to_numpy(float)
    sc=daily["sc"].to_numpy(float)
    n=daily["n"].to_numpy(float)
    th=sh.sum()/n.sum(); tc=sc.sum()/n.sum()
    if metric=="RMSE":
        mh=math.sqrt(th); mc=math.sqrt(tc)
        bh=np.sqrt((Wt@sh)/(Wt@n)); bc=np.sqrt((Wt@sc)/(Wt@n))
    else:
        mh=th; mc=tc
        bh=(Wt@sh)/(Wt@n); bc=(Wt@sc)/(Wt@n)
    delta=mh-mc
    bd=bh-bc
    lo,hi=np.quantile(bd,[0.025,0.975])
    # DM-style daily differential uses daily mean underlying loss.
    dday=sh/n-sc/n
    stat,p,se=nw_hac_p(dday,HAC_LAG,H//24)
    return {
        "history_metric":mh,"cold_metric":mc,"delta_history_minus_cold":delta,
        "bootstrap_ci_low":float(lo),"bootstrap_ci_high":float(hi),
        "bootstrap_reps":N_BOOT,"bootstrap_block_days":BLOCK,
        "dm_hln_stat":stat,"dm_p_raw":p,"dm_daily_n":len(dday),
        "dm_hac_lag_days":HAC_LAG,"dm_h_days":H//24,
        "ci_excludes_zero":bool(lo>0 or hi<0)
    }

def main():
    shutil.rmtree(OUT,ignore_errors=True)
    OUT.mkdir(parents=True)
    freeze_report=FREEZE/"zero_shot_freeze_report_v1.json"
    manifest_path=FREEZE/"zero_shot_freeze_manifest_v1.json"
    if not freeze_report.exists() or not manifest_path.exists():
        raise SystemExit("Freeze outputs missing. Run and audit 00_freeze_zero_shot.sh first.")
    rep=json.loads(freeze_report.read_text())
    if rep.get("status")!="PASS":
        raise SystemExit("Freeze report is not PASS.")
    manifest=json.loads(manifest_path.read_text())
    # Strong immutability gate: rehash every frozen file.
    changed=[]
    for s,h in manifest["files"].items():
        p=Path(s)
        if not p.exists() or sha256(p)!=h:
            changed.append(s)
    if changed:
        (OUT/"changed_after_freeze.json").write_text(json.dumps(changed,indent=2))
        raise SystemExit(f"{len(changed)} frozen files changed/missing after freeze.")

    results=[]
    lead_rows=[]
    pairing_fail=[]

    for H in HORIZONS:
        # Gather origin-date ordered master grid from first city metadata.
        m0=pd.read_csv(BASE/f"H{H}"/"history_aware"/"test"/"cities"/CITIES[0]/"test_window_metadata.csv.gz")
        dates=pd.to_datetime(m0.forecast_origin,utc=True).dt.strftime("%Y-%m-%d").to_numpy()
        udates=np.array(sorted(pd.unique(dates)))
        date_to_i={d:i for i,d in enumerate(udates)}
        Wt=make_boot_weights(len(udates),BOOT_SEED+H)

        # Accumulators keyed daytype/metric/scope.
        acc={}
        for daytype in ("daylight","all_hours"):
            for metric in ("RMSE","MAE"):
                for scope in ["pooled"]+CITIES:
                    acc[(daytype,metric,scope)]={"sh":np.zeros(len(udates)), "sc":np.zeros(len(udates)), "n":np.zeros(len(udates))}

        # Per-lead descriptive accumulators.
        lead_acc={(daytype,metric,lead):[0.0,0.0,0] for daytype in ("daylight","all_hours")
                  for metric in ("RMSE","MAE") for lead in range(1,H+1)}

        for city in CITIES:
            meta_h=pd.read_csv(BASE/f"H{H}"/"history_aware"/"test"/"cities"/city/"test_window_metadata.csv.gz")
            meta_c=pd.read_csv(BASE/f"H{H}"/"cold_start"/"test"/"cities"/city/"test_window_metadata.csv.gz")
            cols=["start_index","forecast_origin","first_valid_time"]
            if not meta_h[cols].equals(meta_c[cols]):
                pairing_fail.append(f"H{H} {city}: metadata mismatch")
                continue
            city_dates=pd.to_datetime(meta_h.forecast_origin,utc=True).dt.strftime("%Y-%m-%d").to_numpy()
            day_idx=np.array([date_to_i[d] for d in city_dates],dtype=np.int64)

            tr={}
            for track in TRACKS:
                sq=[]; ab=[]; true0=None; day0=None; start0=None; time0=None
                for seed in SEEDS:
                    z=np.load(BASE/f"H{H}"/track/"test"/"cities"/city/f"seed_{seed}"/"test_predictions.npz",allow_pickle=False)
                    e=z["pred"].astype(np.float64)-z["true"].astype(np.float64)
                    sq.append(e*e); ab.append(np.abs(e))
                    if true0 is None:
                        true0=z["true"]; day0=z["daylight"]; start0=z["window_start_idx"]; time0=z["series_time_ns"]
                    else:
                        if not np.array_equal(true0,z["true"]) or not np.array_equal(day0,z["daylight"]):
                            pairing_fail.append(f"H{H} {city} {track}: truth/day differs across seeds")
                tr[track]={"sq":np.mean(np.stack(sq),axis=0),
                           "ab":np.mean(np.stack(ab),axis=0),
                           "day":day0.astype(bool),
                           "true":true0,
                           "start":start0,"time":time0}
            if not np.array_equal(tr["history_aware"]["true"],tr["cold_start"]["true"]):
                pairing_fail.append(f"H{H} {city}: truth mismatch H/C")
            if not np.array_equal(tr["history_aware"]["day"],tr["cold_start"]["day"]):
                pairing_fail.append(f"H{H} {city}: daylight mismatch H/C")

            for daytype in ("daylight","all_hours"):
                mask = tr["history_aware"]["day"] if daytype=="daylight" else np.ones_like(tr["history_aware"]["day"],dtype=bool)
                for metric,field in (("RMSE","sq"),("MAE","ab")):
                    lh=tr["history_aware"][field]; lc=tr["cold_start"][field]
                    # Lead descriptive.
                    for lead in range(H):
                        ml=mask[:,lead]
                        a=lead_acc[(daytype,metric,lead+1)]
                        a[0]+=float(lh[:,lead][ml].sum())
                        a[1]+=float(lc[:,lead][ml].sum())
                        a[2]+=int(ml.sum())
                    # Daily sums by origin day. Count is paired residual count.
                    for di in np.unique(day_idx):
                        wm=(day_idx==di)[:,None] & mask
                        n=int(wm.sum())
                        if n==0: continue
                        sh=float(lh[wm].sum()); sc=float(lc[wm].sum())
                        for scope in ("pooled",city):
                            A=acc[(daytype,metric,scope)]
                            A["sh"][di]+=sh; A["sc"][di]+=sc; A["n"][di]+=n

        if pairing_fail:
            (OUT/"pairing_failures.json").write_text(json.dumps(pairing_fail,indent=2))
            raise SystemExit(f"Pairing failed: {len(pairing_fail)} issues.")

        for (daytype,metric,scope),A in acc.items():
            keep=A["n"]>0
            daily=pd.DataFrame({"date":udates[keep],"sh":A["sh"][keep],"sc":A["sc"][keep],"n":A["n"][keep]})
            # Bootstrap weights must correspond to kept dates; in practice all dates should be present.
            if not np.all(keep):
                Wuse=make_boot_weights(int(keep.sum()),BOOT_SEED+H)
            else:
                Wuse=Wt
            r=summarize_scope(daily,metric,Wuse,H)
            r.update({"family":"history_vs_coldstart","horizon":H,"daytype":daytype,
                      "metric":metric,"scope_type":"pooled" if scope=="pooled" else "city",
                      "scope":scope,"effect_sign":"history_minus_cold"})
            results.append(r)

        for (daytype,metric,lead),a in lead_acc.items():
            sh,sc,n=a
            if metric=="RMSE":
                mh=math.sqrt(sh/n); mc=math.sqrt(sc/n)
            else:
                mh=sh/n; mc=sc/n
            lead_rows.append({"horizon":H,"daytype":daytype,"metric":metric,"lead":lead,
                              "history_metric":mh,"cold_metric":mc,
                              "delta_history_minus_cold":mh-mc,"n":n})

    df=pd.DataFrame(results)
    # Holm: pooled singleton; city across 15 within H × daytype × metric.
    df["dm_p_holm"]=np.nan
    for (H,daytype,metric,stype),g in df.groupby(["horizon","daytype","metric","scope_type"],sort=False):
        idx=g.index.to_numpy()
        if stype=="pooled":
            df.loc[idx,"dm_p_holm"]=g["dm_p_raw"].to_numpy()
        else:
            df.loc[idx,"dm_p_holm"]=holm(g["dm_p_raw"].to_numpy())
    df["strong_support"] = df["ci_excludes_zero"] & (df["dm_p_holm"]<0.05)
    df["favored_by_point_estimate"]=np.where(df["delta_history_minus_cold"]<0,"history_aware",
                                      np.where(df["delta_history_minus_cold"]>0,"cold_start","tie"))
    df.to_csv(OUT/"paired_stats_all_v1.csv",index=False)
    df[df.scope_type=="pooled"].to_csv(OUT/"paired_stats_pooled_v1.csv",index=False)
    df[df.scope_type=="city"].to_csv(OUT/"paired_stats_by_city_v1.csv",index=False)
    pd.DataFrame(lead_rows).to_csv(OUT/"paired_per_lead_descriptive_v1.csv",index=False)

    summary={
        "status":"PASS",
        "n_tests":len(df),
        "n_pooled_tests":int((df.scope_type=="pooled").sum()),
        "n_city_tests":int((df.scope_type=="city").sum()),
        "bootstrap_reps":N_BOOT,"block_days":BLOCK,"seed":BOOT_SEED,
        "hac_lag_days":HAC_LAG,"holm_city_family_size":15,
        "pairing_failures":0,
        "frozen_files_rehashed":len(manifest["files"]),
        "changed_after_freeze":0,
        "adaptation_run":False
    }
    (OUT/"paired_stats_report_v1.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
