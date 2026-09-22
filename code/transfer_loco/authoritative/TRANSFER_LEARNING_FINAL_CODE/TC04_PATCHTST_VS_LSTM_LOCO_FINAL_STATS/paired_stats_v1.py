#!/usr/bin/env python3
from __future__ import annotations
import os, json, hashlib, math, shutil
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from scipy.stats import t as t_dist
except Exception:
    t_dist=None

W=Path(os.environ.get("LSTM_WORKSPACE","/workspace/lstm")).resolve()
PBASE=W/"output"/"patchtst_loco_v1"
LBASE=W/"output"/"loco_transfer_v2"
FREEZE=W/"output"/"patchtst_lstm_loco_stats_freeze_v1"
OUT=W/"output"/"patchtst_lstm_loco_paired_stats_v1"
CITIES=["Anbar","Babylon","Baghdad","Basra","Dhi_Qar","Diyala","Karbala","Kirkuk",
        "Maysan","Muthanna","Najaf","Nineveh","Qadisiyyah","Salah_al_Din","Wasit"]
SEEDS=(1,2,3); HORIZONS=(24,48)
N_BOOT=5000; BLOCK_DAYS=7; BOOT_SEED_BASE=20260806; HAC_LAG_DAYS=7

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def holm(pvals):
    p=np.asarray(pvals,float); n=len(p); order=np.argsort(p); adj=np.empty(n,float); running=0.0
    for rank,idx in enumerate(order):
        running=max(running,(n-rank)*p[idx]); adj[idx]=min(1.0,running)
    return adj

def nw_hac_p(d,lag,h_days):
    d=np.asarray(d,float); d=d[np.isfinite(d)]; T=len(d)
    if T<10: return np.nan,np.nan,np.nan
    m=float(d.mean()); x=d-m; gamma0=float(np.dot(x,x)/T); lrv=gamma0
    L=min(int(lag),T-1)
    for l in range(1,L+1):
        gam=float(np.dot(x[l:],x[:-l])/T); w=1.0-l/(L+1.0); lrv+=2*w*gam
    if not np.isfinite(lrv) or lrv<=0: return np.nan,np.nan,np.nan
    se=math.sqrt(lrv/T); stat=m/se; h=float(h_days)
    hln=math.sqrt(max((T+1-2*h+h*(h-1)/T)/T,0.0)); stat_hln=stat*hln
    p=2.0*float(t_dist.sf(abs(stat_hln),df=T-1)) if t_dist is not None else math.erfc(abs(stat_hln)/math.sqrt(2))
    return stat_hln,p,se

def make_boot_weights(T,seed):
    rng=np.random.default_rng(seed); nblocks=math.ceil(T/BLOCK_DAYS)
    starts=rng.integers(0,T,size=(N_BOOT,nblocks),endpoint=False)
    offsets=np.arange(BLOCK_DAYS)[None,None,:]
    idx=(starts[:,:,None]+offsets)%T; idx=idx.reshape(N_BOOT,-1)[:,:T]
    w=np.zeros((N_BOOT,T),dtype=np.uint16)
    rows=np.repeat(np.arange(N_BOOT),T); np.add.at(w,(rows,idx.ravel()),1)
    return w

def ppath(H,city,seed):
    return PBASE/f"H{H}"/"test"/"raw_predictions"/f"{city}_H{H}_seed{seed}.npz"
def lpath(H,city,seed):
    return LBASE/f"H{H}"/"history_aware"/"test"/"cities"/city/f"seed_{seed}"/"test_predictions.npz"

def load_method(path):
    z=np.load(path,allow_pickle=False)
    e=z["pred"].astype(np.float64)-z["true"].astype(np.float64)
    start=z["window_start_idx"].astype(np.int64); st=z["series_time_ns"].astype(np.int64)
    return {"sq":e*e,"ab":np.abs(e),"true":z["true"],"day":z["daylight"].astype(bool),
            "start":start,"time":st,"origin":st[start-1]}

def summarize(daily,metric,weights,H):
    sp=daily["sp"].to_numpy(float); sl=daily["sl"].to_numpy(float); n=daily["n"].to_numpy(float)
    tp=float(sp.sum()/n.sum()); tl=float(sl.sum()/n.sum())
    if metric=="RMSE":
        mp=math.sqrt(tp); ml=math.sqrt(tl)
        bp=np.sqrt((weights@sp)/(weights@n)); bl=np.sqrt((weights@sl)/(weights@n))
    else:
        mp=tp; ml=tl
        bp=(weights@sp)/(weights@n); bl=(weights@sl)/(weights@n)
    delta=mp-ml; bd=bp-bl; lo,hi=np.quantile(bd,[0.025,0.975])
    daily_diff=sp/n-sl/n; stat,p,se=nw_hac_p(daily_diff,HAC_LAG_DAYS,H//24)
    return {
        "patchtst_metric":mp,"lstm_metric":ml,"delta_patchtst_minus_lstm":delta,
        "improvement_pct_vs_lstm":(-delta/ml*100.0) if ml else np.nan,
        "bootstrap_ci_low":float(lo),"bootstrap_ci_high":float(hi),
        "bootstrap_reps":N_BOOT,"bootstrap_block_days":BLOCK_DAYS,
        "dm_hln_stat":stat,"dm_p_raw":p,"dm_daily_n":len(daily_diff),
        "dm_hac_lag_days":HAC_LAG_DAYS,"dm_h_days":H//24,
        "ci_excludes_zero":bool(lo>0 or hi<0),
    }

def main():
    rep=json.loads((FREEZE/"B00_freeze_report_v1.json").read_text())
    man=json.loads((FREEZE/"freeze_manifest_v1.json").read_text())
    if rep.get("status")!="PASS" or not rep.get("statistics_authorized_after_review",False):
        raise SystemExit("B00 not PASS/authorized. Review B00 and set authorization with 05_authorize_B10.sh.")
    changed=[]
    for pstr,meta in man["files"].items():
        p=Path(pstr)
        if not p.is_file() or p.stat().st_size!=meta["bytes"] or sha256(p)!=meta["sha256"]:
            changed.append(pstr)
    if changed:
        raise SystemExit(f"{len(changed)} frozen files changed/missing after B00 freeze.")
    shutil.rmtree(OUT,ignore_errors=True); OUT.mkdir(parents=True)

    results=[]; leads=[]; pairing_fail=[]
    for H in HORIZONS:
        # Master UTC-origin-day grid from the first city/seed.
        z0=np.load(ppath(H,CITIES[0],1),allow_pickle=False)
        origins=z0["origin_ns"].astype(np.int64)
        dates0=pd.to_datetime(origins,unit="ns",utc=True).strftime("%Y-%m-%d").to_numpy()
        udates=np.array(sorted(pd.unique(dates0)))
        date_to_i={d:i for i,d in enumerate(udates)}
        weights=make_boot_weights(len(udates),BOOT_SEED_BASE+H)

        acc={}
        for daytype in ("daylight","all_hours"):
            for metric in ("RMSE","MAE"):
                for scope in ["pooled"]+CITIES:
                    acc[(daytype,metric,scope)]={"sp":np.zeros(len(udates)),"sl":np.zeros(len(udates)),
                                                 "n":np.zeros(len(udates),dtype=np.int64)}
        lead_acc={(daytype,metric,lead):[0.0,0.0,0]
                  for daytype in ("daylight","all_hours") for metric in ("RMSE","MAE")
                  for lead in range(1,H+1)}

        for city in CITIES:
            P=[]; L=[]
            pref=lref=None
            for seed in SEEDS:
                p=load_method(ppath(H,city,seed)); l=load_method(lpath(H,city,seed))
                P.append(p); L.append(l)
                if pref is None: pref=p; lref=l
                else:
                    for k in ("true","day","start","time","origin"):
                        if not np.array_equal(pref[k],p[k]): pairing_fail.append(f"H{H} {city} PatchTST {k} differs across seeds")
                        if not np.array_equal(lref[k],l[k]): pairing_fail.append(f"H{H} {city} LSTM {k} differs across seeds")
                for k in ("true","day","start","time","origin"):
                    if not np.array_equal(p[k],l[k]): pairing_fail.append(f"H{H} {city} seed{seed} PatchTST/LSTM {k} mismatch")
            ploss={"sq":np.mean(np.stack([x["sq"] for x in P]),axis=0),
                   "ab":np.mean(np.stack([x["ab"] for x in P]),axis=0)}
            lloss={"sq":np.mean(np.stack([x["sq"] for x in L]),axis=0),
                   "ab":np.mean(np.stack([x["ab"] for x in L]),axis=0)}
            city_dates=pd.to_datetime(pref["origin"],unit="ns",utc=True).strftime("%Y-%m-%d").to_numpy()
            try:
                day_idx=np.array([date_to_i[d] for d in city_dates],dtype=np.int64)
            except KeyError as e:
                pairing_fail.append(f"H{H} {city} origin date missing from master grid: {e}"); continue

            for daytype in ("daylight","all_hours"):
                mask=pref["day"] if daytype=="daylight" else np.ones_like(pref["day"],dtype=bool)
                for metric,field in (("RMSE","sq"),("MAE","ab")):
                    lp=ploss[field]; ll=lloss[field]
                    for lead in range(H):
                        ml=mask[:,lead]; A=lead_acc[(daytype,metric,lead+1)]
                        A[0]+=float(lp[:,lead][ml].sum()); A[1]+=float(ll[:,lead][ml].sum()); A[2]+=int(ml.sum())
                    for di in np.unique(day_idx):
                        wm=(day_idx==di)[:,None] & mask; n=int(wm.sum())
                        if n==0: continue
                        sp=float(lp[wm].sum()); sl=float(ll[wm].sum())
                        for scope in ("pooled",city):
                            A=acc[(daytype,metric,scope)]
                            A["sp"][di]+=sp; A["sl"][di]+=sl; A["n"][di]+=n

        if pairing_fail:
            (OUT/"pairing_failures.json").write_text(json.dumps(pairing_fail,indent=2))
            raise SystemExit(f"Pairing failed: {len(pairing_fail)} issues.")

        for (daytype,metric,scope),A in acc.items():
            keep=A["n"]>0
            daily=pd.DataFrame({"date":udates[keep],"sp":A["sp"][keep],"sl":A["sl"][keep],"n":A["n"][keep]})
            wuse=weights if np.all(keep) else make_boot_weights(int(keep.sum()),BOOT_SEED_BASE+H)
            r=summarize(daily,metric,wuse,H)
            r.update({"comparison":"PatchTST_LOCO_history_aware_vs_LSTM_LOCO_history_aware",
                      "horizon":H,"daytype":daytype,"metric":metric,
                      "scope_type":"pooled" if scope=="pooled" else "city","scope":scope,
                      "effect_sign":"PatchTST-minus-LSTM; negative_favors_PatchTST"})
            results.append(r)

        for (daytype,metric,lead),A in lead_acc.items():
            sp,sl,n=A
            if n==0: continue
            if metric=="RMSE": mp=math.sqrt(sp/n); ml=math.sqrt(sl/n)
            else: mp=sp/n; ml=sl/n
            leads.append({"horizon":H,"daytype":daytype,"metric":metric,"lead":lead,
                          "patchtst_metric":mp,"lstm_metric":ml,"delta_patchtst_minus_lstm":mp-ml,
                          "improvement_pct_vs_lstm":(-(mp-ml)/ml*100.0) if ml else np.nan,
                          "n":n,"inferential":False})

    df=pd.DataFrame(results); df["dm_p_holm"]=np.nan; df["holm_family_size"]=np.nan
    for (H,daytype,metric,stype),g in df.groupby(["horizon","daytype","metric","scope_type"],sort=False):
        idx=g.index.to_numpy(); df.loc[idx,"dm_p_holm"]=holm(g["dm_p_raw"].to_numpy(float)); df.loc[idx,"holm_family_size"]=len(idx)
    df["strong_support"]=df["ci_excludes_zero"] & (df["dm_p_holm"]<0.05)
    df["favored_by_point_estimate"]=np.where(df["delta_patchtst_minus_lstm"]<0,"PatchTST",
                                      np.where(df["delta_patchtst_minus_lstm"]>0,"LSTM","tie"))
    df.to_csv(OUT/"paired_stats_all_v1.csv",index=False)
    df[df.scope_type=="pooled"].to_csv(OUT/"paired_stats_pooled_v1.csv",index=False)
    df[df.scope_type=="city"].to_csv(OUT/"paired_stats_by_city_v1.csv",index=False)
    pd.DataFrame(leads).to_csv(OUT/"paired_per_lead_descriptive_v1.csv",index=False)

    primary=df[(df.scope_type=="pooled")&(df.daytype=="daylight")].copy()
    summary={
        "status":"PASS","analysis":"PatchTST LOCO history-aware versus LSTM LOCO history-aware",
        "n_tests":int(len(df)),"n_pooled":int((df.scope_type=="pooled").sum()),
        "n_city":int((df.scope_type=="city").sum()),
        "primary_pooled_daylight_tests":int(len(primary)),
        "primary_strong_support":int(primary["strong_support"].sum()),
        "primary_strong_favor_patchtst":int(((primary.strong_support)&(primary.delta_patchtst_minus_lstm<0)).sum()),
        "primary_strong_favor_lstm":int(((primary.strong_support)&(primary.delta_patchtst_minus_lstm>0)).sum()),
        "bootstrap_reps":N_BOOT,"block_days":BLOCK_DAYS,"bootstrap_seed_base":BOOT_SEED_BASE,
        "hac_lag_days":HAC_LAG_DAYS,"pairing_failures":0,
        "frozen_files_rehashed":len(man["files"]),"changed_after_freeze":0,
        "training_performed":False,"model_inference_performed":False,"raw_test_csv_opened":False,
        "interpretation_caveat":"End-to-end transfer-system comparison; internal predictor representations differ, so do not claim a pure architecture-only causal effect."
    }
    (OUT/"paired_stats_report_v1.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    print("PATCHTST-LSTM LOCO B10 PAIRED STATS PASS")

if __name__=="__main__":
    main()
