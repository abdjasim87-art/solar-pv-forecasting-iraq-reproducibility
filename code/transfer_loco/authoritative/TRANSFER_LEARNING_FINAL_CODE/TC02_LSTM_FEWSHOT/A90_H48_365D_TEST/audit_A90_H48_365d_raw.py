#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd

W=Path("/workspace/lstm")
ROOT=W/"output"/"loco_fewshot"/"H48"/"test"/"365d"
OUT=ROOT/"raw_audit"
ZERO=W/"output"/"loco_transfer_v2"/"H48"/"cold_start"/"test"
CITIES=["Anbar","Babylon","Baghdad","Basra","Dhi_Qar","Diyala","Karbala","Kirkuk","Maysan","Muthanna","Najaf","Nineveh","Qadisiyyah","Salah_al_Din","Wasit"]
SEEDS=[1,2,3]; H=48; REF=3370.0
EW=17161; EP=257415; ER=12355920; ED=6199242; EU=258120; TRUEE=171314.7648391358

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()
def met(pred,true,day):
 p=pred.astype(np.float64).ravel(); y=true.astype(np.float64).ravel(); d=day.astype(bool).ravel(); e=p-y
 out={"per_kwp_mae":np.mean(abs(e)),"per_kwp_rmse":np.sqrt(np.mean(e*e)),"per_kwp_r2":1-np.sum(e*e)/np.sum((y-y.mean())**2),"per_kwp_bias":np.mean(e),"per_kwp_count":len(y)}
 ed=e[d]; yd=y[d]
 out.update({"daylight_per_kwp_mae":np.mean(abs(ed)),"daylight_per_kwp_rmse":np.sqrt(np.mean(ed*ed)),"daylight_per_kwp_r2":1-np.sum(ed*ed)/np.sum((yd-yd.mean())**2),"daylight_per_kwp_bias":np.mean(ed),"daylight_per_kwp_count":len(yd)})
 return {k:(int(v) if "count" in k else float(v)) for k,v in out.items()}

def main():
 OUT.mkdir(parents=True,exist_ok=True); fail=[]; rows=[]; energy=[]; inv=[]; pairing=[]
 saved=pd.read_csv(ROOT/"adapted_pooled_test_seed_results.csv")
 saved_e=pd.read_csv(ROOT/"adapted_aggregate_energy_by_seed.csv")
 for seed in SEEDS:
  ps=[];ys=[];ds=[]
  for city in CITIES:
   pth=ROOT/"cities"/city/f"seed_{seed}"/"test_predictions.npz"
   z=np.load(pth,allow_pickle=False); zz=np.load(ZERO/"cities"/city/"seed_1"/"test_predictions.npz",allow_pickle=False)
   pred=z["pred"]; true=z["true"]; day=z["daylight"]; starts=z["window_start_idx"].astype(np.int64); times=z["series_time_ns"].astype(np.int64)
   inv.append({"city":city,"seed":seed,"sha256":sha(pth),"size_bytes":pth.stat().st_size})
   if pred.shape!=(EW,H): fail.append(f"{city} seed{seed}: shape")
   if not np.array_equal(starts,zz["window_start_idx"]) or not np.array_equal(times,zz["series_time_ns"]): fail.append(f"{city} seed{seed}: grid differs Stage80")
   if not np.array_equal(true,zz["true"]) or not np.array_equal(day,zz["daylight"]): fail.append(f"{city} seed{seed}: truth/day differs Stage80")
   if pred.min() < -1e-7 or pred.max()>1.2000001 or (np.any(~day.astype(bool)) and np.max(np.abs(pred[~day.astype(bool)]))>1e-7): fail.append(f"{city} seed{seed}: physical")
   valid=times[starts[:,None]+np.arange(H)[None,:]]
   f=pd.DataFrame({"t":valid.ravel(),"p":pred.astype(float).ravel(),"y":true.astype(float).ravel()}); u=f.groupby("t")[["p","y"]].mean()
   act=u.y.sum()*REF/1000; pr=u.p.sum()*REF/1000
   energy.append({"city":city,"seed":seed,"n":len(u),"actual":act,"pred":pr,"error":pr-act})
   ps.append(pred);ys.append(true);ds.append(day)
  pp=np.concatenate(ps); yy=np.concatenate(ys); dd=np.concatenate(ds); m=met(pp,yy,dd)
  if pp.shape!=(EP,H) or m["per_kwp_count"]!=ER or m["daylight_per_kwp_count"]!=ED: fail.append(f"seed{seed}: pooled counts")
  sr=saved[saved.seed==seed].iloc[0]
  for k,v in m.items():
   if k in sr and not np.isclose(v,sr[k],rtol=2e-7,atol=2e-8 if not k.startswith("reference") else 2e-5): fail.append(f"seed{seed}: metric {k}")
  rows.append({"seed":seed,**m})
 e=pd.DataFrame(energy); agg=e.groupby("seed").agg(actual=("actual","sum"),pred=("pred","sum"),error=("error","sum"),n=("n","sum")).reset_index()
 agg["pct"]=100*agg.error/agg.actual
 for _,r in agg.iterrows():
  if int(r.n)!=EU or not np.isclose(r.actual,TRUEE,rtol=0,atol=1e-6): fail.append(f"seed{int(r.seed)}: energy grid/true")
  sr=saved_e[saved_e.seed==int(r.seed)].iloc[0]
  for a,b in [("actual","actual_reference_energy_mwh"),("pred","predicted_reference_energy_mwh"),("error","reference_energy_error_mwh"),("pct","energy_error_pct")]:
   if not np.isclose(float(r[a]),float(sr[b]),rtol=1e-8,atol=1e-6): fail.append(f"seed{int(r.seed)}: energy {a}")
 pd.DataFrame(rows).to_csv(OUT/"raw_recomputed_pooled_metrics.csv",index=False)
 agg.to_csv(OUT/"raw_recomputed_energy.csv",index=False)
 pd.DataFrame(inv).to_csv(OUT/"raw_npz_inventory.csv",index=False)
 report={"status":"PASS" if not fail else "FAIL","n_failures":len(fail),"failures":fail,
  "npz_files_found":len(inv),"expected_npz_files":45,"pooled_windows_per_seed":EP,"pooled_residuals_per_seed":ER,
  "pooled_daylight_per_seed":ED,"unique_city_hours_per_seed":EU,"true_energy_mwh":TRUEE,
  "stage80_grid_truth_daylight_identity_checked":True,"raw_metrics_recomputed":True,"raw_energy_recomputed":True,"physical_bounds_checked":True}
 (OUT/"A90_raw_audit_report.json").write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
 raise SystemExit(0 if not fail else 2)
if __name__=="__main__": main()
