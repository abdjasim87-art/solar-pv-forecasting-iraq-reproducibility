#!/usr/bin/env python3
from __future__ import annotations
import json, hashlib, math, shutil
from pathlib import Path
import numpy as np
import pandas as pd

W = Path("/workspace/lstm")
BASE = W/"output"/"loco_transfer_v2"
OUT = W/"output"/"loco_zero_shot_freeze_v1"
CITIES = ["Anbar","Babylon","Baghdad","Basra","Dhi_Qar","Diyala","Karbala",
          "Kirkuk","Maysan","Muthanna","Najaf","Nineveh","Qadisiyyah",
          "Salah_al_Din","Wasit"]
SEEDS=(1,2,3)
HORIZONS=(24,48)
TRACKS=("history_aware","cold_start")
EXPECTED = {
    24: {"windows_city":17185, "windows_pool":257775, "residuals":6186600, "daylight":3103207},
    48: {"windows_city":17161, "windows_pool":257415, "residuals":12355920, "daylight":6199242},
}

def sha256(p: Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def canon_digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def np_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def find_one(root: Path, names):
    for n in names:
        p=root/n
        if p.exists(): return p
    return None

def main():
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    failures=[]
    manifest={"version":"1.0","files":{},"pairing":{}}
    summary_rows=[]
    pairing_rows=[]
    test_shas={}
    n_npz=0

    # Load protocol lock from package copied into /workspace/lstm.
    protocol_path=W/"PAIRED_STATS_PROTOCOL_LOCK_v1.json"
    if not protocol_path.exists():
        failures.append("Missing PAIRED_STATS_PROTOCOL_LOCK_v1.json in /workspace/lstm")
    else:
        manifest["protocol_lock_sha256"]=sha256(protocol_path)
        manifest["files"][str(protocol_path)] = sha256(protocol_path)

    for H in HORIZONS:
        per_track={}
        for track in TRACKS:
            tdir=BASE/f"H{H}"/track/"test"
            if not tdir.exists():
                failures.append(f"Missing test dir {tdir}")
                continue
            per_track[track]=tdir

            audit=find_one(tdir,["test_audit.json","zero_shot_test_audit.json"])
            if audit:
                manifest["files"][str(audit)] = sha256(audit)
                try:
                    ad=json.loads(audit.read_text())
                    if ad.get("selection_used_test") not in (False,None):
                        failures.append(f"H{H} {track}: selection_used_test not false")
                    if ad.get("horizon") not in (H,str(H),None):
                        failures.append(f"H{H} {track}: audit horizon mismatch")
                except Exception as e:
                    failures.append(f"H{H} {track}: audit parse error {e}")

            pool=find_one(tdir,["zero_shot_pooled_test_seed_results.csv"])
            summ=find_one(tdir,["zero_shot_pooled_test_summary_ddof1.csv"])
            energy=find_one(tdir,["zero_shot_aggregate_energy_by_seed.csv"])
            for p in [pool,summ,energy]:
                if p:
                    manifest["files"][str(p)]=sha256(p)

            if pool:
                df=pd.read_csv(pool)
                if len(df)!=3 or sorted(df.seed.tolist())!=[1,2,3]:
                    failures.append(f"H{H} {track}: pooled seed table invalid")
                for _,r in df.iterrows():
                    summary_rows.append({
                        "horizon":H,"track":track,"seed":int(r.seed),
                        **{c:r[c] for c in df.columns if c!="seed"}
                    })

        if len(per_track)!=2:
            continue

        # Pairing and raw-file freeze.
        for city in CITIES:
            metas={}
            for track in TRACKS:
                mp=per_track[track]/"cities"/city/"test_window_metadata.csv.gz"
                if not mp.exists():
                    failures.append(f"H{H} {track} {city}: missing metadata")
                    continue
                manifest["files"][str(mp)] = sha256(mp)
                m=pd.read_csv(mp)
                metas[track]=m
                if len(m)!=EXPECTED[H]["windows_city"]:
                    failures.append(f"H{H} {track} {city}: metadata windows {len(m)}")
            if len(metas)==2:
                cols=["start_index","forecast_origin","first_valid_time"]
                if not metas["history_aware"][cols].equals(metas["cold_start"][cols]):
                    failures.append(f"H{H} {city}: History/Cold metadata grid mismatch")
                pairing_rows.append({
                    "horizon":H,"city":city,
                    "n_windows":len(metas["history_aware"]),
                    "metadata_exact_match":metas["history_aware"][cols].equals(metas["cold_start"][cols])
                })

            # Canonical truth/daylight/time comes from history seed1.
            canonical=None
            for track in TRACKS:
                for seed in SEEDS:
                    p=per_track[track]/"cities"/city/f"seed_{seed}"/"test_predictions.npz"
                    if not p.exists():
                        failures.append(f"H{H} {track} {city} seed{seed}: missing NPZ")
                        continue
                    n_npz += 1
                    manifest["files"][str(p)] = sha256(p)
                    z=np.load(p,allow_pickle=False)
                    for key in ("pred","true","daylight","window_start_idx","series_time_ns"):
                        if key not in z.files:
                            failures.append(f"H{H} {track} {city} seed{seed}: missing key {key}")
                    if "pred" not in z.files: continue
                    if z["pred"].shape != (EXPECTED[H]["windows_city"],H):
                        failures.append(f"H{H} {track} {city} seed{seed}: pred shape {z['pred'].shape}")
                    sig={
                        "true":np_hash(z["true"]),
                        "daylight":np_hash(z["daylight"]),
                        "start":np_hash(z["window_start_idx"]),
                        "time":np_hash(z["series_time_ns"]),
                    }
                    if canonical is None:
                        canonical=sig
                    else:
                        for k in sig:
                            if sig[k]!=canonical[k]:
                                failures.append(f"H{H} {city}: {k} differs across track/seed")
            manifest["pairing"][f"H{H}/{city}"]=canonical

        # Test SHA consistency if audit exposes it.
        for track in TRACKS:
            audit=find_one(per_track[track],["test_audit.json","zero_shot_test_audit.json"])
            if audit:
                try:
                    ad=json.loads(audit.read_text())
                    tsha=ad.get("test_sha256") or ad.get("test_sha")
                    if tsha: test_shas[f"H{H}/{track}"]=tsha
                except: pass
        a=test_shas.get(f"H{H}/history_aware")
        b=test_shas.get(f"H{H}/cold_start")
        if a and b and a!=b:
            failures.append(f"H{H}: test SHA differs between tracks")

    if n_npz != 180:
        failures.append(f"Expected 180 NPZ files, found {n_npz}")

    pd.DataFrame(pairing_rows).to_csv(OUT/"zero_shot_pairing_audit.csv",index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(OUT/"zero_shot_seed_metrics_frozen.csv",index=False)

    manifest["n_npz"]=n_npz
    manifest["test_shas"]=test_shas
    manifest["manifest_digest_sha256"]=canon_digest({k:v for k,v in manifest.items() if k!="manifest_digest_sha256"})
    (OUT/"zero_shot_freeze_manifest_v1.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    report={
        "status":"PASS" if not failures else "FAIL",
        "n_failures":len(failures),
        "failures":failures,
        "n_npz_frozen":n_npz,
        "expected_npz":180,
        "horizons":[24,48],
        "tracks":["history_aware","cold_start"],
        "cities":15,
        "seeds":[1,2,3],
        "pairing_key":["city","forecast_origin","horizon_step"],
        "history_cold_metadata_exact_match_all": all(r["metadata_exact_match"] for r in pairing_rows) if pairing_rows else False,
        "manifest_digest_sha256":manifest["manifest_digest_sha256"],
        "statistics_not_run":True,
        "adaptation_not_run":True
    }
    (OUT/"zero_shot_freeze_report_v1.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if not failures else 2)

if __name__=="__main__":
    main()
