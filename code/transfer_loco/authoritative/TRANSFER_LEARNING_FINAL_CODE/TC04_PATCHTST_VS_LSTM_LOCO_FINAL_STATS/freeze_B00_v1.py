#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

W = Path(os.environ.get("LSTM_WORKSPACE", "/workspace/lstm")).resolve()
PBASE = W/"output"/"patchtst_loco_v1"
LBASE = W/"output"/"loco_transfer_v2"
OUT = W/"output"/"patchtst_lstm_loco_stats_freeze_v1"
PKG = W/"patchtst_lstm_loco_paired_stats_v1"
CITIES = ["Anbar","Babylon","Baghdad","Basra","Dhi_Qar","Diyala","Karbala","Kirkuk",
          "Maysan","Muthanna","Najaf","Nineveh","Qadisiyyah","Salah_al_Din","Wasit"]
SEEDS=(1,2,3)
HORIZONS=(24,48)

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def req(c,msg):
    if not c:
        raise RuntimeError(msg)

def load(p: Path):
    return np.load(p,allow_pickle=False)

def patch_path(H,city,seed):
    return PBASE/f"H{H}"/"test"/"raw_predictions"/f"{city}_H{H}_seed{seed}.npz"

def lstm_path(H,city,seed):
    return LBASE/f"H{H}"/"history_aware"/"test"/"cities"/city/f"seed_{seed}"/"test_predictions.npz"

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    failures=[]
    files={}
    pairing=[]

    # Freeze analysis code/protocol before statistics.
    for name in ("PAIRED_STATS_PROTOCOL_v1.md","paired_stats_v1.py"):
        p=PKG/name
        req(p.is_file(),f"Missing package file: {p}")
        files[str(p)]={"bytes":p.stat().st_size,"sha256":sha256(p)}

    for H in HORIZONS:
        for city in CITIES:
            for seed in SEEDS:
                pp=patch_path(H,city,seed)
                lp=lstm_path(H,city,seed)
                req(pp.is_file(),f"Missing PatchTST NPZ: {pp}")
                req(lp.is_file(),f"Missing LSTM NPZ: {lp}")
                files[str(pp)]={"bytes":pp.stat().st_size,"sha256":sha256(pp)}
                files[str(lp)]={"bytes":lp.stat().st_size,"sha256":sha256(lp)}

                p=load(pp); l=load(lp)
                preq=("pred","true","daylight","window_start_idx","series_time_ns",
                      "origin_ns","valid_ns","horizon","lookback","seed","city")
                lreq=("pred","true","daylight","window_start_idx","series_time_ns",
                      "horizon","lookback","seed","city")
                for k in preq: req(k in p.files,f"{pp}: missing {k}")
                for k in lreq: req(k in l.files,f"{lp}: missing {k}")

                req(int(p["horizon"])==H and int(l["horizon"])==H,f"Horizon mismatch H{H} {city} s{seed}")
                req(int(p["lookback"])==168,f"PatchTST lookback mismatch H{H} {city} s{seed}")
                req(int(l["lookback"])==24,f"LSTM lookback mismatch H{H} {city} s{seed}")
                req(int(p["seed"])==seed and int(l["seed"])==seed,f"Seed mismatch H{H} {city} s{seed}")
                req(str(p["city"])==city and str(l["city"])==city,f"City mismatch H{H} {city} s{seed}")
                req(p["pred"].shape==l["pred"].shape,f"Prediction shape mismatch H{H} {city} s{seed}")
                req(np.array_equal(p["true"],l["true"]),f"Truth mismatch H{H} {city} s{seed}")
                req(np.array_equal(p["daylight"].astype(bool),l["daylight"].astype(bool)),
                    f"Daylight mismatch H{H} {city} s{seed}")
                req(np.array_equal(p["window_start_idx"],l["window_start_idx"]),
                    f"Start-index mismatch H{H} {city} s{seed}")
                req(np.array_equal(p["series_time_ns"],l["series_time_ns"]),
                    f"Series-time mismatch H{H} {city} s{seed}")

                start=l["window_start_idx"].astype(np.int64)
                st=l["series_time_ns"].astype(np.int64)
                origin=st[start-1]
                valid=np.stack([st[start+j] for j in range(H)],axis=1)
                req(np.array_equal(p["origin_ns"],origin),f"Origin mismatch H{H} {city} s{seed}")
                req(np.array_equal(p["valid_ns"],valid),f"Valid-time mismatch H{H} {city} s{seed}")

                pairing.append({
                    "horizon":H,"city":city,"seed":seed,
                    "windows":int(p["true"].shape[0]),
                    "residuals":int(p["true"].size),
                    "daylight_residuals":int(p["daylight"].astype(bool).sum()),
                    "pairing_exact":True,
                })

    manifest={
        "status":"PASS",
        "stage":"B00",
        "purpose":"Freeze PatchTST-LOCO and LSTM-LOCO history-aware Test predictions and exact pairing before inferential statistics",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "statistics_run":False,
        "training_performed":False,
        "model_inference_performed":False,
        "raw_test_csv_opened":False,
        "horizons":[24,48],
        "cities":CITIES,
        "seeds":[1,2,3],
        "pairing_key":["city","forecast_origin","horizon_step"],
        "expected_pairing_rows":90,
        "exact_pairing_rows":len(pairing),
        "frozen_npz":180,
        "files":files,
        "pairing":pairing,
    }
    mp=OUT/"freeze_manifest_v1.json"
    mp.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    report={
        "status":"PASS",
        "stage":"B00",
        "stage_complete":True,
        "n_failures":0,
        "frozen_npz":180,
        "exact_pairing_rows":len(pairing),
        "expected_pairing_rows":90,
        "statistics_run":False,
        "training_performed":False,
        "model_inference_performed":False,
        "raw_test_csv_opened":False,
        "manifest_sha256":sha256(mp),
        "paired_stats_code_sha256":sha256(PKG/"paired_stats_v1.py"),
        "protocol_sha256":sha256(PKG/"PAIRED_STATS_PROTOCOL_v1.md"),
        "statistics_authorized_after_review":False,
    }
    rp=OUT/"B00_freeze_report_v1.json"
    rp.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    print("PATCHTST-LSTM LOCO B00 FREEZE PASS | statistics not run | test.csv not opened")

if __name__=="__main__":
    main()
