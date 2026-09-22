#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, tarfile
from pathlib import Path
from datetime import datetime, timezone

CITIES = [
    "Anbar","Babylon","Baghdad","Basra","Dhi_Qar","Diyala","Karbala","Kirkuk",
    "Maysan","Muthanna","Najaf","Nineveh","Qadisiyyah","Salah_al_Din","Wasit"
]
SEEDS = [1,2,3]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def require(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)

def rel_to_workspace(path: Path, workspace: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except Exception:
        return str(path.resolve())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="/workspace/lstm")
    args = ap.parse_args()

    ws = Path(args.workspace).resolve()
    root = ws / "output" / "patchtst_loco_v1"
    freeze_dir = root / "selection_freeze_v1"
    freeze_dir.mkdir(parents=True, exist_ok=True)

    # Selection freeze MUST occur before any Test outputs exist.
    for h in (24,48):
        tdir = root / f"H{h}" / "test"
        if tdir.exists() and any(p.is_file() for p in tdir.rglob("*")):
            raise RuntimeError(f"Refusing freeze: Test output already exists under {tdir}")

    stage_locks = {}
    immutable_keys = [
        "train_sha256","validation_sha256","patchtst_reference_sha256","base_code_sha256",
        "loco_code_sha256","protocol_markdown_sha256","source_reference_sha256"
    ]
    all_files = set()
    checkpoint_count = 0

    for h in (24,48):
        sel = root / f"H{h}" / "selection"
        stage_path = sel / "stage_lock.json"
        require(stage_path.is_file(), f"Missing stage lock: {stage_path}")
        stage = load_json(stage_path)
        stage_locks[h] = stage

        require(stage.get("stage_complete") is True, f"H{h} stage_complete != true")
        require(stage.get("selection_used_test") is False, f"H{h} selection_used_test != false")
        require(stage.get("test_data_loaded") is False, f"H{h} test_data_loaded != false")
        require(int(stage.get("horizon",-1)) == h, f"H{h} horizon mismatch")
        require(int(stage.get("lookback",-1)) == 168, f"H{h} lookback must be 168")
        require(stage.get("seeds") == SEEDS, f"H{h} seeds mismatch")
        require(int(stage.get("n_folds",-1)) == 15, f"H{h} n_folds mismatch")
        require(int(stage.get("source_training_runs",-1)) == 45, f"H{h} source_training_runs mismatch")
        require(stage.get("cities") == CITIES, f"H{h} city order/list mismatch")
        require(len(stage.get("folds",[])) == 15, f"H{h} fold lock count mismatch")

        all_files.add(stage_path)

        stage_fold_map = {x["target_city"]: x for x in stage["folds"]}
        require(set(stage_fold_map) == set(CITIES), f"H{h} stage fold cities mismatch")

        for city in CITIES:
            fd = sel / "folds" / city
            fold_path = fd / "fold_lock.json"
            require(fold_path.is_file(), f"Missing fold lock: {fold_path}")
            fl = load_json(fold_path)

            require(fl.get("selection_complete") is True, f"H{h}/{city} selection incomplete")
            require(fl.get("selection_used_test") is False, f"H{h}/{city} selection_used_test != false")
            require(fl.get("test_data_loaded") is False, f"H{h}/{city} test_data_loaded != false")
            require(fl.get("target_city") == city, f"H{h}/{city} target_city mismatch")
            require(int(fl.get("horizon",-1)) == h, f"H{h}/{city} horizon mismatch")
            require(int(fl.get("lookback",-1)) == 168, f"H{h}/{city} lookback mismatch")
            require(fl.get("seeds") == SEEDS, f"H{h}/{city} seeds mismatch")
            require(fl.get("all_seeds_retained") is True, f"H{h}/{city} all_seeds_retained != true")
            require(int(fl.get("sample_sd_ddof",-1)) == 1, f"H{h}/{city} ddof mismatch")
            require(fl.get("patchtst_input_mode") == "S", f"H{h}/{city} input mode mismatch")
            require(int(fl.get("strict_alignment_feature_count",-1)) == 53, f"H{h}/{city} feature count mismatch")
            require(int(fl.get("n_source_cities",-1)) == 14, f"H{h}/{city} source count mismatch")
            src = fl.get("source_cities",[])
            require(len(src) == 14 and len(set(src)) == 14 and city not in src,
                    f"H{h}/{city} target city leakage in source_cities")

            expected_fold_sha = stage_fold_map[city]["fold_lock_sha256"]
            require(sha256(fold_path) == expected_fold_sha, f"H{h}/{city} fold_lock SHA mismatch")
            all_files.add(fold_path)

            scaler_path = fd / "source_target_scaler.json"
            require(scaler_path.is_file(), f"Missing scaler: {scaler_path}")
            scaler = load_json(scaler_path)
            require(scaler.get("fit_scope") == "14-city source Training only",
                    f"H{h}/{city} scaler scope mismatch")
            require(scaler.get("target_city_excluded") == city,
                    f"H{h}/{city} scaler target exclusion mismatch")
            require(sha256(scaler_path) == fl["source_target_scaler_sha256"],
                    f"H{h}/{city} scaler SHA mismatch")
            all_files.add(scaler_path)

            feat_path = fd / "strict_alignment_feature_cols.json"
            require(feat_path.is_file(), f"Missing feature list: {feat_path}")
            feat = load_json(feat_path)
            require(isinstance(feat,list) and len(feat)==53, f"H{h}/{city} feature list mismatch")
            all_files.add(feat_path)

            seed_map = {int(x["seed"]): x for x in fl.get("seed_artifacts",[])}
            require(set(seed_map) == set(SEEDS), f"H{h}/{city} seed artifact list mismatch")
            for seed in SEEDS:
                sa = seed_map[seed]
                sd = fd / f"seed_{seed}"
                checkpoint = sd / "best_checkpoint.pth"
                correction = sd / "correction.json"
                valres = sd / "validation_seed_result.csv"
                for p in (checkpoint, correction, valres):
                    require(p.is_file(), f"Missing frozen artifact: {p}")
                    all_files.add(p)
                require(sha256(checkpoint) == sa["checkpoint_sha256"],
                        f"H{h}/{city}/seed{seed} checkpoint SHA mismatch")
                require(sha256(correction) == sa["correction_sha256"],
                        f"H{h}/{city}/seed{seed} correction SHA mismatch")
                require(sha256(valres) == sa["validation_seed_result_sha256"],
                        f"H{h}/{city}/seed{seed} validation result SHA mismatch")
                require(sa.get("selected_correction_method") == "horizon_daylight_bias",
                        f"H{h}/{city}/seed{seed} correction method mismatch")
                checkpoint_count += 1

            # Freeze every other selection artifact too (histories, calibration tables, summaries, inventory).
            for p in fd.rglob("*"):
                if p.is_file():
                    all_files.add(p)

        for p in sel.glob("*"):
            if p.is_file():
                all_files.add(p)

    # Both horizons must share the same immutable protocol/data/code hashes.
    for key in immutable_keys:
        require(stage_locks[24].get(key) == stage_locks[48].get(key),
                f"H24/H48 immutable hash mismatch: {key}")

    records = []
    total_bytes = 0
    for p in sorted(all_files, key=lambda x: str(x)):
        b = p.stat().st_size
        total_bytes += b
        records.append({
            "path": rel_to_workspace(p, ws),
            "bytes": b,
            "sha256": sha256(p),
        })

    manifest = {
        "status": "PASS",
        "freeze_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Freeze PatchTST LOCO H24/H48 selection before first Test access",
        "test_file_read": False,
        "test_output_present_before_freeze": False,
        "horizons": [24,48],
        "lookback": 168,
        "seeds": SEEDS,
        "n_folds_per_horizon": 15,
        "source_training_runs_total": 90,
        "checkpoint_count": checkpoint_count,
        "selection_file_count": len(records),
        "selection_total_bytes": total_bytes,
        "immutable_hashes": {k: stage_locks[24][k] for k in immutable_keys},
        "files": records,
    }

    manifest_path = freeze_dir / "freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_sha = sha256(manifest_path)

    report = {
        "status": "PASS",
        "freeze_manifest": str(manifest_path),
        "freeze_manifest_sha256": manifest_sha,
        "checkpoint_count": checkpoint_count,
        "selection_file_count": len(records),
        "test_file_read": False,
        "message": "H24/H48 selection frozen. Test may be opened only through freeze-gated wrappers."
    }
    report_path = freeze_dir / "freeze_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("PATCHTST LOCO SELECTION FREEZE PASS")
    print(f"Frozen selection files : {len(records)}")
    print(f"Frozen checkpoints     : {checkpoint_count}")
    print(f"Manifest SHA256        : {manifest_sha}")
    print("Test file read         : NO")
    print(f"Manifest               : {manifest_path}")
    print(f"Report                 : {report_path}")

if __name__ == "__main__":
    main()
