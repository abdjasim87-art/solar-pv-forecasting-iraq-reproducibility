#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

PROTOCOL_VERSION = "1.0"
LOOKBACK = 168
HORIZONS = (24, 48)
SEEDS = (1, 2, 3)
N_FOLDS = 15
REFERENCE_CAPACITY_KWP = 3370.0
EXPECTED_TRUE_ENERGY_GWH = 171.31476483913576
EXPECTED_REF_SHA = "769f9949eff6c4556e42bb5e19066f0732f80b123c5e4ef9ecd5163e971189ec"
EXPECTED_BASE_SHA = "41f7892cbdd296f6ea4d6887a5dd92669cd3a1a42ef0fb4695c29e7d79369438"
EXPECTED_TRAIN_SHA = "bd9fce556216c806ce812f05d0556ad84f9ccc3453f7f430e99df9013d73db06"
EXPECTED_VAL_SHA = "86f95e8581b864e2cc6d67640bbcd802fdf7176716b5119aa721d1a64369ca5a"
EXPECTED_TEST_SHA = "b0a3f938359eb1d5da615d0db5f34333b764217d2ce41454f5ff220b6ac9a222"
EXPECTED_SOURCE_PACKAGE_SHA = "da305fe26567a03449bb7d93ae277b91010564a3d65efdc8297ae7ca3f1f5cfa"

EXPECTED_WINDOWS_PER_CITY = {24: 17185, 48: 17161}
EXPECTED_POOLED = {
    24: {"windows": 257775, "residuals": 6186600, "daylight": 3103207, "unique_city_hours": 258120},
    48: {"windows": 257415, "residuals": 12355920, "daylight": 6199242, "unique_city_hours": 258120},
}
EXPECTED_SOURCE_TRAIN_WINDOWS = {24: 61009 * 14, 48: 60985 * 14}
EXPECTED_SOURCE_VAL_WINDOWS = {24: 17161 * 14, 48: 17137 * 14}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_modules(args):
    ref_path = Path(args.patchtst_reference).resolve()
    base_path = Path(args.base_code).resolve()
    if sha256(ref_path) != EXPECTED_REF_SHA:
        raise RuntimeError("Audited PatchTST reference code SHA mismatch")
    if sha256(base_path) != EXPECTED_BASE_SHA:
        raise RuntimeError("Authoritative preprocessing code SHA mismatch")
    ref = import_module_from_path("patchtst_original_reference_for_loco", ref_path)
    base = import_module_from_path("solar_base_for_patchtst_loco", base_path)
    return ref, base, {
        "patchtst_reference_sha256": sha256(ref_path),
        "base_code_sha256": sha256(base_path),
        "loco_code_sha256": sha256(Path(__file__).resolve()),
        "protocol_markdown_sha256": sha256(Path(args.protocol_markdown).resolve()),
        "source_reference_sha256": sha256(Path(args.source_reference).resolve()),
    }


def prepare_unscaled_splits(base, train_path: Path, val_path: Path):
    train_df, target_col = base.load_split(str(train_path), "train", "auto")
    val_df, val_target = base.load_split(str(val_path), "valid", "auto")
    if target_col != val_target:
        raise RuntimeError("Train/Validation target mismatch")

    common_cols = sorted(set(train_df.columns) & set(val_df.columns))
    train_df = train_df[common_cols].copy()
    val_df = val_df[common_cols].copy()

    train_df = base.prepare_physical_target(train_df, target_col)
    val_df = base.prepare_physical_target(val_df, target_col)
    train_df = base.add_ar_solar_features(train_df, target_col)
    val_df = base.add_ar_solar_features(val_df, target_col)

    feature_cols, _ = base.build_feature_columns(train_df, feature_set="basic", use_solar_history=True)
    feature_cols = list(feature_cols)
    if len(feature_cols) != 53:
        raise RuntimeError(f"Expected 53 strict alignment features, got {len(feature_cols)}")

    required = feature_cols + [base.TARGET_VALUE_COL, base.RAW_TARGET_COL, base.DAYLIGHT_COL]
    for label, df in (("train", train_df), ("valid", val_df)):
        before = len(df)
        df.dropna(subset=required, inplace=True)
        dropped = before - len(df)
        if dropped != 168 * 15:
            raise RuntimeError(f"Unexpected {label} split-local history drop: {dropped}")

    ty = pd.to_datetime(train_df[base.TIME_COL], utc=True).dt.year
    vy = pd.to_datetime(val_df[base.TIME_COL], utc=True).dt.year
    if (int(ty.min()), int(ty.max())) != (2015, 2021):
        raise RuntimeError("Train years are not exactly 2015-2021")
    if (int(vy.min()), int(vy.max())) != (2022, 2023):
        raise RuntimeError("Validation years are not exactly 2022-2023")

    train_cities = sorted(train_df[base.CITY_COL].astype(str).unique().tolist())
    val_cities = sorted(val_df[base.CITY_COL].astype(str).unique().tolist())
    if train_cities != val_cities or len(train_cities) != 15:
        raise RuntimeError("Expected the same 15 cities in Train and Validation")
    return train_df, val_df, target_col, feature_cols, train_cities


def prepare_target_test(base, test_path: Path, target_col: str, feature_cols: Sequence[str], city: str):
    full, test_target = base.load_split(str(test_path), "test", "auto")
    if test_target != target_col:
        raise RuntimeError("Test target mismatch")
    full = full[full[base.CITY_COL].astype(str) == str(city)].copy()
    if full.empty:
        raise RuntimeError(f"Held-out city absent from Test: {city}")
    full = base.prepare_physical_target(full, target_col)
    full = base.add_ar_solar_features(full, target_col)
    missing = [c for c in feature_cols if c not in full.columns]
    if missing:
        raise RuntimeError(f"Test missing locked alignment columns for {city}: {missing}")
    required = list(feature_cols) + [base.TARGET_VALUE_COL, base.RAW_TARGET_COL, base.DAYLIGHT_COL]
    before = len(full)
    full.dropna(subset=required, inplace=True)
    if before - len(full) != 168:
        raise RuntimeError(f"Unexpected target Test history drop for {city}: {before-len(full)}")
    years = pd.to_datetime(full[base.TIME_COL], utc=True).dt.year
    if (int(years.min()), int(years.max())) != (2024, 2025):
        raise RuntimeError("Test years are not exactly 2024-2025")
    return full


def fit_source_scaler(base, source_train: pd.DataFrame, source_val: pd.DataFrame):
    scaler = StandardScaler()
    tr = source_train.copy()
    va = source_val.copy()
    tr[base.TARGET_SCALED_COL] = scaler.fit_transform(
        tr[[base.TARGET_VALUE_COL]].to_numpy(np.float32)
    ).reshape(-1)
    va[base.TARGET_SCALED_COL] = scaler.transform(
        va[[base.TARGET_VALUE_COL]].to_numpy(np.float32)
    ).reshape(-1)
    return tr, va, scaler


def metric_summary(seed_df: pd.DataFrame, ref) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for col in ref.METRICS:
        x = pd.to_numeric(seed_df[col], errors="coerce").to_numpy(float)
        row[col + "_mean"] = float(np.mean(x))
        row[col + "_sd"] = float(np.std(x, ddof=1))
    return row


def verify_preflight_lock(args, modules_hashes: Dict[str, str]) -> Dict[str, Any]:
    p = Path(args.preflight_lock).resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    pre = json.loads(p.read_text(encoding="utf-8"))
    if pre.get("status") != "PASS" or pre.get("test_file_read") is not False:
        raise RuntimeError("Preflight lock is invalid")
    for k, v in modules_hashes.items():
        if pre.get(k) != v:
            raise RuntimeError(f"Preflight code hash mismatch: {k}")
    if pre.get("train_sha256") != EXPECTED_TRAIN_SHA or pre.get("validation_sha256") != EXPECTED_VAL_SHA:
        raise RuntimeError("Preflight data hash mismatch")
    return pre


def preflight(args):
    ref, base, hashes = load_modules(args)
    train_path = Path(args.train_data).resolve()
    val_path = Path(args.val_data).resolve()
    if sha256(train_path) != EXPECTED_TRAIN_SHA:
        raise RuntimeError("Train data SHA differs from audited source")
    if sha256(val_path) != EXPECTED_VAL_SHA:
        raise RuntimeError("Validation data SHA differs from audited source")

    repo_audit = ref.preflight(Path(args.official_repo).resolve(), Path(args.base_code).resolve())
    train, val, target_col, feature_cols, cities = prepare_unscaled_splits(base, train_path, val_path)
    source_ref = json.loads(Path(args.source_reference).read_text(encoding="utf-8"))
    if source_ref.get("audited_final_global_H24_lookback_from_raw_npz") != LOOKBACK:
        raise RuntimeError("Source H24 lookback provenance mismatch")
    if source_ref.get("audited_final_global_H48_lookback_from_raw_npz") != LOOKBACK:
        raise RuntimeError("Source H48 lookback provenance mismatch")

    fold_checks = []
    for heldout in cities:
        source = [c for c in cities if c != heldout]
        st = train[train[base.CITY_COL].astype(str) != heldout]
        sv = val[val[base.CITY_COL].astype(str) != heldout]
        if len(source) != 14 or heldout in source:
            raise RuntimeError("Held-out fold construction failure")
        if sorted(st[base.CITY_COL].astype(str).unique()) != source:
            raise RuntimeError("Held-out Train leakage")
        if sorted(sv[base.CITY_COL].astype(str).unique()) != source:
            raise RuntimeError("Held-out Validation leakage")
        fold_checks.append({"target_city": heldout, "n_source_cities": 14, "source_cities": source})

    out = {
        "status": "PASS",
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "PatchTST LOCO history-aware zero-shot preflight",
        "test_file_read": False,
        "test_file_required": False,
        "history_aware_zero_shot": True,
        "pure_cold_start": False,
        "lookback_fixed": LOOKBACK,
        "horizons": list(HORIZONS),
        "seeds": list(SEEDS),
        "n_folds": N_FOLDS,
        "source_training_runs_total": N_FOLDS * len(HORIZONS) * len(SEEDS),
        "target_col": target_col,
        "strict_alignment_feature_count": len(feature_cols),
        "strict_alignment_feature_cols": feature_cols,
        "patchtst_input_mode": "S",
        "patchtst_input_channel": "strictly past normalized target history",
        "cities": cities,
        "fold_checks": fold_checks,
        "train_sha256": sha256(train_path),
        "validation_sha256": sha256(val_path),
        "audited_source_package_sha256": EXPECTED_SOURCE_PACKAGE_SHA,
        "official_model_preflight": repo_audit,
        **hashes,
    }
    json_dump(Path(args.out_dir).resolve() / "preflight_lock.json", out)
    print(json.dumps(out, indent=2))
    print("PATCHTST LOCO PREFLIGHT PASS | Test file was not read.")


def seed_paths(fold_dir: Path, seed: int):
    sdir = fold_dir / f"seed_{seed}"
    return {
        "dir": sdir,
        "checkpoint": sdir / "best_checkpoint.pth",
        "correction": sdir / "correction.json",
        "seed_result": sdir / "validation_seed_result.csv",
        "history": sdir / "training_history.csv",
        "calibration": sdir / "validation_calibration_candidates.csv",
    }


def load_completed_seed(paths: Dict[str, Path], seed: int, horizon: int) -> Optional[Dict[str, Any]]:
    req = (paths["checkpoint"], paths["correction"], paths["seed_result"])
    if not all(p.is_file() for p in req):
        return None
    df = pd.read_csv(paths["seed_result"])
    if len(df) != 1:
        return None
    row = df.iloc[0].to_dict()
    if int(row.get("seed", -1)) != seed or int(row.get("horizon", -1)) != horizon or int(row.get("lookback", -1)) != LOOKBACK:
        return None
    if row.get("checkpoint_sha256") != sha256(paths["checkpoint"]):
        return None
    if row.get("correction_sha256") != sha256(paths["correction"]):
        return None
    return row


def select_source(args):
    if args.horizon not in HORIZONS:
        raise ValueError("Invalid horizon")
    ref, base, hashes = load_modules(args)
    pre = verify_preflight_lock(args, hashes)

    train_path = Path(args.train_data).resolve()
    val_path = Path(args.val_data).resolve()
    if sha256(train_path) != EXPECTED_TRAIN_SHA or sha256(val_path) != EXPECTED_VAL_SHA:
        raise RuntimeError("Train/Validation changed after preflight")
    train, val, target_col, feature_cols, cities = prepare_unscaled_splits(base, train_path, val_path)
    if target_col != pre["target_col"] or cities != pre["cities"] or feature_cols != pre["strict_alignment_feature_cols"]:
        raise RuntimeError("Prepared data schema changed after preflight")

    repo = Path(args.official_repo).resolve()
    # Recheck official repo source/commit/worktree before training.
    ref.preflight(repo, Path(args.base_code).resolve())

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    all_fold_summary = []

    for heldout in cities:
        fold_dir = out / "folds" / heldout
        fold_dir.mkdir(parents=True, exist_ok=True)
        lock_path = fold_dir / "fold_lock.json"
        if args.resume and lock_path.is_file():
            old = json.loads(lock_path.read_text(encoding="utf-8"))
            if old.get("selection_complete") is True and old.get("selection_used_test") is False and int(old.get("horizon", -1)) == args.horizon:
                print(f"RESUME: frozen fold exists for {heldout}; will be revalidated")
                all_fold_summary.append(pd.read_csv(fold_dir / "fold_validation_summary_ddof1.csv"))
                continue

        source_train = train[train[base.CITY_COL].astype(str) != heldout].copy()
        source_val = val[val[base.CITY_COL].astype(str) != heldout].copy()
        source_cities = sorted(source_train[base.CITY_COL].astype(str).unique().tolist())
        if len(source_cities) != 14 or heldout in source_cities:
            raise RuntimeError(f"Held-out leakage in {heldout}")
        if source_cities != sorted(source_val[base.CITY_COL].astype(str).unique().tolist()):
            raise RuntimeError("Source Train/Validation city mismatch")

        tr, va, scaler = fit_source_scaler(base, source_train, source_val)
        y_mean = float(scaler.mean_[0])
        y_scale = float(scaler.scale_[0])
        train_ds = ref.SolarPatchTSTDataset(tr, base, LOOKBACK, args.horizon, common_start=None)
        val_ds = ref.SolarPatchTSTDataset(va, base, LOOKBACK, args.horizon, common_start=LOOKBACK)
        if len(train_ds) != EXPECTED_SOURCE_TRAIN_WINDOWS[args.horizon]:
            raise RuntimeError(f"Unexpected source Train windows for {heldout}: {len(train_ds)}")
        if len(val_ds) != EXPECTED_SOURCE_VAL_WINDOWS[args.horizon]:
            raise RuntimeError(f"Unexpected source Validation windows for {heldout}: {len(val_ds)}")

        scaler_record = {
            "type": "StandardScaler",
            "fit_scope": "14-city source Training only",
            "target_city_excluded": heldout,
            "mean": y_mean,
            "scale": y_scale,
        }
        json_dump(fold_dir / "source_target_scaler.json", scaler_record)
        json_dump(fold_dir / "strict_alignment_feature_cols.json", feature_cols)

        seed_rows = []
        for seed in SEEDS:
            paths = seed_paths(fold_dir, seed)
            paths["dir"].mkdir(parents=True, exist_ok=True)
            existing = load_completed_seed(paths, seed, args.horizon) if args.resume else None
            if existing is not None:
                print(f"RESUME {heldout} H{args.horizon} seed={seed}")
                seed_rows.append(existing)
                continue

            print(f"TRAIN {heldout} H{args.horizon} history-aware PatchTST seed={seed}")
            result = ref.train_one_seed(
                repo=repo,
                train_ds=train_ds,
                val_ds=val_ds,
                lookback=LOOKBACK,
                horizon=args.horizon,
                seed=seed,
                checkpoint=paths["checkpoint"],
                y_mean=y_mean,
                y_scale=y_scale,
            )
            json_dump(paths["correction"], result["correction_params"])
            pd.DataFrame(result["history"]).to_csv(paths["history"], index=False)
            pd.DataFrame(result["calibration"]).to_csv(paths["calibration"], index=False)
            row = {k: v for k, v in result.items() if k not in ("history", "calibration", "correction_params")}
            row["target_city"] = heldout
            row["source_city_count"] = 14
            row["checkpoint_sha256"] = sha256(paths["checkpoint"])
            row["correction_sha256"] = sha256(paths["correction"])
            pd.DataFrame([row]).to_csv(paths["seed_result"], index=False)
            seed_rows.append(row)

        seed_df = pd.DataFrame(seed_rows).sort_values("seed").reset_index(drop=True)
        if list(seed_df["seed"].astype(int)) != list(SEEDS):
            raise RuntimeError("All three seeds were not retained")
        seed_df.to_csv(fold_dir / "fold_validation_seed_results.csv", index=False)
        summary = {
            "target_city": heldout,
            "horizon": args.horizon,
            "lookback": LOOKBACK,
            "n_source_cities": 14,
            "train_windows": len(train_ds),
            "validation_windows": len(val_ds),
            **metric_summary(seed_df, ref),
        }
        pd.DataFrame([summary]).to_csv(fold_dir / "fold_validation_summary_ddof1.csv", index=False)
        all_fold_summary.append(pd.DataFrame([summary]))

        seed_artifacts = []
        for seed in SEEDS:
            paths = seed_paths(fold_dir, seed)
            row = pd.read_csv(paths["seed_result"]).iloc[0]
            seed_artifacts.append({
                "seed": seed,
                "checkpoint": str(paths["checkpoint"].resolve()),
                "checkpoint_sha256": sha256(paths["checkpoint"]),
                "correction": str(paths["correction"].resolve()),
                "correction_sha256": sha256(paths["correction"]),
                "validation_seed_result": str(paths["seed_result"].resolve()),
                "validation_seed_result_sha256": sha256(paths["seed_result"]),
                "best_epoch": int(row["best_epoch"]),
                "selected_correction_method": str(row["selected_correction_method"]),
            })
        fold_lock = {
            "selection_complete": True,
            "selection_used_test": False,
            "test_data_loaded": False,
            "protocol_version": PROTOCOL_VERSION,
            "transfer_mode": "history_aware_zero_shot",
            "pure_cold_start": False,
            "horizon": args.horizon,
            "lookback": LOOKBACK,
            "seeds": list(SEEDS),
            "all_seeds_retained": True,
            "sample_sd_ddof": 1,
            "target_city": heldout,
            "target_col": target_col,
            "source_cities": source_cities,
            "n_source_cities": 14,
            "patchtst_input_mode": "S",
            "strict_alignment_feature_count": len(feature_cols),
            "strict_alignment_feature_cols": feature_cols,
            "source_target_scaler": scaler_record,
            "source_target_scaler_file": str((fold_dir / "source_target_scaler.json").resolve()),
            "source_target_scaler_sha256": sha256(fold_dir / "source_target_scaler.json"),
            "train_windows": len(train_ds),
            "validation_windows": len(val_ds),
            "train_sha256": EXPECTED_TRAIN_SHA,
            "validation_sha256": EXPECTED_VAL_SHA,
            **hashes,
            "seed_artifacts": seed_artifacts,
        }
        json_dump(lock_path, fold_lock)

    # Revalidate every frozen fold before stage closure.
    fold_entries = []
    for heldout in cities:
        fdir = out / "folds" / heldout
        lp = fdir / "fold_lock.json"
        if not lp.is_file():
            raise RuntimeError(f"Missing fold lock: {heldout}")
        fl = json.loads(lp.read_text(encoding="utf-8"))
        if fl.get("selection_complete") is not True or fl.get("selection_used_test") is not False or fl.get("test_data_loaded") is not False:
            raise RuntimeError(f"Invalid fold lock: {heldout}")
        if fl.get("target_city") != heldout or heldout in fl.get("source_cities", []) or len(fl.get("source_cities", [])) != 14:
            raise RuntimeError("Held-out leakage in frozen fold")
        if int(fl.get("horizon", -1)) != args.horizon or int(fl.get("lookback", -1)) != LOOKBACK:
            raise RuntimeError("Frozen fold horizon/lookback mismatch")
        for k, v in hashes.items():
            if fl.get(k) != v:
                raise RuntimeError(f"Frozen code hash mismatch: {k}")
        if sha256(Path(fl["source_target_scaler_file"])) != fl["source_target_scaler_sha256"]:
            raise RuntimeError("Source scaler file changed")
        for art in fl["seed_artifacts"]:
            if sha256(Path(art["checkpoint"])) != art["checkpoint_sha256"]:
                raise RuntimeError("Checkpoint changed after training")
            if sha256(Path(art["correction"])) != art["correction_sha256"]:
                raise RuntimeError("Correction changed after training")
            if sha256(Path(art["validation_seed_result"])) != art["validation_seed_result_sha256"]:
                raise RuntimeError("Validation result changed after training")
        fold_entries.append({"target_city": heldout, "fold_lock": str(lp.resolve()), "fold_lock_sha256": sha256(lp)})

    pd.concat(all_fold_summary, ignore_index=True).to_csv(out / "all_fold_validation_summaries_ddof1.csv", index=False)
    stage_lock = {
        "stage_complete": True,
        "selection_used_test": False,
        "test_data_loaded": False,
        "protocol_version": PROTOCOL_VERSION,
        "transfer_mode": "history_aware_zero_shot",
        "horizon": args.horizon,
        "lookback": LOOKBACK,
        "seeds": list(SEEDS),
        "n_folds": N_FOLDS,
        "source_training_runs": N_FOLDS * len(SEEDS),
        "cities": cities,
        "target_col": target_col,
        "train_sha256": EXPECTED_TRAIN_SHA,
        "validation_sha256": EXPECTED_VAL_SHA,
        **hashes,
        "folds": fold_entries,
    }
    json_dump(out / "stage_lock.json", stage_lock)
    print(f"PATCHTST LOCO SOURCE SELECTION PASS: H{args.horizon} | 15 folds x 3 seeds = 45 runs")
    print("Test file was not read.")


def selection_dir(root: Path, horizon: int) -> Path:
    return root / f"H{horizon}" / "selection"


def verify_both_selection_locks(args) -> Dict[str, Any]:
    ref, base, hashes = load_modules(args)
    root = Path(args.selection_root).resolve()
    locks = []
    shared_train = set()
    shared_val = set()
    for horizon in HORIZONS:
        sdir = selection_dir(root, horizon)
        sp = sdir / "stage_lock.json"
        if not sp.is_file():
            raise RuntimeError(f"Missing required selection stage lock before Test: {sp}")
        st = json.loads(sp.read_text(encoding="utf-8"))
        if st.get("stage_complete") is not True or st.get("selection_used_test") is not False or st.get("test_data_loaded") is not False:
            raise RuntimeError("Invalid selection stage lock")
        if int(st.get("horizon", -1)) != horizon or int(st.get("lookback", -1)) != LOOKBACK:
            raise RuntimeError("Selection stage protocol mismatch")
        if st.get("cities") is None or len(st["cities"]) != 15:
            raise RuntimeError("Selection city set invalid")
        for k, v in hashes.items():
            if st.get(k) != v:
                raise RuntimeError(f"Selection code hash mismatch: {k}")
        shared_train.add(st["train_sha256"])
        shared_val.add(st["validation_sha256"])
        for fe in st["folds"]:
            lp = Path(fe["fold_lock"])
            if sha256(lp) != fe["fold_lock_sha256"]:
                raise RuntimeError("Fold lock SHA mismatch")
            fl = json.loads(lp.read_text(encoding="utf-8"))
            if fl["target_city"] in fl["source_cities"] or len(fl["source_cities"]) != 14:
                raise RuntimeError("Held-out leakage in selection lock")
            if sha256(Path(fl["source_target_scaler_file"])) != fl["source_target_scaler_sha256"]:
                raise RuntimeError("Source scaler changed")
            for art in fl["seed_artifacts"]:
                if sha256(Path(art["checkpoint"])) != art["checkpoint_sha256"]:
                    raise RuntimeError("Frozen checkpoint changed")
                if sha256(Path(art["correction"])) != art["correction_sha256"]:
                    raise RuntimeError("Frozen correction changed")
        locks.append({"horizon": horizon, "path": str(sp), "sha256": sha256(sp)})
    if shared_train != {EXPECTED_TRAIN_SHA} or shared_val != {EXPECTED_VAL_SHA}:
        raise RuntimeError("Shared Train/Validation hashes across H24/H48 are invalid")
    return {"selection_stage_locks": locks, "train_sha256": EXPECTED_TRAIN_SHA, "validation_sha256": EXPECTED_VAL_SHA, **hashes}


def save_rich_npz(path: Path, ds, pred, true, day, city: str, seed: int):
    if len(ds.groups) != 1:
        raise RuntimeError("Target Test dataset must contain exactly one held-out city")
    starts = np.asarray([t for _, t in ds.index], dtype=np.int32)
    series_ns = ds.groups[0]["ts"].astype("datetime64[ns]").astype(np.int64)
    cities, origins, valid_ns = ds.metadata_arrays()
    if not np.all(cities == city):
        raise RuntimeError("Target-city metadata mismatch")
    np.savez_compressed(
        path,
        pred=np.asarray(pred, np.float32),
        true=np.asarray(true, np.float32),
        daylight=(np.asarray(day) > 0.5).astype(np.uint8),
        city=np.asarray(city),
        window_start_idx=starts,
        series_time_ns=series_ns,
        origin_ns=origins.astype(np.int64),
        valid_ns=valid_ns.astype(np.int64),
        lookback=np.int16(LOOKBACK),
        horizon=np.int16(ds.horizon),
        seed=np.int16(seed),
    )


def evaluate_zero_shot(args):
    if args.horizon not in HORIZONS:
        raise ValueError("Invalid horizon")
    # CRITICAL: verify both H24 and H48 selections before touching Test path.
    seal = verify_both_selection_locks(args)
    ref, base, hashes = load_modules(args)
    stage_dir = selection_dir(Path(args.selection_root).resolve(), args.horizon)
    stage = json.loads((stage_dir / "stage_lock.json").read_text(encoding="utf-8"))

    # Recheck the pinned official implementation before Test is touched.
    ref.preflight(Path(args.official_repo).resolve(), Path(args.base_code).resolve())

    # Only now may Test be hashed/read.
    test_path = Path(args.test_data).resolve()
    test_sha = sha256(test_path)
    if test_sha != EXPECTED_TEST_SHA:
        raise RuntimeError("Test data SHA differs from audited source")

    # Load Test once structurally to verify the city set. No model choice follows.
    test_all, target_col = base.load_split(str(test_path), "test", "auto")
    test_cities = sorted(test_all[base.CITY_COL].astype(str).unique().tolist())
    if test_cities != sorted(stage["cities"]):
        raise RuntimeError("Test city set differs from frozen selection cities")
    if target_col != stage.get("target_col"):
        raise RuntimeError("Test target differs from frozen source target")
    del test_all
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "raw_predictions"
    raw_dir.mkdir(parents=True, exist_ok=True)

    per_city_rows = []
    per_lead_rows = []
    energy_city_rows = []
    pooled = {seed: {"pred": [], "true": [], "day": [], "city": [], "origin": [], "valid": []} for seed in SEEDS}

    for heldout in stage["cities"]:
        fold_dir = stage_dir / "folds" / heldout
        fl = json.loads((fold_dir / "fold_lock.json").read_text(encoding="utf-8"))
        if heldout in fl["source_cities"]:
            raise RuntimeError("Held-out leakage discovered at Test")
        scaler_info = json.loads(Path(fl["source_target_scaler_file"]).read_text(encoding="utf-8"))
        y_mean = float(scaler_info["mean"])
        y_scale = float(scaler_info["scale"])
        feature_cols = fl["strict_alignment_feature_cols"]

        target_df = prepare_target_test(base, test_path, target_col, feature_cols, heldout)
        target_df[base.TARGET_SCALED_COL] = (
            target_df[base.TARGET_VALUE_COL].to_numpy(np.float64) - y_mean
        ) / y_scale
        ds = ref.SolarPatchTSTDataset(target_df, base, LOOKBACK, args.horizon, common_start=LOOKBACK)
        if len(ds) != EXPECTED_WINDOWS_PER_CITY[args.horizon]:
            raise RuntimeError(f"Unexpected Test window count for {heldout}: {len(ds)}")
        cities_arr, origins, valid_ns = ds.metadata_arrays()

        seed_map = {int(x["seed"]): x for x in fl["seed_artifacts"]}
        for seed in SEEDS:
            art = seed_map[seed]
            checkpoint = Path(art["checkpoint"])
            corr_path = Path(art["correction"])
            if sha256(checkpoint) != art["checkpoint_sha256"] or sha256(corr_path) != art["correction_sha256"]:
                raise RuntimeError("Frozen seed artifact changed before Test")
            correction = json.loads(corr_path.read_text(encoding="utf-8"))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = ref.build_model(Path(args.official_repo).resolve(), LOOKBACK, args.horizon, device)
            model.load_state_dict(ref.torch_load_state(checkpoint, device), strict=True)
            loader = ref.make_loader(ds, False, device)
            pred_s, _, true, day = ref.collect(model, loader, device)
            pred = ref.inverse_scaled(pred_s, y_mean, y_scale)
            pred = ref.apply_correction(pred, day, correction)
            pred = ref.postprocess(pred, day)
            metrics = ref.metric_dict(pred, true, day)
            energy = ref.energy_from_overlaps(pred, true, cities_arr, valid_ns)
            per_city_rows.append({
                "target_city": heldout,
                "horizon": args.horizon,
                "lookback": LOOKBACK,
                "seed": seed,
                "n_windows": len(ds),
                **metrics,
                **energy,
            })
            ph = ref.per_horizon(pred, true, day)
            ph.insert(0, "target_city", heldout)
            ph.insert(1, "seed", seed)
            ph.insert(2, "horizon", args.horizon)
            per_lead_rows.append(ph)
            energy_city_rows.append({"target_city": heldout, "seed": seed, "horizon": args.horizon, **energy})
            save_rich_npz(raw_dir / f"{heldout}_H{args.horizon}_seed{seed}.npz", ds, pred, true, day, heldout, seed)

            pooled[seed]["pred"].append(pred)
            pooled[seed]["true"].append(true)
            pooled[seed]["day"].append(day)
            pooled[seed]["city"].append(cities_arr)
            pooled[seed]["origin"].append(origins)
            pooled[seed]["valid"].append(valid_ns)

    per_city_df = pd.DataFrame(per_city_rows)
    per_city_df.to_csv(out / "loco_patchtst_test_by_city_and_seed.csv", index=False)
    pd.concat(per_lead_rows, ignore_index=True).to_csv(out / "loco_patchtst_test_per_lead_by_city.csv", index=False)
    pd.DataFrame(energy_city_rows).to_csv(out / "loco_patchtst_energy_by_city_seed.csv", index=False)

    pooled_rows = []
    pooled_lead = []
    energy_rows = []
    expected = EXPECTED_POOLED[args.horizon]
    for seed in SEEDS:
        pred = np.concatenate(pooled[seed]["pred"], axis=0)
        true = np.concatenate(pooled[seed]["true"], axis=0)
        day = np.concatenate(pooled[seed]["day"], axis=0)
        cities_arr = np.concatenate(pooled[seed]["city"], axis=0)
        valid_ns = np.concatenate(pooled[seed]["valid"], axis=0)
        m = ref.metric_dict(pred, true, day)
        e = ref.energy_from_overlaps(pred, true, cities_arr, valid_ns)
        row = {"scope": "patchtst_loco_history_aware_zero_shot", "horizon": args.horizon, "lookback": LOOKBACK, "seed": seed, "n_windows": pred.shape[0], **m, **e}
        pooled_rows.append(row)
        ph = ref.per_horizon(pred, true, day)
        ph.insert(0, "seed", seed)
        ph.insert(1, "horizon", args.horizon)
        pooled_lead.append(ph)
        energy_rows.append({"seed": seed, "horizon": args.horizon, **e})

        if pred.shape[0] != expected["windows"]:
            raise RuntimeError("Pooled window count mismatch")
        if pred.size != expected["residuals"]:
            raise RuntimeError("Pooled residual count mismatch")
        if int((day > 0.5).sum()) != expected["daylight"]:
            raise RuntimeError("Pooled daylight count mismatch")
        if int(e["unique_city_hours"]) != expected["unique_city_hours"]:
            raise RuntimeError("Unique city-hour count mismatch")
        if abs(float(e["true_energy_gwh"]) - EXPECTED_TRUE_ENERGY_GWH) > 1e-5:
            raise RuntimeError("True energy mismatch")

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(out / "loco_patchtst_pooled_test_seed_results.csv", index=False)
    pd.concat(pooled_lead, ignore_index=True).to_csv(out / "loco_patchtst_pooled_test_per_lead.csv", index=False)
    pd.DataFrame(energy_rows).to_csv(out / "loco_patchtst_aggregate_energy_by_seed.csv", index=False)

    summary = {
        "scope": "patchtst_loco_history_aware_zero_shot",
        "horizon": args.horizon,
        "lookback": LOOKBACK,
        "n_folds": N_FOLDS,
        "n_seeds": len(SEEDS),
        "sample_sd_ddof": 1,
        "pure_cold_start": False,
        "target_city_used_in_model_fitting": False,
        "target_city_past_history_used_at_inference": True,
    }
    for col in ref.METRICS + ["true_energy_gwh", "predicted_energy_gwh", "energy_error_gwh", "energy_error_pct"]:
        x = pd.to_numeric(pooled_df[col], errors="coerce").to_numpy(float)
        summary[col + "_mean"] = float(np.mean(x))
        summary[col + "_sd"] = float(np.std(x, ddof=1))
    json_dump(out / "test_summary.json", summary)

    audit = {
        "status": "PASS",
        "protocol_version": PROTOCOL_VERSION,
        "analysis": "PatchTST LOCO history-aware zero-shot",
        "horizon": args.horizon,
        "lookback": LOOKBACK,
        "n_folds": N_FOLDS,
        "seeds": list(SEEDS),
        "selection_used_test": False,
        "both_H24_H48_selection_locks_verified_before_test_open": True,
        "verified_selection_locks": seal["selection_stage_locks"],
        "target_city_excluded_from_source_fit_all_folds": True,
        "source_scaler_excludes_target_city_all_folds": True,
        "target_city_history_used_at_inference": True,
        "pure_cold_start": False,
        "test_sha256": test_sha,
        "expected_common_set": expected,
        "actual_common_set": {
            "windows": int(pooled_df.iloc[0]["n_windows"]),
            "residuals": int(pooled_df.iloc[0]["count"]),
            "daylight_residuals": int(pooled_df.iloc[0]["daylight_count"]),
            "unique_city_hours": int(pooled_df.iloc[0]["unique_city_hours"]),
        },
        "expected_true_energy_gwh": EXPECTED_TRUE_ENERGY_GWH,
        "actual_true_energy_gwh": float(pooled_df.iloc[0]["true_energy_gwh"]),
        **hashes,
    }
    json_dump(out / "test_audit.json", audit)

    lines = [
        "PATCHTST LOCO HISTORY-AWARE ZERO-SHOT — INDEPENDENT TEST",
        "=" * 94,
        f"Horizon: H{args.horizon}",
        f"Lookback: L{LOOKBACK} (frozen; no LOCO tuning)",
        "Held-out target city used in model fitting: NO",
        "Target city past PV history used at inference: YES",
        "Pure cold-start: NO",
        f"Folds: {N_FOLDS}; seeds: {list(SEEDS)}",
        f"Windows: {audit['actual_common_set']['windows']:,}",
        f"Residuals: {audit['actual_common_set']['residuals']:,}",
        f"Daylight residuals: {audit['actual_common_set']['daylight_residuals']:,}",
        "",
        f"RMSE: {summary['per_kwp_rmse_mean']:.9f} ± {summary['per_kwp_rmse_sd']:.9f}",
        f"MAE: {summary['per_kwp_mae_mean']:.9f} ± {summary['per_kwp_mae_sd']:.9f}",
        f"R2: {summary['per_kwp_r2_mean']:.9f} ± {summary['per_kwp_r2_sd']:.9f}",
        f"Daylight RMSE: {summary['daylight_per_kwp_rmse_mean']:.9f} ± {summary['daylight_per_kwp_rmse_sd']:.9f}",
        f"Daylight MAE: {summary['daylight_per_kwp_mae_mean']:.9f} ± {summary['daylight_per_kwp_mae_sd']:.9f}",
        f"Energy error (%): {summary['energy_error_pct_mean']:.6f} ± {summary['energy_error_pct_sd']:.6f}",
        "STATUS: PASS",
    ]
    (out / "final_test_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def add_common(p):
    p.add_argument("--patchtst-reference", dest="patchtst_reference", default="/workspace/lstm/patchtst_loco_history_zero_shot_v1/solar_patchtst_original_protocol_reference.py")
    p.add_argument("--base-code", dest="base_code", default="/workspace/lstm/patchtst_loco_history_zero_shot_v1/solar_lstm_pv_per_kwp_v3.py")
    p.add_argument("--protocol-markdown", dest="protocol_markdown", default="/workspace/lstm/patchtst_loco_history_zero_shot_v1/PATCHTST_LOCO_PROTOCOL_v1.md")
    p.add_argument("--source-reference", dest="source_reference", default="/workspace/lstm/patchtst_loco_history_zero_shot_v1/SOURCE_AUDITED_REFERENCE.json")
    p.add_argument("--official-repo", dest="official_repo", default="/workspace/lstm/vendor/PatchTST_official")


def parse_args():
    p = argparse.ArgumentParser(description="Audited PatchTST LOCO history-aware zero-shot protocol")
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("preflight")
    add_common(q)
    q.add_argument("--train-data", dest="train_data", default="/workspace/lstm/data2/train.csv")
    q.add_argument("--val-data", dest="val_data", default="/workspace/lstm/data2/valid.csv")
    q.add_argument("--out-dir", dest="out_dir", required=True)

    q = sub.add_parser("select-source")
    add_common(q)
    q.add_argument("--train-data", dest="train_data", default="/workspace/lstm/data2/train.csv")
    q.add_argument("--val-data", dest="val_data", default="/workspace/lstm/data2/valid.csv")
    q.add_argument("--preflight-lock", dest="preflight_lock", required=True)
    q.add_argument("--horizon", type=int, choices=HORIZONS, required=True)
    q.add_argument("--out-dir", dest="out_dir", required=True)
    q.add_argument("--resume", action="store_true")

    q = sub.add_parser("evaluate-zero-shot")
    add_common(q)
    q.add_argument("--selection-root", dest="selection_root", required=True)
    q.add_argument("--test-data", dest="test_data", required=True)
    q.add_argument("--horizon", type=int, choices=HORIZONS, required=True)
    q.add_argument("--out-dir", dest="out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    if args.command == "preflight":
        preflight(args)
    elif args.command == "select-source":
        select_source(args)
    elif args.command == "evaluate-zero-shot":
        evaluate_zero_shot(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
