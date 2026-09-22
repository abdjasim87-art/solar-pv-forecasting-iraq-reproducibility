#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

PROTOCOL_VERSION = "2.0"
TRACKS = ("history_aware", "cold_start")
HORIZONS = (24, 48)
SEEDS = (1, 2, 3)
LOOKBACK = 24
ALIGNMENT_ROWS = 168
EXPECTED_FEATURES = {"history_aware": 53, "cold_start": 34}
EXPECTED_AR_FEATURES = 19
BASE_EXPECTED_SHA = "41f7892cbdd296f6ea4d6887a5dd92669cd3a1a42ef0fb4695c29e7d79369438"
STRICT_EXPECTED_SHA = "24e6ad5b53f8c8a3160c8772eeb6bf74b931f0496a6674b13346b7fc87952f70"
PROTOCOL_MD_EXPECTED_SHA = "4cffa5c7ec3459f2f403d6717cf236dafda6958d45ddb0a0b74e614c32e5c314"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def conv(x):
        if isinstance(x, Path): return str(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=conv), encoding="utf-8")


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    m = importlib.util.module_from_spec(spec)
    # Register before execution so dataclasses/type inspection can resolve __module__
    # consistently across Python versions.
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def current_paths(args) -> Tuple[Path, Path, Path]:
    base_path = Path(args.base_code).resolve()
    strict_path = Path(args.strict_reference).resolve()
    protocol_md = Path(args.protocol_markdown).resolve()
    return base_path, strict_path, protocol_md


def verify_code_files(args) -> Dict[str, str]:
    base_path, strict_path, protocol_md = current_paths(args)
    for p in (base_path, strict_path, protocol_md):
        if not p.is_file():
            raise FileNotFoundError(p)
    got = {
        "base_code_sha256": sha256(base_path),
        "strict_reference_sha256": sha256(strict_path),
        "protocol_markdown_sha256": sha256(protocol_md),
        "loco_code_sha256": sha256(Path(__file__).resolve()),
    }
    if got["base_code_sha256"] != BASE_EXPECTED_SHA:
        raise RuntimeError(f"Authoritative base LSTM SHA mismatch: {got['base_code_sha256']} != {BASE_EXPECTED_SHA}")
    if got["strict_reference_sha256"] != STRICT_EXPECTED_SHA:
        raise RuntimeError(f"Strict reference SHA mismatch: {got['strict_reference_sha256']} != {STRICT_EXPECTED_SHA}")
    if got["protocol_markdown_sha256"] != PROTOCOL_MD_EXPECTED_SHA:
        raise RuntimeError("Frozen LOCO protocol markdown SHA mismatch")
    return got


def modules(args):
    hashes = verify_code_files(args)
    base = import_file("loco_base_lstm", Path(args.base_code))
    strict = import_file("loco_strict_reference", Path(args.strict_reference))
    return base, strict, hashes


def load_prepared(base, path: Path, label: str, target_arg: str) -> Tuple[pd.DataFrame, str]:
    df, target_col = base.load_split(str(path), label, target_arg)
    return base.prepare_physical_target(df, target_col), target_col


def trim_first_per_city(df: pd.DataFrame, base, n: int) -> pd.DataFrame:
    pieces = []
    for city, g in df.groupby(base.CITY_COL, sort=False):
        g = g.sort_values(base.TIME_COL).reset_index(drop=True)
        if len(g) <= n:
            raise ValueError(f"City {city} has only {len(g)} rows; cannot trim {n}")
        pieces.append(g.iloc[n:].copy())
    return pd.concat(pieces, ignore_index=True).sort_values([base.CITY_COL, base.TIME_COL]).reset_index(drop=True)


def build_track_frames(base, train: pd.DataFrame, val: pd.DataFrame, target_col: str, track: str):
    train = train.copy(); val = val.copy()
    if track == "history_aware":
        before_train, before_val = len(train), len(val)
        train = base.add_ar_solar_features(train, target_col)
        val = base.add_ar_solar_features(val, target_col)
    elif track == "cold_start":
        train = trim_first_per_city(train, base, ALIGNMENT_ROWS)
        val = trim_first_per_city(val, base, ALIGNMENT_ROWS)
        before_train, before_val = len(train), len(val)
    else:
        raise ValueError(track)

    feature_cols, continuous_cols = base.build_feature_columns(
        train, feature_set="basic", use_solar_history=(track == "history_aware")
    )
    required = feature_cols + [base.TARGET_VALUE_COL, base.RAW_TARGET_COL, base.DAYLIGHT_COL]
    pre_drop_train, pre_drop_val = len(train), len(val)
    train = train.dropna(subset=required).copy()
    val = val.dropna(subset=required).copy()
    drop_train = pre_drop_train - len(train)
    drop_val = pre_drop_val - len(val)

    n_cities = train[base.CITY_COL].nunique()
    expected_drop = ALIGNMENT_ROWS * n_cities if track == "history_aware" else 0
    if drop_train != expected_drop or drop_val != expected_drop:
        raise RuntimeError(
            f"Unexpected feature-drop count for {track}: train={drop_train}, val={drop_val}, expected_each={expected_drop}. "
            "Exact origin alignment is not guaranteed; stop and audit data."
        )
    if len(feature_cols) != EXPECTED_FEATURES[track]:
        raise RuntimeError(f"{track}: expected {EXPECTED_FEATURES[track]} features, got {len(feature_cols)}")
    if "city_id" in feature_cols:
        raise RuntimeError("city_id must not be a predictor in LOCO global mode")
    if track == "history_aware":
        ar = [c for c in base.SOLAR_AR_FEATURES if c in feature_cols]
        if len(ar) != EXPECTED_AR_FEATURES:
            raise RuntimeError(f"Expected {EXPECTED_AR_FEATURES} AR features, got {len(ar)}")
    else:
        leaked = [c for c in base.SOLAR_AR_FEATURES if c in feature_cols]
        if leaked:
            raise RuntimeError(f"Cold-start contains target-history features: {leaked}")

    return train, val, list(feature_cols), list(continuous_cols), {
        "feature_drop_train": int(drop_train), "feature_drop_val": int(drop_val),
        "alignment_trim_rows_per_city": ALIGNMENT_ROWS if track == "cold_start" else 0,
    }


def fit_source_scalers(base, train: pd.DataFrame, val: pd.DataFrame, feature_cols, continuous_cols):
    train = train.copy(); val = val.copy()
    x_scaler = None
    if continuous_cols:
        x_scaler = StandardScaler()
        train.loc[:, continuous_cols] = x_scaler.fit_transform(train[continuous_cols].to_numpy(np.float32))
        val.loc[:, continuous_cols] = x_scaler.transform(val[continuous_cols].to_numpy(np.float32))
    y_scaler = StandardScaler()
    train.loc[:, base.TARGET_SCALED_COL] = y_scaler.fit_transform(train[[base.TARGET_VALUE_COL]].to_numpy(np.float32)).reshape(-1)
    val.loc[:, base.TARGET_SCALED_COL] = y_scaler.transform(val[[base.TARGET_VALUE_COL]].to_numpy(np.float32)).reshape(-1)
    for label, df in (("train", train), ("valid", val)):
        if not np.isfinite(df[feature_cols].to_numpy(np.float32)).all():
            raise RuntimeError(f"{label}: non-finite features after scaling")
        if not np.isfinite(df[base.TARGET_SCALED_COL].to_numpy(np.float32)).all():
            raise RuntimeError(f"{label}: non-finite target after scaling")
    return train, val, x_scaler, y_scaler


def ddof1_summary(df: pd.DataFrame, exclude: Sequence[str] = ("seed",)) -> pd.DataFrame:
    row: Dict[str, Any] = {"n_seeds": int(df["seed"].nunique()) if "seed" in df else 0, "sd_ddof": 1}
    for c in df.columns:
        if c in exclude: continue
        if pd.api.types.is_numeric_dtype(df[c]):
            row[f"{c}_mean"] = float(df[c].mean())
            row[f"{c}_std"] = float(df[c].std(ddof=1)) if len(df[c].dropna()) > 1 else float("nan")
    return pd.DataFrame([row])


def preflight(args):
    base, strict, code_hashes = modules(args)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    train_path, val_path = Path(args.train_data), Path(args.val_data)
    train, t1 = load_prepared(base, train_path, "train", args.target_col)
    val, t2 = load_prepared(base, val_path, "valid", args.target_col)
    if t1 != t2: raise RuntimeError("Train/Validation target mismatch")
    train_cities = sorted(train[base.CITY_COL].astype(str).unique().tolist())
    val_cities = sorted(val[base.CITY_COL].astype(str).unique().tolist())
    if train_cities != val_cities or len(train_cities) != 15:
        raise RuntimeError(f"Expected the same 15 cities in Train/Validation; train={train_cities}, val={val_cities}")
    if train.duplicated([base.CITY_COL, base.TIME_COL]).any() or val.duplicated([base.CITY_COL, base.TIME_COL]).any():
        raise RuntimeError("Duplicate city-time rows found")

    track_a = build_track_frames(base, train, val, t1, "history_aware")
    track_b = build_track_frames(base, train, val, t1, "cold_start")
    # Exact timestamp alignment check after feature/intentional trimming.
    for split_idx, split_name in ((0, "train"), (1, "valid")):
        a = track_a[split_idx][[base.CITY_COL, base.TIME_COL]].reset_index(drop=True)
        b = track_b[split_idx][[base.CITY_COL, base.TIME_COL]].reset_index(drop=True)
        if not a.equals(b):
            raise RuntimeError(f"History-aware and cold-start {split_name} timestamp grids are not identical")

    fold_checks = []
    for target in train_cities:
        source = [c for c in train_cities if c != target]
        if len(source) != 14 or target in source:
            raise RuntimeError("LOCO fold construction failure")
        fold_checks.append({"target_city": target, "n_source_cities": 14, "source_cities": source})

    lock = {
        "status": "PASS", "protocol_version": PROTOCOL_VERSION,
        "selection_used_test": False, "test_path_read": False,
        "target_col": t1, "cities": train_cities, "n_cities": 15,
        "lookback_fixed": LOOKBACK, "horizons": list(HORIZONS), "seeds": list(SEEDS),
        "tracks": list(TRACKS), "alignment_rows": ALIGNMENT_ROWS,
        "feature_counts": {"history_aware": len(track_a[2]), "cold_start": len(track_b[2])},
        "history_aware_features": track_a[2], "cold_start_features": track_b[2],
        "train_rows_raw": int(len(train)), "valid_rows_raw": int(len(val)),
        "train_time_min": str(train[base.TIME_COL].min()), "train_time_max": str(train[base.TIME_COL].max()),
        "valid_time_min": str(val[base.TIME_COL].min()), "valid_time_max": str(val[base.TIME_COL].max()),
        "track_a_alignment": track_a[4], "track_b_alignment": track_b[4],
        "fold_checks": fold_checks,
        "train_sha256": sha256(train_path), "validation_sha256": sha256(val_path),
        **code_hashes,
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    json_dump(out/"preflight_lock.json", lock)
    (out/"history_aware_features.json").write_text(json.dumps(track_a[2], indent=2), encoding="utf-8")
    (out/"cold_start_features.json").write_text(json.dumps(track_b[2], indent=2), encoding="utf-8")
    pd.DataFrame(fold_checks).to_csv(out/"loco_folds.csv", index=False)
    print("LOCO v2 PRE-FLIGHT: PASS")
    print(f"15 cities | Track A features={len(track_a[2])} | Track B features={len(track_b[2])}")
    print("Test file was not read.")


def load_and_verify_preflight(args):
    p = Path(args.preflight_lock)
    if not p.is_file(): raise FileNotFoundError(p)
    lock = json.loads(p.read_text(encoding="utf-8"))
    if lock.get("status") != "PASS" or lock.get("selection_used_test") is not False or lock.get("test_path_read") is not False:
        raise RuntimeError("Invalid preflight lock")
    _, _, code_hashes = modules(args)
    for k in ("base_code_sha256", "strict_reference_sha256", "protocol_markdown_sha256", "loco_code_sha256"):
        if lock.get(k) != code_hashes.get(k):
            raise RuntimeError(f"Preflight code hash mismatch: {k}")
    if lock.get("train_sha256") != sha256(Path(args.train_data)):
        raise RuntimeError("Train SHA changed since preflight")
    if lock.get("validation_sha256") != sha256(Path(args.val_data)):
        raise RuntimeError("Validation SHA changed since preflight")
    return lock


def make_training_args(args, horizon: int):
    # Namespace expected by the unchanged strict training helper.
    d = dict(vars(args))
    d.update({
        "horizon": horizon, "feature_set": "basic", "no_solar_history": args.track == "cold_start",
        "hidden_size": 256, "num_layers": 3, "dropout": 0.20, "head_hidden": 128,
        "lr": 0.001, "weight_decay": 1e-4, "grad_clip": 1.0,
        "loss": "huber", "huber_beta": 1.0,
        "daylight_weight": 4.0, "night_weight": 0.25, "production_weight": 1.0,
        "monitor_metric": "daylight_per_kwp_rmse", "calibration_metric": "daylight_per_kwp_rmse",
        "bias_corrections": ["none", "global_bias", "daylight_bias", "horizon_bias", "horizon_daylight_bias"],
        "early_patience": 15, "min_delta": 1e-5, "lr_patience": 5, "lr_factor": 0.5,
        "reference_capacity_kwp": 3370.0, "max_kw_per_kwp": 1.20,
        "clip_per_kwp": True, "force_night_zero": True,
        "epochs": 150, "batch_size": 512, "num_workers": args.num_workers,
        "verbose": args.verbose,
    })
    return argparse.Namespace(**d)


def seed_complete(seed_dir: Path, seed: int, horizon: int) -> Optional[Dict[str, Any]]:
    result = seed_dir/"validation_seed_result.csv"
    ckpt = seed_dir/"best_checkpoint.pt"
    corr = seed_dir/"correction.json"
    if not (result.is_file() and ckpt.is_file() and corr.is_file()): return None
    df = pd.read_csv(result)
    if len(df) != 1: return None
    row = df.iloc[0].to_dict()
    if int(row.get("seed", -1)) != seed or int(row.get("horizon", -1)) != horizon or int(row.get("lookback", -1)) != LOOKBACK:
        return None
    return row


def select_source(args):
    if args.track not in TRACKS or args.horizon not in HORIZONS:
        raise ValueError("Invalid track/horizon")
    pre = load_and_verify_preflight(args)
    base, strict, code_hashes = modules(args)
    train, t1 = load_prepared(base, Path(args.train_data), "train", args.target_col)
    val, t2 = load_prepared(base, Path(args.val_data), "valid", args.target_col)
    if t1 != t2 or t1 != pre["target_col"]: raise RuntimeError("Target mismatch")
    all_cities = pre["cities"]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    train_args = make_training_args(args, args.horizon)
    stage_rows = []

    for heldout in all_cities:
        fold_dir = out/"folds"/strict.safe_name(heldout)
        lock_path = fold_dir/"fold_lock.json"
        if args.resume and lock_path.is_file():
            old = json.loads(lock_path.read_text(encoding="utf-8"))
            if old.get("selection_complete") is True and old.get("selection_used_test") is False and old.get("track") == args.track and int(old.get("horizon")) == args.horizon:
                print(f"RESUME: verified completed fold lock exists for {heldout}; fold will be revalidated at stage-lock time")
                stage_rows.append(pd.read_csv(fold_dir/"fold_validation_summary_ddof1.csv").assign(target_city=heldout))
                continue

        source_train = train[train[base.CITY_COL].astype(str) != str(heldout)].copy()
        source_val = val[val[base.CITY_COL].astype(str) != str(heldout)].copy()
        source_cities = sorted(source_train[base.CITY_COL].astype(str).unique().tolist())
        if len(source_cities) != 14 or heldout in source_cities:
            raise RuntimeError(f"Held-out leakage in fold {heldout}")
        if source_cities != sorted(source_val[base.CITY_COL].astype(str).unique().tolist()):
            raise RuntimeError("Source train/validation city mismatch")

        tr, va, features, continuous, align = build_track_frames(base, source_train, source_val, t1, args.track)
        tr, va, x_scaler, y_scaler = fit_source_scalers(base, tr, va, features, continuous)
        data = strict.PreparedData(tr, va, None, t1, features, continuous, x_scaler, y_scaler, source_cities)
        train_ds = strict.StrictSolarDataset(base, tr, features, LOOKBACK, args.horizon, min_start_t=LOOKBACK)
        val_ds = strict.StrictSolarDataset(base, va, features, LOOKBACK, args.horizon, min_start_t=ALIGNMENT_ROWS)
        if len(train_ds) == 0 or len(val_ds) == 0: raise RuntimeError("No source windows")

        fold_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(y_scaler, fold_dir/"y_scaler.pkl")
        if x_scaler is not None: joblib.dump(x_scaler, fold_dir/"x_scaler.pkl")
        (fold_dir/"feature_cols.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
        (fold_dir/"continuous_cols.json").write_text(json.dumps(continuous, indent=2), encoding="utf-8")
        train_ds.window_counts_by_city().to_csv(fold_dir/"source_train_window_counts.csv", index=False)
        val_ds.window_counts_by_city().to_csv(fold_dir/"source_validation_window_counts.csv", index=False)

        seed_rows = []
        for seed in SEEDS:
            seed_dir = fold_dir/f"L{LOOKBACK}"/f"seed_{seed}"
            existing = seed_complete(seed_dir, seed, args.horizon) if args.resume else None
            if existing is not None:
                print(f"RESUME {heldout} H{args.horizon} {args.track} seed={seed}")
                seed_rows.append(existing)
            else:
                print(f"TRAIN {heldout} H{args.horizon} {args.track} seed={seed}")
                seed_rows.append(strict.train_validation_seed(base, train_ds, val_ds, data, train_args, LOOKBACK, seed, fold_dir))

        seed_df = pd.DataFrame(seed_rows).sort_values("seed").reset_index(drop=True)
        if sorted(seed_df["seed"].astype(int).tolist()) != list(SEEDS): raise RuntimeError("Seed retention failure")
        seed_df.to_csv(fold_dir/"fold_validation_seed_results.csv", index=False)
        summary = ddof1_summary(seed_df)
        summary.insert(0, "target_city", heldout); summary.insert(1, "track", args.track); summary.insert(2, "horizon", args.horizon); summary.insert(3, "lookback", LOOKBACK)
        summary.to_csv(fold_dir/"fold_validation_summary_ddof1.csv", index=False)

        seed_locks = []
        for seed in SEEDS:
            sdir = fold_dir/f"L{LOOKBACK}"/f"seed_{seed}"
            seed_locks.append({
                "seed": seed,
                "checkpoint": str((sdir/"best_checkpoint.pt").resolve()), "checkpoint_sha256": sha256(sdir/"best_checkpoint.pt"),
                "correction": str((sdir/"correction.json").resolve()), "correction_sha256": sha256(sdir/"correction.json"),
                "validation_seed_result_sha256": sha256(sdir/"validation_seed_result.csv"),
            })
        fold_lock = {
            "selection_complete": True, "selection_used_test": False, "test_data_loaded": False,
            "protocol_version": PROTOCOL_VERSION, "track": args.track, "horizon": args.horizon,
            "lookback": LOOKBACK, "seeds": list(SEEDS), "sample_sd_ddof": 1,
            "target_city": heldout, "source_cities": source_cities, "n_source_cities": 14,
            "feature_count": len(features), "feature_cols": features, "continuous_cols": continuous,
            "alignment": align, "train_windows": len(train_ds), "validation_windows": len(val_ds),
            "x_scaler_sha256": sha256(fold_dir/"x_scaler.pkl") if (fold_dir/"x_scaler.pkl").is_file() else None,
            "y_scaler_sha256": sha256(fold_dir/"y_scaler.pkl"),
            "train_sha256": pre["train_sha256"], "validation_sha256": pre["validation_sha256"],
            **code_hashes, "seed_artifacts": seed_locks,
        }
        json_dump(lock_path, fold_lock)
        stage_rows.append(summary)

    # Revalidate every fold before writing the stage lock.
    fold_entries = []
    for heldout in all_cities:
        fdir = out/"folds"/strict.safe_name(heldout); lp = fdir/"fold_lock.json"
        if not lp.is_file(): raise RuntimeError(f"Missing fold lock: {heldout}")
        fl = json.loads(lp.read_text(encoding="utf-8"))
        if fl.get("selection_complete") is not True or fl.get("selection_used_test") is not False:
            raise RuntimeError(f"Invalid fold lock: {heldout}")
        if fl.get("target_city") != heldout or heldout in fl.get("source_cities", []): raise RuntimeError("Held-out leakage in lock")
        if int(fl.get("horizon")) != args.horizon or fl.get("track") != args.track or int(fl.get("lookback")) != LOOKBACK:
            raise RuntimeError("Fold protocol mismatch")
        if fl.get("train_sha256") != pre["train_sha256"] or fl.get("validation_sha256") != pre["validation_sha256"]:
            raise RuntimeError("Data SHA mismatch across folds")
        for k in ("base_code_sha256", "strict_reference_sha256", "protocol_markdown_sha256", "loco_code_sha256"):
            if fl.get(k) != code_hashes[k]:
                raise RuntimeError(f"Code hash mismatch in fold {heldout}: {k}")
        if fl.get("seeds") != list(SEEDS) or int(fl.get("feature_count", -1)) != EXPECTED_FEATURES[args.track]:
            raise RuntimeError(f"Frozen fold grid/feature mismatch: {heldout}")
        for s in fl["seed_artifacts"]:
            if sha256(Path(s["checkpoint"])) != s["checkpoint_sha256"]: raise RuntimeError("Checkpoint SHA mismatch")
            if sha256(Path(s["correction"])) != s["correction_sha256"]: raise RuntimeError("Correction SHA mismatch")
        fold_entries.append({"target_city": heldout, "fold_lock": str(lp.resolve()), "fold_lock_sha256": sha256(lp)})

    combined = pd.concat(stage_rows, ignore_index=True)
    combined.to_csv(out/"all_fold_validation_summaries_ddof1.csv", index=False)
    stage_lock = {
        "stage_complete": True, "selection_used_test": False, "test_data_loaded": False,
        "protocol_version": PROTOCOL_VERSION, "track": args.track, "horizon": args.horizon,
        "lookback": LOOKBACK, "seeds": list(SEEDS), "sample_sd_ddof": 1,
        "n_folds": 15, "cities": all_cities,
        "train_sha256": pre["train_sha256"], "validation_sha256": pre["validation_sha256"],
        **code_hashes, "folds": fold_entries,
    }
    json_dump(out/"stage_lock.json", stage_lock)
    print(f"SOURCE SELECTION PASS: H{args.horizon} {args.track} | 15 folds × 3 seeds = 45 runs")
    print("Test file was not read.")


def expected_stage_dir(root: Path, horizon: int, track: str) -> Path:
    return root/f"H{horizon}"/track/"selection"


def verify_all_stage_locks(args) -> Dict[str, Any]:
    # CRITICAL: this function must finish before any test-file hash/read is attempted.
    _, _, code_hashes = modules(args)
    root = Path(args.selection_root)
    locks = []
    shared_train = set(); shared_val = set()
    for horizon in HORIZONS:
        for track in TRACKS:
            sdir = expected_stage_dir(root, horizon, track); lp = sdir/"stage_lock.json"
            if not lp.is_file(): raise RuntimeError(f"Missing required stage lock: {lp}")
            st = json.loads(lp.read_text(encoding="utf-8"))
            if st.get("stage_complete") is not True or st.get("selection_used_test") is not False or st.get("test_data_loaded") is not False:
                raise RuntimeError(f"Invalid stage lock: {lp}")
            if int(st.get("horizon")) != horizon or st.get("track") != track or int(st.get("lookback")) != LOOKBACK:
                raise RuntimeError(f"Stage identity mismatch: {lp}")
            if st.get("seeds") != list(SEEDS) or int(st.get("n_folds")) != 15: raise RuntimeError("Stage grid mismatch")
            for k in ("base_code_sha256", "strict_reference_sha256", "protocol_markdown_sha256", "loco_code_sha256"):
                if st.get(k) != code_hashes[k]: raise RuntimeError(f"Code hash mismatch in stage {lp}: {k}")
            shared_train.add(st["train_sha256"]); shared_val.add(st["validation_sha256"])
            if len(st.get("folds", [])) != 15: raise RuntimeError("Incomplete folds")
            for entry in st["folds"]:
                fp = Path(entry["fold_lock"])
                if not fp.is_file() or sha256(fp) != entry["fold_lock_sha256"]: raise RuntimeError("Fold lock hash mismatch")
                fl = json.loads(fp.read_text(encoding="utf-8"))
                if fl.get("selection_used_test") is not False or fl.get("test_data_loaded") is not False: raise RuntimeError("Fold used Test")
                if fl["target_city"] in fl["source_cities"] or len(fl["source_cities"]) != 14: raise RuntimeError("Held-out leakage")
                for s in fl["seed_artifacts"]:
                    if sha256(Path(s["checkpoint"])) != s["checkpoint_sha256"]: raise RuntimeError("Checkpoint changed")
                    if sha256(Path(s["correction"])) != s["correction_sha256"]: raise RuntimeError("Correction changed")
            locks.append({"path": str(lp.resolve()), "sha256": sha256(lp), "horizon": horizon, "track": track})
    if len(shared_train) != 1 or len(shared_val) != 1: raise RuntimeError("The four stages do not share one Train/Validation SHA")
    return {"stage_locks": locks, "train_sha256": next(iter(shared_train)), "validation_sha256": next(iter(shared_val)), **code_hashes}


def prepare_target_test(base, full_test: pd.DataFrame, target_col: str, target_city: str, track: str, features: List[str], continuous: List[str], x_scaler, y_scaler):
    test = full_test[full_test[base.CITY_COL].astype(str) == str(target_city)].copy()
    if test.empty: raise RuntimeError(f"Target city absent from Test: {target_city}")
    if track == "history_aware":
        test = base.add_ar_solar_features(test, target_col)
    else:
        test = trim_first_per_city(test, base, ALIGNMENT_ROWS)
    built, built_cont = base.build_feature_columns(test, "basic", track == "history_aware")
    if list(built) != list(features) or list(built_cont) != list(continuous):
        raise RuntimeError(f"Target feature schema mismatch for {target_city}")
    required = features + [base.TARGET_VALUE_COL, base.RAW_TARGET_COL, base.DAYLIGHT_COL]
    before = len(test); test = test.dropna(subset=required).copy(); dropped = before-len(test)
    expected_drop = ALIGNMENT_ROWS if track == "history_aware" else 0
    if dropped != expected_drop: raise RuntimeError(f"Unexpected Test feature drop for {target_city}: {dropped} != {expected_drop}")
    if continuous:
        test.loc[:, continuous] = x_scaler.transform(test[continuous].to_numpy(np.float32))
    test.loc[:, base.TARGET_SCALED_COL] = y_scaler.transform(test[[base.TARGET_VALUE_COL]].to_numpy(np.float32)).reshape(-1)
    return test


def save_rich_npz(path: Path, ds, pred, true, day, city: str, seed: int):
    series_times_ns = np.asarray(ds.times_by_city[0]).astype("datetime64[ns]").astype(np.int64)
    np.savez_compressed(
        path, pred=pred.astype(np.float32), true=true.astype(np.float32), daylight=day.astype(np.float32),
        window_start_idx=ds.window_start_indices(), series_time_ns=series_times_ns,
        horizon=np.int16(ds.horizon), lookback=np.int16(ds.lookback), seed=np.int16(seed), city=np.asarray(city),
    )


def evaluate_zero_shot(args):
    if args.track not in TRACKS or args.horizon not in HORIZONS: raise ValueError("Invalid track/horizon")
    # FIRST: verify ALL FOUR selection stages. Test path is untouched until this returns.
    seal = verify_all_stage_locks(args)
    base, strict, code_hashes = modules(args)
    stage_dir = expected_stage_dir(Path(args.selection_root), args.horizon, args.track)
    stage = json.loads((stage_dir/"stage_lock.json").read_text(encoding="utf-8"))

    # NOW and only now may the Test file be hashed and loaded.
    test_path = Path(args.test_data)
    test_sha = sha256(test_path)
    full_test, target_col = load_prepared(base, test_path, "test", args.target_col)
    all_test_cities = sorted(full_test[base.CITY_COL].astype(str).unique().tolist())
    if all_test_cities != sorted(stage["cities"]): raise RuntimeError("Test city set differs from frozen selection city set")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    train_args = make_training_args(args, args.horizon)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    per_city_rows = []
    ph_rows = []
    energy_rows = []
    pooled = {seed: {"pred": [], "true": [], "day": []} for seed in SEEDS}
    total_windows = {seed: 0 for seed in SEEDS}

    for heldout in stage["cities"]:
        fdir = stage_dir/"folds"/strict.safe_name(heldout)
        fl = json.loads((fdir/"fold_lock.json").read_text(encoding="utf-8"))
        features = fl["feature_cols"]; continuous = fl["continuous_cols"]
        x_scaler = joblib.load(fdir/"x_scaler.pkl") if (fdir/"x_scaler.pkl").is_file() else None
        y_scaler = joblib.load(fdir/"y_scaler.pkl")
        if (sha256(fdir/"x_scaler.pkl") if x_scaler is not None else None) != fl["x_scaler_sha256"]: raise RuntimeError("x_scaler changed")
        if sha256(fdir/"y_scaler.pkl") != fl["y_scaler_sha256"]: raise RuntimeError("y_scaler changed")
        target_test = prepare_target_test(base, full_test, target_col, heldout, args.track, features, continuous, x_scaler, y_scaler)
        data = strict.PreparedData(target_test.iloc[0:0].copy(), target_test.iloc[0:0].copy(), target_test, target_col, features, continuous, x_scaler, y_scaler, [heldout])
        ds = strict.StrictSolarDataset(base, target_test, features, LOOKBACK, args.horizon, min_start_t=ALIGNMENT_ROWS)
        city_out = out/"cities"/strict.safe_name(heldout); city_out.mkdir(parents=True, exist_ok=True)
        ds.window_metadata().to_csv(city_out/"test_window_metadata.csv.gz", index=False, compression="gzip")
        ds.window_counts_by_city().to_csv(city_out/"test_window_count.csv", index=False)

        seed_map = {int(x["seed"]): x for x in fl["seed_artifacts"]}
        for seed in SEEDS:
            art = seed_map[seed]
            checkpoint = Path(art["checkpoint"]); corr_path = Path(art["correction"])
            if sha256(checkpoint) != art["checkpoint_sha256"] or sha256(corr_path) != art["correction_sha256"]:
                raise RuntimeError("Frozen seed artifact changed")
            correction = json.loads(corr_path.read_text(encoding="utf-8"))
            model = strict.load_model_checkpoint(base, train_args, len(features), checkpoint, device)
            metrics, ph, pred, true, day = strict.evaluate_checkpoint(base, model, ds, y_scaler, correction, train_args, device)
            per_city_rows.append({"target_city": heldout, "track": args.track, "horizon": args.horizon, "lookback": LOOKBACK, "seed": seed, "n_windows": len(ds), **metrics})
            ph.insert(0, "target_city", heldout); ph.insert(1, "track", args.track); ph.insert(2, "seed", seed); ph_rows.append(ph)
            en = strict.energy_by_location(ds, pred, train_args.reference_capacity_kwp); en.insert(0, "seed", seed); en.insert(0, "track", args.track); energy_rows.append(en)
            sdir = city_out/f"seed_{seed}"; sdir.mkdir(parents=True, exist_ok=True)
            save_rich_npz(sdir/"test_predictions.npz", ds, pred, true, day, heldout, seed)
            pooled[seed]["pred"].append(pred); pooled[seed]["true"].append(true); pooled[seed]["day"].append(day); total_windows[seed] += len(ds)

    per_city_df = pd.DataFrame(per_city_rows); per_city_df.to_csv(out/"zero_shot_test_by_city_and_seed.csv", index=False)
    pd.concat(ph_rows, ignore_index=True).to_csv(out/"zero_shot_test_per_horizon_by_city.csv", index=False)
    energy_df = pd.concat(energy_rows, ignore_index=True); energy_df.to_csv(out/"zero_shot_test_energy_by_city.csv", index=False)

    pooled_rows=[]; pooled_ph=[]
    for seed in SEEDS:
        pred=np.concatenate(pooled[seed]["pred"],axis=0); true=np.concatenate(pooled[seed]["true"],axis=0); day=np.concatenate(pooled[seed]["day"],axis=0)
        met=base.compute_all_metrics(pred,true,day,train_args.reference_capacity_kwp)
        pooled_rows.append({"scope":"loco_zero_shot_pooled","track":args.track,"horizon":args.horizon,"lookback":LOOKBACK,"seed":seed,"n_windows":total_windows[seed],**met})
        ph=base.per_horizon_metrics(pred,true,day,train_args.reference_capacity_kwp); ph.insert(0,"seed",seed); ph.insert(0,"track",args.track); pooled_ph.append(ph)
    pooled_df=pd.DataFrame(pooled_rows); pooled_df.to_csv(out/"zero_shot_pooled_test_seed_results.csv",index=False)
    ddof1_summary(pooled_df).to_csv(out/"zero_shot_pooled_test_summary_ddof1.csv",index=False)
    pd.concat(pooled_ph,ignore_index=True).to_csv(out/"zero_shot_pooled_test_per_horizon.csv",index=False)

    energy_agg=energy_df.groupby("seed",as_index=False).agg(
        actual_reference_energy_mwh=("actual_reference_energy_mwh","sum"),
        predicted_reference_energy_mwh=("predicted_reference_energy_mwh","sum"),
        reference_energy_error_mwh=("reference_energy_error_mwh","sum"),
        n_unique_target_hours_total=("n_unique_target_hours","sum"),
    )
    energy_agg["energy_error_pct"]=100.0*energy_agg["reference_energy_error_mwh"]/energy_agg["actual_reference_energy_mwh"]
    energy_agg["aggregate_reference_capacity_kwp"]=15*train_args.reference_capacity_kwp
    energy_agg.to_csv(out/"zero_shot_aggregate_energy_by_seed.csv",index=False)

    audit={
        "status":"PASS","protocol_version":PROTOCOL_VERSION,"track":args.track,"horizon":args.horizon,"lookback":LOOKBACK,
        "selection_used_test":False,"all_four_selection_stage_locks_verified_before_test_open":True,
        "verified_stage_locks":seal["stage_locks"],"test_sha256":test_sha,"test_cities":all_test_cities,
        "seeds":list(SEEDS),"sample_sd_ddof":1,"n_folds":15,"total_windows_by_seed":total_windows,**code_hashes,
    }
    json_dump(out/"test_audit.json",audit)
    print(f"ZERO-SHOT TEST PASS: H{args.horizon} {args.track}")


def add_common(p):
    p.add_argument("--base_code", default="/workspace/lstm/loco_transfer_v2/solar_lstm_pv_per_kwp_v3.py")
    p.add_argument("--strict_reference", default="/workspace/lstm/loco_transfer_v2/solar_lstm_strict_protocol_reference.py")
    p.add_argument("--protocol_markdown", default="/workspace/lstm/loco_transfer_v2/LOCO_TRANSFER_PROTOCOL_v2.md")
    p.add_argument("--target_col", default="auto")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--verbose", action="store_true")


def parse_args():
    p=argparse.ArgumentParser(description="Leakage-controlled LOCO transfer protocol for solar LSTM")
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("preflight"); add_common(a)
    a.add_argument("--train_data",default="/workspace/lstm/data2/train.csv"); a.add_argument("--val_data",default="/workspace/lstm/data2/valid.csv"); a.add_argument("--out_dir",required=True)
    s=sub.add_parser("select-source"); add_common(s)
    s.add_argument("--train_data",default="/workspace/lstm/data2/train.csv"); s.add_argument("--val_data",default="/workspace/lstm/data2/valid.csv")
    s.add_argument("--preflight_lock",required=True); s.add_argument("--track",choices=TRACKS,required=True); s.add_argument("--horizon",type=int,choices=HORIZONS,required=True); s.add_argument("--out_dir",required=True); s.add_argument("--resume",action="store_true")
    e=sub.add_parser("evaluate-zero-shot"); add_common(e)
    e.add_argument("--selection_root",required=True); e.add_argument("--test_data",required=True); e.add_argument("--track",choices=TRACKS,required=True); e.add_argument("--horizon",type=int,choices=HORIZONS,required=True); e.add_argument("--out_dir",required=True)
    return p.parse_args()


def main():
    args=parse_args()
    if args.command=="preflight": preflight(args)
    elif args.command=="select-source": select_source(args)
    elif args.command=="evaluate-zero-shot": evaluate_zero_shot(args)
    else: raise ValueError(args.command)

if __name__=="__main__": main()
