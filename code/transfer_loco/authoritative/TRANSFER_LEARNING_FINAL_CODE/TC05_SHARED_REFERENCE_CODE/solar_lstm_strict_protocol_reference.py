#!/usr/bin/env python3
"""
Strict, validation-only protocol for the solar LSTM experiments.

Scientific rules enforced
-------------------------
1. Training uses 2015-2021 data only.
2. Checkpointing, early stopping, bias-correction choice, and architecture/lookback
   selection use validation data only.
3. Global lookback is selected by the mean corrected validation daylight RMSE
   across seeds 1, 2, and 3.
4. Test data are not loaded during the selection stage.
5. Final test results are reported as mean +/- population standard deviation across
   the three seeds; no "best test seed" is selected.
6. Global and location-specific LSTM use the same globally selected lookback for
   the same horizon. The location-specific path tunes no additional architecture.
7. Validation and test windows use a common evaluation start (normally 168 h), so
   candidate lookbacks are compared on the same forecast origins.
8. Raw final predictions are saved compactly as NPZ arrays plus one-row-per-window
   metadata. They can be reconstructed into a long table when needed.

Typical order
-------------
A. Select global lookback (does not read test.csv):
   python solar_lstm_strict_protocol.py select-global --horizon 24 ...
B. Evaluate selected global model on test:
   python solar_lstm_strict_protocol.py evaluate-global --selection_dir ... --test_data ...
C. Train/evaluate location-specific models with the same lookback:
   python solar_lstm_strict_protocol.py evaluate-location-specific --selection_dir ... --test_data ...
"""

from __future__ import annotations

import argparse
import copy
import gzip
import importlib.util
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def import_base(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("solar_lstm_base", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import base code: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class PreparedData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: Optional[pd.DataFrame]
    target_col: str
    feature_cols: List[str]
    continuous_cols: List[str]
    x_scaler: Optional[StandardScaler]
    y_scaler: StandardScaler
    cities: List[str]


class StrictSolarDataset(Dataset):
    """Dataset compatible with the original LSTM code, with fixed evaluation starts."""

    def __init__(
        self,
        base,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        lookback: int,
        horizon: int,
        min_start_t: Optional[int] = None,
    ) -> None:
        self.base = base
        self.feature_cols = list(feature_cols)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.min_start_t = max(self.lookback, int(min_start_t or self.lookback))

        self.x_by_city: List[np.ndarray] = []
        self.y_scaled_by_city: List[np.ndarray] = []
        self.y_raw_by_city: List[np.ndarray] = []
        self.daylight_by_city: List[np.ndarray] = []
        self.times_by_city: List[np.ndarray] = []
        self.city_names: List[str] = []
        self.index_map: List[Tuple[int, int]] = []

        for city, g in df.groupby(base.CITY_COL, sort=False):
            g = g.sort_values(base.TIME_COL).reset_index(drop=True)
            city_idx = len(self.city_names)
            self.city_names.append(str(city))
            self.x_by_city.append(np.ascontiguousarray(g[self.feature_cols].to_numpy(np.float32)))
            self.y_scaled_by_city.append(np.ascontiguousarray(g[base.TARGET_SCALED_COL].to_numpy(np.float32)))
            self.y_raw_by_city.append(np.ascontiguousarray(g[base.TARGET_VALUE_COL].to_numpy(np.float32)))
            self.daylight_by_city.append(np.ascontiguousarray(g[base.DAYLIGHT_COL].to_numpy(np.float32)))
            self.times_by_city.append(pd.to_datetime(g[base.TIME_COL], utc=True).to_numpy())

            n = len(g)
            for t in range(self.min_start_t, n - self.horizon + 1):
                self.index_map.append((city_idx, t))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        city_idx, t = self.index_map[idx]
        x = self.x_by_city[city_idx][t - self.lookback:t].copy()
        y_scaled = self.y_scaled_by_city[city_idx][t:t + self.horizon].copy()
        y_raw = self.y_raw_by_city[city_idx][t:t + self.horizon].copy()
        daylight = self.daylight_by_city[city_idx][t:t + self.horizon].copy()
        return (
            torch.from_numpy(x),
            torch.from_numpy(y_scaled),
            torch.from_numpy(y_raw),
            torch.from_numpy(daylight),
        )

    def window_metadata(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for window_id, (city_idx, t) in enumerate(self.index_map):
            times = self.times_by_city[city_idx]
            rows.append({
                "window_id": window_id,
                "city": self.city_names[city_idx],
                "forecast_origin": pd.Timestamp(times[t - 1]).isoformat(),
                "first_valid_time": pd.Timestamp(times[t]).isoformat(),
                "start_index": int(t),
            })
        return pd.DataFrame(rows)

    def window_city_indices(self) -> np.ndarray:
        return np.asarray([c for c, _ in self.index_map], dtype=np.int16)

    def window_start_indices(self) -> np.ndarray:
        return np.asarray([t for _, t in self.index_map], dtype=np.int32)

    def window_counts_by_city(self) -> pd.DataFrame:
        idx = self.window_city_indices()
        rows = []
        for city_idx, city in enumerate(self.city_names):
            rows.append({"city": city, "windows": int((idx == city_idx).sum()), "horizon": self.horizon})
        return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------


def _load_and_prepare_one(base, path: str, label: str, target_arg: str) -> Tuple[pd.DataFrame, str]:
    df, target_col = base.load_split(path, label, target_arg)
    df = base.prepare_physical_target(df, target_col)
    return df, target_col


def prepare_data(base, args, include_test: bool) -> PreparedData:
    train, train_target = _load_and_prepare_one(base, args.train_data, "train", args.target_col)
    val, val_target = _load_and_prepare_one(base, args.val_data, "valid", args.target_col)
    test = None
    test_target = train_target
    if include_test:
        if not args.test_data:
            raise ValueError("--test_data is required for an evaluation stage.")
        test, test_target = _load_and_prepare_one(base, args.test_data, "test", args.target_col)

    if not (train_target == val_target == test_target):
        raise ValueError(f"Target mismatch: train={train_target}, val={val_target}, test={test_target}")

    train_cities = sorted(train[base.CITY_COL].unique().tolist())
    val_cities = sorted(val[base.CITY_COL].unique().tolist())
    if train_cities != val_cities:
        raise ValueError(f"City mismatch between train and validation: {train_cities} vs {val_cities}")
    if test is not None:
        test_cities = sorted(test[base.CITY_COL].unique().tolist())
        if train_cities != test_cities:
            raise ValueError(f"City mismatch between train and test: {train_cities} vs {test_cities}")

    frames = [train, val] + ([test] if test is not None else [])
    common_cols = set(frames[0].columns)
    for df in frames[1:]:
        common_cols &= set(df.columns)
    common_cols = sorted(common_cols)
    train = train[common_cols].copy()
    val = val[common_cols].copy()
    if test is not None:
        test = test[common_cols].copy()

    if not args.no_solar_history:
        train = base.add_ar_solar_features(train, train_target)
        val = base.add_ar_solar_features(val, val_target)
        if test is not None:
            test = base.add_ar_solar_features(test, test_target)

    feature_cols, continuous_cols = base.build_feature_columns(
        train,
        feature_set=args.feature_set,
        use_solar_history=not args.no_solar_history,
    )
    required = feature_cols + [base.TARGET_VALUE_COL, base.RAW_TARGET_COL, base.DAYLIGHT_COL]
    for name, df in [("train", train), ("valid", val), ("test", test)]:
        if df is None:
            continue
        before = len(df)
        df.dropna(subset=required, inplace=True)
        print(f"{name}: kept {len(df):,} rows; dropped {before-len(df):,} rows after feature construction")

    x_scaler: Optional[StandardScaler] = None
    if continuous_cols:
        x_scaler = StandardScaler()
        train.loc[:, continuous_cols] = x_scaler.fit_transform(train[continuous_cols].to_numpy(np.float32))
        val.loc[:, continuous_cols] = x_scaler.transform(val[continuous_cols].to_numpy(np.float32))
        if test is not None:
            test.loc[:, continuous_cols] = x_scaler.transform(test[continuous_cols].to_numpy(np.float32))

    y_scaler = StandardScaler()
    train.loc[:, base.TARGET_SCALED_COL] = y_scaler.fit_transform(
        train[[base.TARGET_VALUE_COL]].to_numpy(np.float32)
    ).reshape(-1)
    val.loc[:, base.TARGET_SCALED_COL] = y_scaler.transform(
        val[[base.TARGET_VALUE_COL]].to_numpy(np.float32)
    ).reshape(-1)
    if test is not None:
        test.loc[:, base.TARGET_SCALED_COL] = y_scaler.transform(
            test[[base.TARGET_VALUE_COL]].to_numpy(np.float32)
        ).reshape(-1)

    return PreparedData(
        train=train,
        val=val,
        test=test,
        target_col=train_target,
        feature_cols=feature_cols,
        continuous_cols=continuous_cols,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        cities=train_cities,
    )


def save_preparation_artifacts(data: PreparedData, args, out_dir: Path, include_test: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_cols.json").write_text(json.dumps(data.feature_cols, indent=2), encoding="utf-8")
    (out_dir / "continuous_cols.json").write_text(json.dumps(data.continuous_cols, indent=2), encoding="utf-8")
    joblib.dump(data.y_scaler, out_dir / "y_scaler.pkl")
    if data.x_scaler is not None:
        joblib.dump(data.x_scaler, out_dir / "x_scaler.pkl")

    config = vars(args).copy()
    config.update({
        "resolved_target_col": data.target_col,
        "feature_cols": data.feature_cols,
        "continuous_cols": data.continuous_cols,
        "cities": data.cities,
        "include_test": include_test,
        "selection_rule": "minimum mean corrected validation daylight RMSE across seeds",
        "test_selection_rule": "test is never used for configuration or seed selection",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    (out_dir / "protocol_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")


# -----------------------------------------------------------------------------
# Model training and evaluation
# -----------------------------------------------------------------------------


def build_model(base, n_features: int, horizon: int, args, device: str):
    model = base.SolarLSTM(
        n_features=n_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        horizon=horizon,
        head_hidden=args.head_hidden,
    ).to(device)
    return model


def make_loader(ds: Dataset, args, shuffle: bool, device: str) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )


def json_dump(path: Path, obj: Any) -> None:
    def convert(x: Any):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, indent=2, default=convert), encoding="utf-8")


def train_validation_seed(
    base,
    train_ds: StrictSolarDataset,
    val_ds: StrictSolarDataset,
    data: PreparedData,
    args,
    lookback: int,
    seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base.set_seed(seed)
    model = build_model(base, len(data.feature_cols), args.horizon, args, device)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_loader = make_loader(train_ds, args, True, device)
    val_loader = make_loader(val_ds, args, False, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience
    )
    early = base.EarlyStopping(patience=args.early_patience, min_delta=args.min_delta)
    horizon_weights = base.build_horizon_weights(args.horizon, device)

    best_state = None
    best_epoch = 0
    best_monitor = float("inf")
    history: List[Dict[str, Any]] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = base.train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            horizon_weights=horizon_weights,
            device=device,
            use_city_embedding=False,
            grad_clip=args.grad_clip,
            loss_name=args.loss,
            huber_beta=args.huber_beta,
            daylight_weight=args.daylight_weight,
            night_weight=args.night_weight,
            production_weight=args.production_weight,
            max_kw_per_kwp=args.max_kw_per_kwp,
        )
        val_metrics_raw, _ = base.evaluate_model(
            model=model,
            loader=val_loader,
            y_scaler=data.y_scaler,
            device=device,
            use_city_embedding=False,
            reference_capacity_kwp=args.reference_capacity_kwp,
            clip_per_kwp=args.clip_per_kwp,
            max_kw_per_kwp=args.max_kw_per_kwp,
            force_night_zero=args.force_night_zero,
        )
        monitor = float(val_metrics_raw[args.monitor_metric])
        scheduler.step(monitor)
        stop, improved = early.step(monitor)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"val_raw_{k}": v for k, v in val_metrics_raw.items()},
        })
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_monitor = monitor
        if args.verbose:
            print(
                f"L{lookback} seed={seed} epoch={epoch:03d} train={train_loss:.6f} "
                f"val_day={val_metrics_raw.get('daylight_per_kwp_rmse', float('nan')):.6f}"
            )
        if stop:
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    val_pred, val_true, val_day = base.collect_predictions(
        model=model,
        loader=val_loader,
        y_scaler=data.y_scaler,
        device=device,
        use_city_embedding=False,
        clip_per_kwp=args.clip_per_kwp,
        max_kw_per_kwp=args.max_kw_per_kwp,
        force_night_zero=args.force_night_zero,
    )
    correction, calibration_df = base.choose_validation_correction(
        val_pred=val_pred,
        val_true=val_true,
        val_daylight=val_day,
        correction_methods=args.bias_corrections,
        selection_metric=args.calibration_metric,
        reference_capacity_kwp=args.reference_capacity_kwp,
        clip_per_kwp=args.clip_per_kwp,
        max_kw_per_kwp=args.max_kw_per_kwp,
        force_night_zero=args.force_night_zero,
    )
    val_pred_corr = base.apply_bias_correction(val_pred, val_day, correction)
    val_pred_corr = base.final_physical_postprocess(
        val_pred_corr,
        val_day,
        clip_per_kwp=args.clip_per_kwp,
        max_kw_per_kwp=args.max_kw_per_kwp,
        force_night_zero=args.force_night_zero,
    )
    val_metrics = base.compute_all_metrics(val_pred_corr, val_true, val_day, args.reference_capacity_kwp)
    val_per_horizon = base.per_horizon_metrics(val_pred_corr, val_true, val_day, args.reference_capacity_kwp)

    seed_dir = out_dir / f"L{lookback}" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = seed_dir / "best_checkpoint.pt"
    torch.save(best_state, checkpoint)
    json_dump(seed_dir / "correction.json", correction)
    pd.DataFrame(history).to_csv(seed_dir / "training_history.csv", index=False)
    calibration_df.to_csv(seed_dir / "validation_calibration_candidates.csv", index=False)
    val_per_horizon.to_csv(seed_dir / "validation_per_horizon.csv", index=False)

    row: Dict[str, Any] = {
        "lookback": lookback,
        "horizon": args.horizon,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_uncorrected_monitor": best_monitor,
        "selected_correction_method": correction.get("method", "none"),
        "checkpoint": str(checkpoint),
        "correction_json": str(seed_dir / "correction.json"),
        "n_parameters": n_params,
        "train_windows": len(train_ds),
        "validation_windows": len(val_ds),
        "training_seconds": time.time() - start_time,
        **{f"val_{k}": v for k, v in val_metrics.items()},
    }
    pd.DataFrame([row]).to_csv(seed_dir / "validation_seed_result.csv", index=False)
    return row


def load_model_checkpoint(base, args, n_features: int, checkpoint: Path, device: str):
    model = build_model(base, n_features, args.horizon, args, device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_checkpoint(
    base,
    model,
    ds: StrictSolarDataset,
    y_scaler: StandardScaler,
    correction: Dict[str, Any],
    args,
    device: str,
) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    loader = make_loader(ds, args, False, device)
    pred, true, day = base.collect_predictions(
        model=model,
        loader=loader,
        y_scaler=y_scaler,
        device=device,
        use_city_embedding=False,
        clip_per_kwp=args.clip_per_kwp,
        max_kw_per_kwp=args.max_kw_per_kwp,
        force_night_zero=args.force_night_zero,
    )
    pred = base.apply_bias_correction(pred, day, correction)
    pred = base.final_physical_postprocess(
        pred, day,
        clip_per_kwp=args.clip_per_kwp,
        max_kw_per_kwp=args.max_kw_per_kwp,
        force_night_zero=args.force_night_zero,
    )
    metrics = base.compute_all_metrics(pred, true, day, args.reference_capacity_kwp)
    ph = base.per_horizon_metrics(pred, true, day, args.reference_capacity_kwp)
    return metrics, ph, pred, true, day


# -----------------------------------------------------------------------------
# Additional reports
# -----------------------------------------------------------------------------


def per_location_metrics(base, ds: StrictSolarDataset, pred: np.ndarray, true: np.ndarray, day: np.ndarray, capacity: float) -> pd.DataFrame:
    city_idx = ds.window_city_indices()
    rows = []
    for idx, city in enumerate(ds.city_names):
        mask = city_idx == idx
        m = base.compute_all_metrics(pred[mask], true[mask], day[mask], capacity)
        rows.append({"city": city, "n_windows": int(mask.sum()), **m})
    return pd.DataFrame(rows)


def energy_by_location(ds: StrictSolarDataset, pred: np.ndarray, capacity: float, timestep_hours: float = 1.0) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    by_city_windows: Dict[int, List[Tuple[int, int]]] = {}
    for window_pos, (city_idx, t) in enumerate(ds.index_map):
        by_city_windows.setdefault(city_idx, []).append((window_pos, t))

    for city_idx, windows in by_city_windows.items():
        n = len(ds.y_raw_by_city[city_idx])
        sums = np.zeros(n, dtype=np.float64)
        counts = np.zeros(n, dtype=np.int32)
        for window_pos, t in windows:
            target_indices = np.arange(t, t + ds.horizon)
            sums[target_indices] += pred[window_pos].astype(np.float64)
            counts[target_indices] += 1
        valid = counts > 0
        mean_pred = np.zeros(n, dtype=np.float64)
        mean_pred[valid] = sums[valid] / counts[valid]
        true_series = ds.y_raw_by_city[city_idx].astype(np.float64)
        day_series = ds.daylight_by_city[city_idx] > 0.5

        actual_per_kwp = float(true_series[valid].sum() * timestep_hours)
        predicted_per_kwp = float(mean_pred[valid].sum() * timestep_hours)
        error_per_kwp = predicted_per_kwp - actual_per_kwp
        day_valid = valid & day_series
        rows.append({
            "city": ds.city_names[city_idx],
            "n_unique_target_hours": int(valid.sum()),
            "n_unique_daylight_hours": int(day_valid.sum()),
            "first_valid_time": pd.Timestamp(ds.times_by_city[city_idx][np.flatnonzero(valid)[0]]).isoformat(),
            "last_valid_time": pd.Timestamp(ds.times_by_city[city_idx][np.flatnonzero(valid)[-1]]).isoformat(),
            "actual_energy_kwh_per_kwp": actual_per_kwp,
            "predicted_energy_kwh_per_kwp": predicted_per_kwp,
            "energy_error_kwh_per_kwp": error_per_kwp,
            "energy_error_pct": 100.0 * error_per_kwp / max(abs(actual_per_kwp), 1e-12),
            "actual_reference_energy_mwh": actual_per_kwp * capacity / 1000.0,
            "predicted_reference_energy_mwh": predicted_per_kwp * capacity / 1000.0,
            "reference_energy_error_mwh": error_per_kwp * capacity / 1000.0,
        })
    return pd.DataFrame(rows)


def aggregate_mean_std(seed_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    id_cols = [c for c in ["scope", "horizon", "lookback"] if c in seed_df.columns]
    numeric = [c for c in seed_df.select_dtypes(include=[np.number]).columns if c not in {"seed"}]
    row: Dict[str, Any] = {}
    for c in id_cols:
        vals = seed_df[c].dropna().unique()
        row[c] = vals[0] if len(vals) == 1 else ",".join(map(str, vals))
    row["n_seeds"] = int(seed_df["seed"].nunique())
    for c in numeric:
        row[f"{c}_mean"] = float(seed_df[c].mean())
        row[f"{c}_std"] = float(seed_df[c].std(ddof=0))
    return pd.DataFrame([row])


def save_compact_predictions(path: Path, ds: StrictSolarDataset, pred: np.ndarray, true: np.ndarray, day: np.ndarray) -> None:
    np.savez_compressed(
        path,
        pred=pred.astype(np.float32),
        true=true.astype(np.float32),
        daylight=day.astype(np.float32),
        window_city_idx=ds.window_city_indices(),
        window_start_idx=ds.window_start_indices(),
    )


# -----------------------------------------------------------------------------
# Stages
# -----------------------------------------------------------------------------


def select_global(base, args) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = prepare_data(base, args, include_test=False)
    save_preparation_artifacts(data, args, out, include_test=False)
    common_start = args.common_eval_start or max(args.lookbacks)

    seed_rows: List[Dict[str, Any]] = []
    count_rows: List[pd.DataFrame] = []
    for lookback in args.lookbacks:
        if lookback <= 0:
            raise ValueError("Lookbacks must be positive")
        train_ds = StrictSolarDataset(base, data.train, data.feature_cols, lookback, args.horizon, min_start_t=lookback)
        val_ds = StrictSolarDataset(base, data.val, data.feature_cols, lookback, args.horizon, min_start_t=common_start)
        if len(train_ds) == 0 or len(val_ds) == 0:
            raise ValueError(f"No windows for lookback={lookback}")
        for split_name, ds in [("train", train_ds), ("validation", val_ds)]:
            cdf = ds.window_counts_by_city()
            cdf.insert(0, "split", split_name)
            cdf.insert(0, "lookback", lookback)
            count_rows.append(cdf)
        for seed in args.seeds:
            print(f"\n=== SELECT GLOBAL H{args.horizon} L{lookback} seed {seed} ===")
            seed_rows.append(train_validation_seed(base, train_ds, val_ds, data, args, lookback, seed, out))

    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(out / "global_validation_seed_results.csv", index=False)
    pd.concat(count_rows, ignore_index=True).to_csv(out / "global_selection_sample_counts_by_location.csv", index=False)
    metrics = [
        "val_per_kwp_rmse", "val_per_kwp_mae", "val_per_kwp_r2", "val_bias_pct_of_reference_capacity",
        "val_daylight_per_kwp_rmse", "val_daylight_per_kwp_mae", "val_daylight_per_kwp_r2",
        "val_reference_kw_rmse", "val_reference_kw_mae", "best_epoch", "training_seconds",
    ]
    rows = []
    for lookback, g in seeds.groupby("lookback", sort=True):
        row: Dict[str, Any] = {
            "lookback": int(lookback),
            "horizon": args.horizon,
            "n_seeds": int(g["seed"].nunique()),
            "seeds": ",".join(map(str, sorted(g["seed"].astype(int).unique()))),
            "common_validation_start_t": common_start,
            "selected_correction_methods": ",".join(sorted(g["selected_correction_method"].astype(str).unique())),
        }
        for m in metrics:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_std"] = float(g[m].std(ddof=0))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("val_daylight_per_kwp_rmse_mean").reset_index(drop=True)
    summary.to_csv(out / "global_validation_config_summary.csv", index=False)
    selected = summary.iloc[[0]].copy()
    selected["selection_metric"] = "val_daylight_per_kwp_rmse_mean"
    selected["selection_used_test"] = False
    selected.to_csv(out / "selected_global_lstm_config.csv", index=False)
    print("\nSelected lookback using validation only:")
    print(selected.to_string(index=False))


def _read_selected(selection_dir: Path) -> Tuple[pd.Series, pd.DataFrame]:
    selected_path = selection_dir / "selected_global_lstm_config.csv"
    seeds_path = selection_dir / "global_validation_seed_results.csv"
    if not selected_path.exists() or not seeds_path.exists():
        raise FileNotFoundError("Selection outputs are incomplete")
    selected = pd.read_csv(selected_path).iloc[0]
    seed_rows = pd.read_csv(seeds_path)
    seed_rows = seed_rows[seed_rows["lookback"] == int(selected["lookback"])].copy()
    return selected, seed_rows


def evaluate_global(base, args) -> None:
    selection_dir = Path(args.selection_dir)
    selected, selected_seeds = _read_selected(selection_dir)
    args.horizon = int(selected["horizon"])
    lookback = int(selected["lookback"])
    common_start = int(selected.get("common_validation_start_t", args.common_eval_start or 168))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = prepare_data(base, args, include_test=True)
    save_preparation_artifacts(data, args, out, include_test=True)
    assert data.test is not None
    test_ds = StrictSolarDataset(base, data.test, data.feature_cols, lookback, args.horizon, min_start_t=common_start)
    test_ds.window_metadata().to_csv(out / "test_window_metadata.csv.gz", index=False, compression="gzip")
    test_ds.window_counts_by_city().to_csv(out / "global_test_sample_counts_by_location.csv", index=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_metric_rows = []
    all_ph = []
    all_city = []
    all_energy = []
    for _, srow in selected_seeds.sort_values("seed").iterrows():
        seed = int(srow["seed"])
        checkpoint = Path(srow["checkpoint"])
        correction = json.loads(Path(srow["correction_json"]).read_text(encoding="utf-8"))
        model = load_model_checkpoint(base, args, len(data.feature_cols), checkpoint, device)
        metrics, ph, pred, true, day = evaluate_checkpoint(base, model, test_ds, data.y_scaler, correction, args, device)
        row = {
            "scope": "global",
            "horizon": args.horizon,
            "lookback": lookback,
            "seed": seed,
            "selected_correction_method": correction.get("method", "none"),
            **{f"test_{k}": v for k, v in metrics.items()},
        }
        seed_metric_rows.append(row)
        ph.insert(0, "seed", seed); ph.insert(0, "scope", "global")
        all_ph.append(ph)
        city = per_location_metrics(base, test_ds, pred, true, day, args.reference_capacity_kwp)
        city.insert(0, "seed", seed); city.insert(0, "scope", "global")
        all_city.append(city)
        energy = energy_by_location(test_ds, pred, args.reference_capacity_kwp)
        energy.insert(0, "seed", seed); energy.insert(0, "scope", "global")
        all_energy.append(energy)
        save_compact_predictions(out / f"test_predictions_seed_{seed}.npz", test_ds, pred, true, day)

    seed_df = pd.DataFrame(seed_metric_rows)
    seed_df.to_csv(out / "global_test_seed_results.csv", index=False)
    aggregate_mean_std(seed_df, "test").to_csv(out / "global_test_summary_mean_std.csv", index=False)
    pd.concat(all_ph, ignore_index=True).to_csv(out / "global_test_per_horizon.csv", index=False)
    pd.concat(all_city, ignore_index=True).to_csv(out / "global_test_per_location.csv", index=False)
    energy_df = pd.concat(all_energy, ignore_index=True)
    energy_df.to_csv(out / "global_test_energy_by_location.csv", index=False)
    energy_agg = energy_df.groupby("seed", as_index=False).agg(
        actual_reference_energy_mwh=("actual_reference_energy_mwh", "sum"),
        predicted_reference_energy_mwh=("predicted_reference_energy_mwh", "sum"),
        reference_energy_error_mwh=("reference_energy_error_mwh", "sum"),
        n_unique_target_hours_total=("n_unique_target_hours", "sum"),
    )
    energy_agg["energy_error_pct"] = 100.0 * energy_agg["reference_energy_error_mwh"] / energy_agg["actual_reference_energy_mwh"]
    energy_agg["aggregate_reference_capacity_kwp"] = len(data.cities) * args.reference_capacity_kwp
    energy_agg.to_csv(out / "global_test_aggregate_energy_by_seed.csv", index=False)


def _subset_city(df: pd.DataFrame, base, city: str) -> pd.DataFrame:
    out = df[df[base.CITY_COL].astype(str) == str(city)].copy()
    if out.empty:
        raise ValueError(f"No rows for city {city}")
    return out



def select_location_specific(base, args) -> None:
    """Tune each location independently using the same grid, seeds, and validation metric."""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = prepare_data_without_scaling(base, args, include_test=False)
    common_start = args.common_eval_start or max(args.lookbacks)
    save_raw_protocol_config(raw, args, out, lookback=-1, common_start=common_start)

    all_rows: List[Dict[str, Any]] = []
    count_rows: List[Dict[str, Any]] = []
    for city in raw["cities"]:
        city_root = out / "cities" / safe_name(city)
        city_root.mkdir(parents=True, exist_ok=True)
        city_data = scale_city_data(base, raw, city)
        for lookback in args.lookbacks:
            train_ds = StrictSolarDataset(base, city_data.train, city_data.feature_cols, lookback, args.horizon, min_start_t=lookback)
            val_ds = StrictSolarDataset(base, city_data.val, city_data.feature_cols, lookback, args.horizon, min_start_t=common_start)
            if len(train_ds) == 0 or len(val_ds) == 0:
                raise ValueError(f"No windows for city={city}, lookback={lookback}")
            count_rows.extend([
                {"city": city, "lookback": lookback, "split": "train", "windows": len(train_ds), "horizon": args.horizon},
                {"city": city, "lookback": lookback, "split": "validation", "windows": len(val_ds), "horizon": args.horizon},
            ])
            for seed in args.seeds:
                print(f"\n=== SELECT LOCATION {city} H{args.horizon} L{lookback} seed {seed} ===")
                row = train_validation_seed(base, train_ds, val_ds, city_data, args, lookback, int(seed), city_root)
                row["city"] = city
                all_rows.append(row)

    seeds = pd.DataFrame(all_rows)
    seeds.to_csv(out / "location_validation_seed_results.csv", index=False)
    pd.DataFrame(count_rows).to_csv(out / "location_selection_sample_counts.csv", index=False)
    metric_cols = [
        "val_per_kwp_rmse", "val_per_kwp_mae", "val_per_kwp_r2", "val_bias_pct_of_reference_capacity",
        "val_daylight_per_kwp_rmse", "val_daylight_per_kwp_mae", "val_daylight_per_kwp_r2",
        "val_reference_kw_rmse", "val_reference_kw_mae", "best_epoch", "training_seconds",
    ]
    summaries: List[Dict[str, Any]] = []
    for (city, lookback), g in seeds.groupby(["city", "lookback"], sort=True):
        row: Dict[str, Any] = {
            "city": city,
            "lookback": int(lookback),
            "horizon": args.horizon,
            "n_seeds": int(g["seed"].nunique()),
            "seeds": ",".join(map(str, sorted(g["seed"].astype(int).unique()))),
            "common_validation_start_t": common_start,
            "selected_correction_methods": ",".join(sorted(g["selected_correction_method"].astype(str).unique())),
        }
        for m in metric_cols:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_std"] = float(g[m].std(ddof=0))
        summaries.append(row)
    summary = pd.DataFrame(summaries).sort_values(["city", "val_daylight_per_kwp_rmse_mean"]).reset_index(drop=True)
    summary.to_csv(out / "location_validation_config_summary.csv", index=False)
    selected_rows = []
    for city, g in summary.groupby("city", sort=True):
        best = g.sort_values("val_daylight_per_kwp_rmse_mean").iloc[0].to_dict()
        best["selection_metric"] = "val_daylight_per_kwp_rmse_mean"
        best["selection_used_test"] = False
        selected_rows.append(best)
    selected = pd.DataFrame(selected_rows).sort_values("city").reset_index(drop=True)
    selected.to_csv(out / "selected_location_lstm_configs.csv", index=False)
    print("\nSelected location-specific configurations using validation only:")
    print(selected[["city", "lookback", "val_daylight_per_kwp_rmse_mean", "val_daylight_per_kwp_rmse_std"]].to_string(index=False))


def _read_location_selected(selection_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    config_path = selection_dir / "selected_location_lstm_configs.csv"
    seeds_path = selection_dir / "location_validation_seed_results.csv"
    if not config_path.exists() or not seeds_path.exists():
        raise FileNotFoundError("Location-selection outputs are incomplete")
    return pd.read_csv(config_path), pd.read_csv(seeds_path)


def evaluate_location_specific(base, args) -> None:
    selection_dir = Path(args.location_selection_dir)
    selected_configs, validation_seeds = _read_location_selected(selection_dir)
    horizons = selected_configs["horizon"].astype(int).unique()
    if len(horizons) != 1:
        raise ValueError(f"Expected one horizon, found {horizons}")
    args.horizon = int(horizons[0])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = prepare_data_without_scaling(base, args, include_test=True)
    assert raw["test"] is not None
    json_dump(out / "selected_location_lstm_configs.json", selected_configs.to_dict(orient="records"))

    pooled_by_seed: Dict[int, Dict[str, List[np.ndarray]]] = {
        int(seed): {"pred": [], "true": [], "day": []} for seed in args.seeds
    }
    all_seed_city_rows: List[Dict[str, Any]] = []
    all_ph: List[pd.DataFrame] = []
    all_energy: List[pd.DataFrame] = []
    test_count_rows: List[Dict[str, Any]] = []

    for _, cfg in selected_configs.sort_values("city").iterrows():
        city = str(cfg["city"])
        lookback = int(cfg["lookback"])
        common_start = int(cfg.get("common_validation_start_t", args.common_eval_start or 168))
        city_dir = out / "cities" / safe_name(city)
        city_dir.mkdir(parents=True, exist_ok=True)
        city_data = scale_city_data(base, raw, city)
        assert city_data.test is not None
        test_ds = StrictSolarDataset(base, city_data.test, city_data.feature_cols, lookback, args.horizon, min_start_t=common_start)
        test_ds.window_metadata().to_csv(city_dir / "test_window_metadata.csv.gz", index=False, compression="gzip")
        test_count_rows.append({"city": city, "lookback": lookback, "horizon": args.horizon, "test_windows": len(test_ds)})

        selected_seed_rows = validation_seeds[
            (validation_seeds["city"].astype(str) == city) &
            (validation_seeds["lookback"].astype(int) == lookback) &
            (validation_seeds["seed"].astype(int).isin([int(s) for s in args.seeds]))
        ].copy()
        if len(selected_seed_rows) != len(args.seeds):
            raise ValueError(f"Missing selected validation checkpoints for {city}, L{lookback}")

        for _, vrow in selected_seed_rows.sort_values("seed").iterrows():
            seed = int(vrow["seed"])
            checkpoint = Path(vrow["checkpoint"])
            correction = json.loads(Path(vrow["correction_json"]).read_text(encoding="utf-8"))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = load_model_checkpoint(base, args, len(city_data.feature_cols), checkpoint, device)
            metrics, ph, pred, true, day = evaluate_checkpoint(base, model, test_ds, city_data.y_scaler, correction, args, device)
            all_seed_city_rows.append({
                "scope": "location_specific_tuned",
                "city": city,
                "horizon": args.horizon,
                "lookback": lookback,
                "seed": seed,
                "selected_correction_method": correction.get("method", "none"),
                "validation_selection_score": float(vrow["val_daylight_per_kwp_rmse"]),
                **{f"test_{k}": value for k, value in metrics.items()},
            })
            ph.insert(0, "city", city); ph.insert(0, "seed", seed); ph.insert(0, "scope", "location_specific_tuned")
            all_ph.append(ph)
            e = energy_by_location(test_ds, pred, args.reference_capacity_kwp)
            e.insert(0, "seed", seed); e.insert(0, "scope", "location_specific_tuned")
            all_energy.append(e)
            save_compact_predictions(city_dir / f"test_predictions_seed_{seed}.npz", test_ds, pred, true, day)
            pooled_by_seed[seed]["pred"].append(pred)
            pooled_by_seed[seed]["true"].append(true)
            pooled_by_seed[seed]["day"].append(day)

    city_seed_df = pd.DataFrame(all_seed_city_rows)
    city_seed_df.to_csv(out / "location_specific_test_by_city_and_seed.csv", index=False)
    pd.DataFrame(test_count_rows).to_csv(out / "location_specific_test_sample_counts.csv", index=False)
    pd.concat(all_ph, ignore_index=True).to_csv(out / "location_specific_test_per_horizon_by_city.csv", index=False)
    energy_df = pd.concat(all_energy, ignore_index=True)
    energy_df.to_csv(out / "location_specific_test_energy_by_city.csv", index=False)

    pooled_rows = []
    for seed, parts in pooled_by_seed.items():
        pred = np.concatenate(parts["pred"], axis=0)
        true = np.concatenate(parts["true"], axis=0)
        day = np.concatenate(parts["day"], axis=0)
        metrics = base.compute_all_metrics(pred, true, day, args.reference_capacity_kwp)
        pooled_rows.append({
            "scope": "location_specific_tuned_pooled",
            "horizon": args.horizon,
            "seed": seed,
            **{f"test_{k}": value for k, value in metrics.items()},
        })
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(out / "location_specific_pooled_test_seed_results.csv", index=False)
    aggregate_mean_std(pooled_df, "test").to_csv(out / "location_specific_pooled_test_summary_mean_std.csv", index=False)

    energy_agg = energy_df.groupby("seed", as_index=False).agg(
        actual_reference_energy_mwh=("actual_reference_energy_mwh", "sum"),
        predicted_reference_energy_mwh=("predicted_reference_energy_mwh", "sum"),
        reference_energy_error_mwh=("reference_energy_error_mwh", "sum"),
        n_unique_target_hours_total=("n_unique_target_hours", "sum"),
    )
    energy_agg["energy_error_pct"] = 100.0 * energy_agg["reference_energy_error_mwh"] / energy_agg["actual_reference_energy_mwh"]
    energy_agg["aggregate_reference_capacity_kwp"] = len(raw["cities"]) * args.reference_capacity_kwp
    energy_agg.to_csv(out / "location_specific_aggregate_energy_by_seed.csv", index=False)


def prepare_data_without_scaling(base, args, include_test: bool) -> Dict[str, Any]:
    train, train_target = _load_and_prepare_one(base, args.train_data, "train", args.target_col)
    val, val_target = _load_and_prepare_one(base, args.val_data, "valid", args.target_col)
    test = None
    if include_test:
        test, test_target = _load_and_prepare_one(base, args.test_data, "test", args.target_col)
        if test_target != train_target:
            raise ValueError("Target mismatch")
    if val_target != train_target:
        raise ValueError("Target mismatch")
    frames = [train, val] + ([test] if test is not None else [])
    common = set(frames[0].columns)
    for f in frames[1:]: common &= set(f.columns)
    common = sorted(common)
    train = train[common].copy(); val = val[common].copy()
    if test is not None: test = test[common].copy()
    if not args.no_solar_history:
        train = base.add_ar_solar_features(train, train_target)
        val = base.add_ar_solar_features(val, val_target)
        if test is not None: test = base.add_ar_solar_features(test, train_target)
    feature_cols, continuous_cols = base.build_feature_columns(train, args.feature_set, not args.no_solar_history)
    required = feature_cols + [base.TARGET_VALUE_COL, base.RAW_TARGET_COL, base.DAYLIGHT_COL]
    for df in [train, val, test]:
        if df is not None: df.dropna(subset=required, inplace=True)
    cities = sorted(set(train[base.CITY_COL]) & set(val[base.CITY_COL]) & (set(test[base.CITY_COL]) if test is not None else set(train[base.CITY_COL])))
    return {
        "train": train, "val": val, "test": test, "target_col": train_target,
        "feature_cols": feature_cols, "continuous_cols": continuous_cols, "cities": cities,
    }


def scale_city_data(base, raw: Dict[str, Any], city: str) -> PreparedData:
    train = _subset_city(raw["train"], base, city)
    val = _subset_city(raw["val"], base, city)
    test = _subset_city(raw["test"], base, city) if raw["test"] is not None else None
    continuous = raw["continuous_cols"]
    x_scaler = None
    if continuous:
        x_scaler = StandardScaler()
        train.loc[:, continuous] = x_scaler.fit_transform(train[continuous].to_numpy(np.float32))
        val.loc[:, continuous] = x_scaler.transform(val[continuous].to_numpy(np.float32))
        if test is not None: test.loc[:, continuous] = x_scaler.transform(test[continuous].to_numpy(np.float32))
    y_scaler = StandardScaler()
    train.loc[:, base.TARGET_SCALED_COL] = y_scaler.fit_transform(train[[base.TARGET_VALUE_COL]].to_numpy(np.float32)).reshape(-1)
    val.loc[:, base.TARGET_SCALED_COL] = y_scaler.transform(val[[base.TARGET_VALUE_COL]].to_numpy(np.float32)).reshape(-1)
    if test is not None: test.loc[:, base.TARGET_SCALED_COL] = y_scaler.transform(test[[base.TARGET_VALUE_COL]].to_numpy(np.float32)).reshape(-1)
    return PreparedData(train, val, test, raw["target_col"], raw["feature_cols"], continuous, x_scaler, y_scaler, [city])


def save_raw_protocol_config(raw: Dict[str, Any], args, out: Path, lookback: int, common_start: int) -> None:
    config = vars(args).copy()
    config.update({
        "selected_global_lookback": lookback,
        "common_evaluation_start_t": common_start,
        "location_specific_architecture_tuning": False,
        "selection_used_test": False,
        "feature_cols": raw["feature_cols"],
        "continuous_cols": raw["continuous_cols"],
        "cities": raw["cities"],
    })
    json_dump(out / "protocol_config.json", config)
    json_dump(out / "feature_cols.json", raw["feature_cols"])


def safe_name(value: str) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return s or "location"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def add_common(parser: argparse.ArgumentParser, require_test: bool = False) -> None:
    parser.add_argument("--base_code", type=Path, default=Path("/workspace/lstm/solar_lstm_pv_per_kwp_v3.py"))
    parser.add_argument("--train_data", default="/workspace/lstm/data2/train.csv")
    parser.add_argument("--val_data", default="/workspace/lstm/data2/valid.csv")
    parser.add_argument("--test_data", default="/workspace/lstm/data2/test.csv" if require_test else None)
    parser.add_argument("--target_col", default="auto")
    parser.add_argument("--feature_set", choices=["basic", "physics"], default="basic")
    parser.add_argument("--no_solar_history", action="store_true")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--common_eval_start", type=int, default=168)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--head_hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss", choices=["huber", "mse", "mae"], default="huber")
    parser.add_argument("--huber_beta", type=float, default=1.0)
    parser.add_argument("--daylight_weight", type=float, default=4.0)
    parser.add_argument("--night_weight", type=float, default=0.25)
    parser.add_argument("--production_weight", type=float, default=1.0)
    parser.add_argument("--monitor_metric", default="daylight_per_kwp_rmse")
    parser.add_argument("--calibration_metric", default="daylight_per_kwp_rmse")
    parser.add_argument("--bias_corrections", nargs="+", default=["none", "global_bias", "daylight_bias", "horizon_bias", "horizon_daylight_bias"])
    parser.add_argument("--early_patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--reference_capacity_kwp", type=float, default=3370.0)
    parser.add_argument("--max_kw_per_kwp", type=float, default=1.20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_clip_per_kwp", action="store_true")
    parser.add_argument("--no_force_night_zero", action="store_true")
    parser.add_argument("--verbose", action="store_true")


def parse_args():
    p = argparse.ArgumentParser(description="Strict validation-only solar LSTM protocol")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("select-global", help="Tune global lookback using validation only; test is not read")
    add_common(s, require_test=False)
    s.add_argument("--lookbacks", nargs="+", type=int, default=[24, 48, 72, 168])
    s.add_argument("--out_dir", required=True)

    g = sub.add_parser("evaluate-global", help="Evaluate the validation-selected global configuration on test")
    add_common(g, require_test=True)
    g.add_argument("--selection_dir", required=True)
    g.add_argument("--out_dir", required=True)

    ls = sub.add_parser("select-location-specific", help="Tune each location using validation only; test is not read")
    add_common(ls, require_test=False)
    ls.add_argument("--lookbacks", nargs="+", type=int, default=[24, 48, 72, 168])
    ls.add_argument("--out_dir", required=True)

    c = sub.add_parser("evaluate-location-specific", help="Evaluate the validation-selected location-specific configurations on test")
    add_common(c, require_test=True)
    c.add_argument("--location_selection_dir", required=True)
    c.add_argument("--out_dir", required=True)

    args = p.parse_args()
    args.clip_per_kwp = not args.no_clip_per_kwp
    args.force_night_zero = not args.no_force_night_zero
    return args


def main() -> None:
    args = parse_args()
    base = import_base(args.base_code)
    if args.command == "select-global":
        select_global(base, args)
    elif args.command == "evaluate-global":
        evaluate_global(base, args)
    elif args.command == "select-location-specific":
        select_location_specific(base, args)
    elif args.command == "evaluate-location-specific":
        evaluate_location_specific(base, args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
