#!/usr/bin/env python3
"""
Original PatchTST solar PV re-execution protocol.

MODEL ARCHITECTURE
------------------
This program DOES NOT reimplement PatchTST. It imports the supervised Model
directly from the official yuqinie98/PatchTST repository, pinned to an exact
Git commit. If the commit or source path differs, execution stops.

STUDY PROTOCOL
--------------
- Train: 2015-2021
- Validation/calibration: 2022-2023
- Independent Test: 2024-2025
- Direct H24 and H48
- Candidate lookbacks for BOTH horizons: 24, 48, 72, 168 h
- Common Validation/Test origin warm-up: 168 h
- Seeds: 1, 2, 3; all retained; sample SD ddof=1
- Lookback/checkpoint/correction selected from Validation only
- Test is inaccessible to selection commands
- Global pooled and location-specific workflows
- Postprocess: night zero, nonnegative, clip to 1.2 kW/kWp
- Energy: average overlapping forecasts per city+valid timestamp, then sum
  and scale by 3370 kWp per city across 15 locations.

ORIGINAL PATCHTST INPUT MODE
----------------------------
Canonical univariate supervised mode (S): one channel containing strictly past
normalized AC PV target history. This preserves the original channel-independent
PatchTST architecture exactly. The existing 53-feature LSTM preprocessing is
constructed only to reproduce the same split-local initial history loss and
common evaluation origins; those 53 exogenous channels are NOT fused into a
custom PatchTST architecture.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


PINNED_PATCHTST_SHA = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"

LOOKBACKS = [24, 48, 72, 168]
SEEDS = [1, 2, 3]
COMMON_MAX_LOOKBACK = 168

# Fixed official PatchTST settings from the supervised implementation/examples.
PATCH_LEN = 16
STRIDE = 8
PADDING_PATCH = "end"
REVIN = 1
AFFINE = 0
SUBTRACT_LAST = 0
DECOMPOSITION = 0
INDIVIDUAL = 0
KERNEL_SIZE = 25

D_MODEL = 128
N_HEADS = 16
E_LAYERS = 3
D_FF = 256
DROPOUT = 0.20
FC_DROPOUT = 0.20
HEAD_DROPOUT = 0.0

LEARNING_RATE = 1e-4
BATCH_SIZE = 128
TRAIN_EPOCHS = 100
PATIENCE = 20

REFERENCE_CAPACITY_KWP = 3370.0
MAX_KW_PER_KWP = 1.20

CORRECTIONS = [
    "none",
    "global_bias",
    "daylight_bias",
    "horizon_bias",
    "horizon_daylight_bias",
]
SELECTION_METRIC = "daylight_per_kwp_rmse"

EXPECTED = {
    ("valid", 24): {
        "windows": 257415,
        "residuals": 6177960,
        "daylight_residuals": 3099655,
    },
    ("valid", 48): {
        "windows": 257055,
        "residuals": 12338640,
        "daylight_residuals": 6192138,
    },
    ("test", 24): {
        "windows": 257775,
        "residuals": 6186600,
        "daylight_residuals": 3103207,
        "unique_city_hours": 258120,
    },
    ("test", 48): {
        "windows": 257415,
        "residuals": 12355920,
        "daylight_residuals": 6199242,
        "unique_city_hours": 258120,
    },
}
EXPECTED_TRUE_ENERGY_GWH = 171.31476483913576


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def git(repo: Path, *args: str) -> str:
    cp = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return cp.stdout.strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_official_import(repo: Path):
    if not repo.is_dir():
        raise FileNotFoundError(f"Official repo missing: {repo}")

    actual_sha = git(repo, "rev-parse", "HEAD")
    if actual_sha != PINNED_PATCHTST_SHA:
        raise RuntimeError(
            "OFFICIAL PATCHTST COMMIT MISMATCH\n"
            f"expected: {PINNED_PATCHTST_SHA}\n"
            f"actual  : {actual_sha}"
        )

    # A pinned commit alone is insufficient if tracked files were edited locally.
    # Ignore untracked __pycache__ files, but fail on any tracked/staged change.
    tracked_changes = git(
        repo, "status", "--porcelain", "--untracked-files=no"
    )
    if tracked_changes:
        raise RuntimeError(
            "OFFICIAL PATCHTST WORKTREE IS NOT CLEAN\n"
            "Tracked/staged changes were detected under the pinned repository:\n"
            f"{tracked_changes}"
        )

    supervised = (repo / "PatchTST_supervised").resolve()
    required = [
        supervised / "models" / "PatchTST.py",
        supervised / "layers" / "PatchTST_backbone.py",
        supervised / "layers" / "PatchTST_layers.py",
        supervised / "layers" / "RevIN.py",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    if str(supervised) not in sys.path:
        sys.path.insert(0, str(supervised))

    from models import PatchTST  # type: ignore
    from layers.PatchTST_backbone import PatchTST_backbone, TSTiEncoder  # type: ignore

    model_source = Path(inspect.getsourcefile(PatchTST.Model) or "").resolve()
    expected_source = (supervised / "models" / "PatchTST.py").resolve()
    if model_source != expected_source:
        raise RuntimeError(
            "Imported PatchTST from an unexpected location:\n"
            f"actual  : {model_source}\nexpected: {expected_source}"
        )

    backbone_source = Path(
        inspect.getsourcefile(PatchTST_backbone) or ""
    ).resolve()
    encoder_source = Path(
        inspect.getsourcefile(TSTiEncoder) or ""
    ).resolve()
    expected_backbone = (
        supervised / "layers" / "PatchTST_backbone.py"
    ).resolve()
    if backbone_source != expected_backbone or encoder_source != expected_backbone:
        raise RuntimeError(
            "Imported PatchTST backbone/encoder from an unexpected location"
        )

    return PatchTST, PatchTST_backbone, TSTiEncoder, {
        "repo": str(repo.resolve()),
        "commit": actual_sha,
        "tracked_worktree_clean": True,
        "model_source": str(model_source),
        "backbone_class_source": str(backbone_source),
        "encoder_class_source": str(encoder_source),
        "model_source_sha256": sha256(expected_source),
        "backbone_source_sha256": sha256(supervised / "layers" / "PatchTST_backbone.py"),
        "layers_source_sha256": sha256(supervised / "layers" / "PatchTST_layers.py"),
        "revin_source_sha256": sha256(supervised / "layers" / "RevIN.py"),
    }


def make_config(lookback: int, horizon: int) -> SimpleNamespace:
    return SimpleNamespace(
        enc_in=1,
        seq_len=int(lookback),
        pred_len=int(horizon),
        e_layers=E_LAYERS,
        n_heads=N_HEADS,
        d_model=D_MODEL,
        d_ff=D_FF,
        dropout=DROPOUT,
        fc_dropout=FC_DROPOUT,
        head_dropout=HEAD_DROPOUT,
        individual=INDIVIDUAL,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        padding_patch=PADDING_PATCH,
        revin=REVIN,
        affine=AFFINE,
        subtract_last=SUBTRACT_LAST,
        decomposition=DECOMPOSITION,
        kernel_size=KERNEL_SIZE,
    )


def build_model(repo: Path, lookback: int, horizon: int, device: str) -> nn.Module:
    PatchTST, _, _, _ = ensure_official_import(repo)
    model = PatchTST.Model(make_config(lookback, horizon)).float().to(device)

    # Structural checks against the official classes/attributes.
    if model.__class__.__module__ != "models.PatchTST":
        raise AssertionError(f"Unexpected model module: {model.__class__.__module__}")
    if not hasattr(model, "model"):
        raise AssertionError("Expected original non-decomposition model.model")
    backbone = model.model
    if backbone.__class__.__name__ != "PatchTST_backbone":
        raise AssertionError(type(backbone))
    if backbone.backbone.__class__.__name__ != "TSTiEncoder":
        raise AssertionError(type(backbone.backbone))
    if not bool(backbone.revin):
        raise AssertionError("RevIN must be enabled")
    if int(backbone.patch_len) != PATCH_LEN or int(backbone.stride) != STRIDE:
        raise AssertionError("Patch/stride mismatch")
    if backbone.padding_patch != PADDING_PATCH:
        raise AssertionError("padding_patch mismatch")
    if bool(backbone.individual):
        raise AssertionError("individual head must be false")

    with torch.no_grad():
        probe = torch.zeros(2, lookback, 1, dtype=torch.float32, device=device)
        out = model(probe)
    if tuple(out.shape) != (2, horizon, 1):
        raise AssertionError(f"Unexpected output shape: {tuple(out.shape)}")
    return model


def preflight(repo: Path, base_code: Path) -> Dict[str, Any]:
    _, _, _, repo_meta = ensure_official_import(repo)
    if not base_code.is_file():
        raise FileNotFoundError(base_code)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    shapes = {}
    parameters = {}
    for h in (24, 48):
        for l in LOOKBACKS:
            model = build_model(repo, l, h, device)
            shapes[f"L{l}_H{h}"] = [2, h, 1]
            parameters[f"L{l}_H{h}"] = int(
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            )
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    return {
        "status": "PASS",
        "architecture_reimplemented": False,
        "official_model_imported_unmodified": True,
        "official_repo": repo_meta,
        "pinned_commit": PINNED_PATCHTST_SHA,
        "input_mode": "S",
        "input_channels": 1,
        "input_channel": "past normalized AC PV target",
        "channel_independence_preserved": True,
        "horizons": [24, 48],
        "lookbacks_for_each_horizon": LOOKBACKS,
        "common_max_lookback": COMMON_MAX_LOOKBACK,
        "architecture": {
            "patch_len": PATCH_LEN,
            "stride": STRIDE,
            "padding_patch": PADDING_PATCH,
            "revin": REVIN,
            "affine": AFFINE,
            "subtract_last": SUBTRACT_LAST,
            "decomposition": DECOMPOSITION,
            "individual": INDIVIDUAL,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "e_layers": E_LAYERS,
            "d_ff": D_FF,
            "dropout": DROPOUT,
            "fc_dropout": FC_DROPOUT,
            "head_dropout": HEAD_DROPOUT,
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "criterion": "MSE",
            "batch_size": BATCH_SIZE,
            "epochs": TRAIN_EPOCHS,
            "patience": PATIENCE,
            "lr_adjustment": "official type3",
        },
        "shape_checks": shapes,
        "parameter_counts": parameters,
        "base_lstm_code": str(base_code.resolve()),
        "base_lstm_code_sha256": sha256(base_code),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }


def load_base(base_code: Path):
    if not base_code.is_file():
        raise FileNotFoundError(base_code)
    return import_module_from_path("solar_lstm_base_for_original_patchtst", base_code)


def prepare_two_splits(base, train_path: Path, val_path: Path):
    train_df, target_col = base.load_split(str(train_path), "train", "auto")
    val_df, val_target = base.load_split(str(val_path), "valid", "auto")
    if val_target != target_col:
        raise RuntimeError(f"Target mismatch: {target_col} vs {val_target}")

    common_cols = sorted(set(train_df.columns) & set(val_df.columns))
    train_df = train_df[common_cols].copy()
    val_df = val_df[common_cols].copy()

    train_df = base.prepare_physical_target(train_df, target_col)
    val_df = base.prepare_physical_target(val_df, target_col)

    # Build the exact strict LSTM past-only history features only to reproduce
    # the exact same split-local initial history loss and row alignment.
    train_df = base.add_ar_solar_features(train_df, target_col)
    val_df = base.add_ar_solar_features(val_df, target_col)

    feature_cols, _ = base.build_feature_columns(
        train_df, feature_set="basic", use_solar_history=True
    )
    if len(feature_cols) != 53:
        raise RuntimeError(f"Expected strict 53-feature policy; found {len(feature_cols)}")

    required = list(feature_cols) + [
        base.TARGET_VALUE_COL,
        base.RAW_TARGET_COL,
        base.DAYLIGHT_COL,
    ]
    for label, df in [("train", train_df), ("valid", val_df)]:
        before = len(df)
        df.dropna(subset=required, inplace=True)
        print(f"{label}: dropped {before-len(df):,} rows after split-local history construction")

    t_year = pd.to_datetime(train_df[base.TIME_COL], utc=True).dt.year
    v_year = pd.to_datetime(val_df[base.TIME_COL], utc=True).dt.year
    if (int(t_year.min()), int(t_year.max())) != (2015, 2021):
        raise RuntimeError("Train is not exactly 2015-2021")
    if (int(v_year.min()), int(v_year.max())) != (2022, 2023):
        raise RuntimeError("Validation is not exactly 2022-2023")
    if pd.to_datetime(train_df[base.TIME_COL], utc=True).max() >= pd.to_datetime(
        val_df[base.TIME_COL], utc=True
    ).min():
        raise RuntimeError("Train/Validation overlap")
    print("Split integrity check: PASSED")

    scaler = StandardScaler()
    train_df[base.TARGET_SCALED_COL] = scaler.fit_transform(
        train_df[[base.TARGET_VALUE_COL]].to_numpy(np.float32)
    )
    val_df[base.TARGET_SCALED_COL] = scaler.transform(
        val_df[[base.TARGET_VALUE_COL]].to_numpy(np.float32)
    )
    return train_df, val_df, target_col, list(feature_cols), scaler


def rescale_scope_target(base, train_df: pd.DataFrame, val_df: pd.DataFrame, city: Optional[str]):
    """Fit target scaling on Training only for the requested scope.

    Global scope uses all locations. Location-specific scope fits a separate
    target scaler for each city, matching the strict LSTM location-specific
    protocol and the natural single-series data-loader behavior.
    """
    if city is None:
        tr = train_df.copy()
        va = val_df.copy()
    else:
        tr = train_df[train_df[base.CITY_COL].astype(str) == str(city)].copy()
        va = val_df[val_df[base.CITY_COL].astype(str) == str(city)].copy()
        if tr.empty or va.empty:
            raise RuntimeError(f"Missing scope rows for city: {city}")

    scaler = StandardScaler()
    tr[base.TARGET_SCALED_COL] = scaler.fit_transform(
        tr[[base.TARGET_VALUE_COL]].to_numpy(np.float32)
    ).reshape(-1)
    va[base.TARGET_SCALED_COL] = scaler.transform(
        va[[base.TARGET_VALUE_COL]].to_numpy(np.float32)
    ).reshape(-1)
    return tr, va, scaler


def prepare_test_split(
    base,
    test_path: Path,
    target_col: str,
    locked_feature_cols: Sequence[str],
):
    test_df, test_target = base.load_split(str(test_path), "test", "auto")
    if test_target != target_col:
        raise RuntimeError(f"Locked target {target_col} != Test target {test_target}")

    test_df = base.prepare_physical_target(test_df, target_col)
    test_df = base.add_ar_solar_features(test_df, target_col)

    missing = [c for c in locked_feature_cols if c not in test_df.columns]
    if missing:
        raise RuntimeError(f"Test missing locked columns: {missing}")

    required = list(locked_feature_cols) + [
        base.TARGET_VALUE_COL,
        base.RAW_TARGET_COL,
        base.DAYLIGHT_COL,
    ]
    before = len(test_df)
    test_df.dropna(subset=required, inplace=True)
    print(f"test: dropped {before-len(test_df):,} rows after split-local history construction")

    years = pd.to_datetime(test_df[base.TIME_COL], utc=True).dt.year
    if (int(years.min()), int(years.max())) != (2024, 2025):
        raise RuntimeError("Test is not exactly 2024-2025")

    # Scaling is deliberately deferred until evaluation because each
    # location-specific model has its own Training-fitted target scaler.
    print("Independent Test integrity check: PASSED")
    return test_df


class SolarPatchTSTDataset(Dataset):
    """
    Original PatchTST S-mode windows.

    Input at forecast origin:
        y[t-L : t]  -> [L, 1]
    Direct target:
        y[t : t+H]  -> [H]

    For Validation/Test, common_start=168 gives identical forecast origins for
    L24/L48/L72/L168. Training uses all eligible windows for each L.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        base,
        lookback: int,
        horizon: int,
        common_start: Optional[int],
        city_filter: Optional[str] = None,
    ):
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.groups: List[Dict[str, Any]] = []
        self.index: List[Tuple[int, int]] = []

        work = df
        if city_filter is not None:
            work = work[work[base.CITY_COL].astype(str) == str(city_filter)].copy()

        for city, g in work.groupby(base.CITY_COL, sort=True):
            g = g.sort_values(base.TIME_COL).reset_index(drop=True)
            group_idx = len(self.groups)
            group = {
                "city": str(city),
                "ys": np.ascontiguousarray(
                    g[base.TARGET_SCALED_COL].to_numpy(np.float32)
                ),
                "yr": np.ascontiguousarray(
                    g[base.TARGET_VALUE_COL].to_numpy(np.float32)
                ),
                "day": np.ascontiguousarray(
                    g[base.DAYLIGHT_COL].to_numpy(np.float32)
                ),
                "ts": pd.to_datetime(
                    g[base.TIME_COL], utc=True
                ).to_numpy(dtype="datetime64[ns]"),
            }
            self.groups.append(group)
            start = self.lookback if common_start is None else max(
                self.lookback, int(common_start)
            )
            for t in range(start, len(g) - self.horizon + 1):
                self.index.append((group_idx, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        gi, t = self.index[idx]
        g = self.groups[gi]
        x = g["ys"][t-self.lookback:t, None].copy()
        y = g["ys"][t:t+self.horizon].copy()
        yr = g["yr"][t:t+self.horizon].copy()
        day = g["day"][t:t+self.horizon].copy()
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(yr),
            torch.from_numpy(day),
        )

    def metadata_arrays(self):
        cities, origins, valids = [], [], []
        for gi, t in self.index:
            g = self.groups[gi]
            cities.append(g["city"])
            origins.append(
                g["ts"][t-1].astype("datetime64[ns]").astype(np.int64)
            )
            valids.append(
                g["ts"][t:t+self.horizon]
                .astype("datetime64[ns]")
                .astype(np.int64)
            )
        return (
            np.asarray(cities, dtype=str),
            np.asarray(origins, dtype=np.int64),
            np.asarray(valids, dtype=np.int64),
        )


def make_loader(ds: Dataset, shuffle: bool, device: str) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )


@torch.no_grad()
def collect(model, loader, device: str):
    model.eval()
    pred_s, true_s, true_r, daylight = [], [], [], []
    for x, y_s, y_r, day in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        if tuple(out.shape[1:]) != (y_s.shape[1], 1):
            raise RuntimeError(f"Unexpected output {tuple(out.shape)}")
        pred_s.append(out[..., 0].detach().cpu().numpy())
        true_s.append(y_s.numpy())
        true_r.append(y_r.numpy())
        daylight.append(day.numpy())
    return (
        np.concatenate(pred_s),
        np.concatenate(true_s),
        np.concatenate(true_r),
        np.concatenate(daylight),
    )


def inverse_scaled(z: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return (np.asarray(z, np.float64) * float(scale) + float(mean)).astype(
        np.float32
    )


def fit_correction(
    method: str,
    pred: np.ndarray,
    true: np.ndarray,
    daylight: np.ndarray,
) -> Dict[str, Any]:
    err = np.asarray(pred, np.float64) - np.asarray(true, np.float64)
    day = np.asarray(daylight) > 0.5
    if method == "none":
        return {"method": "none"}
    if method == "global_bias":
        return {"method": method, "global_bias": float(np.mean(err))}
    if method == "daylight_bias":
        gb = float(np.mean(err))
        return {
            "method": method,
            "global_bias": gb,
            "daylight_bias": float(np.mean(err[day])) if np.any(day) else gb,
            "night_bias": float(np.mean(err[~day])) if np.any(~day) else gb,
        }
    if method == "horizon_bias":
        return {
            "method": method,
            "global_bias": float(np.mean(err)),
            "horizon_bias": np.mean(err, axis=0).astype(float).tolist(),
        }
    if method == "horizon_daylight_bias":
        h_all = np.mean(err, axis=0)
        h_day = np.zeros(err.shape[1], dtype=float)
        h_night = np.zeros(err.shape[1], dtype=float)
        for h in range(err.shape[1]):
            md = day[:, h]
            mn = ~md
            h_day[h] = np.mean(err[md, h]) if np.any(md) else h_all[h]
            h_night[h] = np.mean(err[mn, h]) if np.any(mn) else h_all[h]
        return {
            "method": method,
            "global_bias": float(np.mean(err)),
            "horizon_bias": h_all.astype(float).tolist(),
            "horizon_daylight_bias": h_day.astype(float).tolist(),
            "horizon_night_bias": h_night.astype(float).tolist(),
        }
    raise ValueError(method)


def apply_correction(
    pred: np.ndarray,
    daylight: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    p = np.asarray(pred, np.float64).copy()
    day = np.asarray(daylight) > 0.5
    method = params["method"]

    if method == "none":
        pass
    elif method == "global_bias":
        p -= float(params["global_bias"])
    elif method == "daylight_bias":
        p[day] -= float(params["daylight_bias"])
        p[~day] -= float(params["night_bias"])
    elif method == "horizon_bias":
        p -= np.asarray(params["horizon_bias"])[None, :]
    elif method == "horizon_daylight_bias":
        hd = np.asarray(params["horizon_daylight_bias"])[None, :]
        hn = np.asarray(params["horizon_night_bias"])[None, :]
        p -= np.where(day, hd, hn)
    else:
        raise ValueError(method)
    return p.astype(np.float32)


def postprocess(pred: np.ndarray, daylight: np.ndarray) -> np.ndarray:
    p = np.asarray(pred, np.float32).copy()
    p[np.asarray(daylight) <= 0.5] = 0.0
    return np.clip(p, 0.0, MAX_KW_PER_KWP)


def metric_dict(
    pred: np.ndarray,
    true: np.ndarray,
    daylight: np.ndarray,
) -> Dict[str, float]:
    p = np.asarray(pred, np.float64).reshape(-1)
    y = np.asarray(true, np.float64).reshape(-1)
    day = np.asarray(daylight).reshape(-1) > 0.5
    e = p - y

    den = np.sum((y - np.mean(y)) ** 2)
    out = {
        "per_kwp_rmse": float(np.sqrt(np.mean(e**2))),
        "per_kwp_mae": float(np.mean(np.abs(e))),
        "per_kwp_r2": float(1 - np.sum(e**2)/den) if den > 0 else np.nan,
        "bias_per_kwp": float(np.mean(e)),
        "reference_kw_rmse": float(
            np.sqrt(np.mean(e**2)) * REFERENCE_CAPACITY_KWP
        ),
        "reference_kw_mae": float(
            np.mean(np.abs(e)) * REFERENCE_CAPACITY_KWP
        ),
        "count": int(e.size),
        "daylight_count": int(day.sum()),
    }
    if np.any(day):
        ed = e[day]
        yd = y[day]
        dden = np.sum((yd - np.mean(yd)) ** 2)
        out.update({
            "daylight_per_kwp_rmse": float(np.sqrt(np.mean(ed**2))),
            "daylight_per_kwp_mae": float(np.mean(np.abs(ed))),
            "daylight_per_kwp_r2": float(
                1 - np.sum(ed**2)/dden
            ) if dden > 0 else np.nan,
            "daylight_bias_per_kwp": float(np.mean(ed)),
        })
    return out


def choose_correction(
    pred: np.ndarray,
    true: np.ndarray,
    daylight: np.ndarray,
):
    rows, best, best_score = [], None, float("inf")
    for method in CORRECTIONS:
        params = fit_correction(method, pred, true, daylight)
        corrected = postprocess(
            apply_correction(pred, daylight, params), daylight
        )
        m = metric_dict(corrected, true, daylight)
        rows.append({
            "method": method,
            **m,
            "params_json": json.dumps(params, sort_keys=True),
        })
        score = float(m[SELECTION_METRIC])
        if score < best_score - 1e-15:
            best_score = score
            best = params
    if best is None:
        raise RuntimeError("No correction selected")
    return best, pd.DataFrame(rows)


def type3_lr(epoch: int) -> float:
    # Exact official type3 formula in utils/tools.py.
    return (
        LEARNING_RATE
        if epoch < 3
        else LEARNING_RATE * (0.9 ** (epoch - 3))
    )


def train_one_seed(
    repo: Path,
    train_ds: SolarPatchTSTDataset,
    val_ds: SolarPatchTSTDataset,
    lookback: int,
    horizon: int,
    seed: int,
    checkpoint: Path,
    y_mean: float,
    y_scale: float,
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(repo, lookback, horizon, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_loader = make_loader(train_ds, True, device)
    val_loader = make_loader(val_ds, False, device)

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    bad = 0
    history = []

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        train_loss = []
        for x, y_s, _, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)[..., 0]
            loss = criterion(pred, y_s)
            loss.backward()
            optimizer.step()
            train_loss.append(float(loss.item()))

        val_pred_s, val_true_s, _, _ = collect(model, val_loader, device)
        val_mse = float(np.mean((val_pred_s - val_true_s) ** 2))

        history.append({
            "epoch": epoch,
            "train_mse": float(np.mean(train_loss)),
            "validation_mse": val_mse,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        # Official type3 schedule is applied after each epoch.
        new_lr = type3_lr(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        if bad >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("Training never produced a checkpoint")
    model.load_state_dict(best_state)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)

    val_pred_s, _, val_true_r, val_day = collect(model, val_loader, device)
    val_pred_r = inverse_scaled(val_pred_s, y_mean, y_scale)
    correction, cal_table = choose_correction(
        val_pred_r, val_true_r, val_day
    )
    final_pred = postprocess(
        apply_correction(val_pred_r, val_day, correction), val_day
    )
    m = metric_dict(final_pred, val_true_r, val_day)

    return {
        **m,
        "lookback": lookback,
        "horizon": horizon,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_mse": best_val,
        "selected_correction_method": correction["method"],
        "correction_params": correction,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "history": history,
        "calibration": cal_table.to_dict(orient="records"),
    }


METRICS = [
    "per_kwp_rmse",
    "per_kwp_mae",
    "per_kwp_r2",
    "bias_per_kwp",
    "reference_kw_rmse",
    "reference_kw_mae",
    "daylight_per_kwp_rmse",
    "daylight_per_kwp_mae",
    "daylight_per_kwp_r2",
    "daylight_bias_per_kwp",
]


def summarize_seed_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, lookback, horizon), g in df.groupby(
        ["scope", "lookback", "horizon"], sort=True
    ):
        row = {
            "scope": scope,
            "lookback": int(lookback),
            "horizon": int(horizon),
            "n_seeds": len(g),
        }
        for col in METRICS:
            x = pd.to_numeric(g[col], errors="coerce").to_numpy(float)
            row[col + "_mean"] = float(np.mean(x))
            row[col + "_sd"] = float(np.std(x, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def validate_registered_grid(args):
    if list(args.lookbacks) != LOOKBACKS:
        raise RuntimeError(
            f"Registered lookbacks for BOTH H24/H48 are exactly {LOOKBACKS}; "
            f"received {args.lookbacks}"
        )
    if list(args.seeds) != SEEDS:
        raise RuntimeError(
            f"Registered principal seeds are exactly {SEEDS}; received {args.seeds}"
        )


def run_selection(args):
    validate_registered_grid(args)
    if args.horizon not in (24, 48):
        raise RuntimeError("Only H24 and H48 are registered")

    # Selection parser contains no Test argument by design.
    repo = Path(args.official_repo).resolve()
    base_code = Path(args.base_code).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    audit = preflight(repo, base_code)
    base = load_base(base_code)
    train_df, val_df, target_col, feature_cols, scaler = prepare_two_splits(
        base, Path(args.train_data), Path(args.val_data)
    )
    global_y_mean = float(scaler.mean_[0])
    global_y_scale = float(scaler.scale_[0])

    expected_val = EXPECTED[("valid", args.horizon)]

    if args.command == "select-global":
        scopes = [("__GLOBAL__", None)]
    else:
        train_cities = sorted(train_df[base.CITY_COL].astype(str).unique())
        val_cities = set(val_df[base.CITY_COL].astype(str).unique())
        cities = [c for c in train_cities if c in val_cities]
        if len(cities) != 15:
            raise RuntimeError(f"Expected 15 common cities; got {len(cities)}")
        scopes = [(city, city) for city in cities]

    protocol = {
        "status": "VALIDATION_ONLY_SELECTION",
        "selection_used_test": False,
        "test_data_loaded": False,
        "command": args.command,
        "train_period": "2015-2021",
        "validation_period": "2022-2023",
        "test_period": "2024-2025 (inaccessible during selection)",
        "horizon": args.horizon,
        "candidate_lookbacks": LOOKBACKS,
        "common_validation_warmup": COMMON_MAX_LOOKBACK,
        "seeds": SEEDS,
        "seed_policy": "retain all 3 seeds; no seed cherry-picking",
        "sample_sd_ddof": 1,
        "selection_metric": "mean corrected daylight RMSE across seeds",
        "input_mode": "S",
        "input_channel": "strictly past normalized PV target",
        "original_architecture_imported_unmodified": True,
        "official_pinned_commit": PINNED_PATCHTST_SHA,
        "strict_53_feature_policy_constructed_for_alignment_only": True,
        "strict_feature_count": len(feature_cols),
        "strict_feature_cols": feature_cols,
        "target_col": target_col,
        "scaler_policy": (
            "global_training_target"
            if args.command == "select-global"
            else "per_location_training_target"
        ),
        "global_training_target_scaler_audit": {
            "mean": global_y_mean,
            "scale": global_y_scale,
        },
        "corrections": CORRECTIONS,
        "postprocess": {
            "night_zero": True,
            "nonnegative": True,
            "max_kw_per_kwp": MAX_KW_PER_KWP,
        },
        "reference_capacity_kwp_per_city": REFERENCE_CAPACITY_KWP,
        "official_model_preflight": audit,
        "protocol_code_sha256": sha256(Path(__file__).resolve()),
        "base_code_sha256": sha256(base_code),
        "data_sha256": {
            "train": sha256(Path(args.train_data)),
            "valid": sha256(Path(args.val_data)),
        },
    }
    write_json(out / "protocol_config.json", protocol)

    seed_rows = []
    details = []
    scope_scalers: Dict[str, Tuple[float, float]] = {}
    for scope, city_filter in scopes:
        print("=" * 100)
        print(f"VALIDATION-ONLY: {scope} | H{args.horizon}")
        print("=" * 100)

        # Global uses one Training-only target scaler across all locations.
        # Location-specific models use one Training-only target scaler per city,
        # matching the strict LSTM location-specific protocol.
        scope_train_df, scope_val_df, scope_scaler = rescale_scope_target(
            base, train_df, val_df, city_filter
        )
        scope_y_mean = float(scope_scaler.mean_[0])
        scope_y_scale = float(scope_scaler.scale_[0])
        scope_scalers[scope] = (scope_y_mean, scope_y_scale)

        for lookback in LOOKBACKS:
            train_ds = SolarPatchTSTDataset(
                scope_train_df, base, lookback, args.horizon,
                common_start=None, city_filter=city_filter
            )
            val_ds = SolarPatchTSTDataset(
                scope_val_df, base, lookback, args.horizon,
                common_start=COMMON_MAX_LOOKBACK, city_filter=city_filter
            )

            if city_filter is None:
                if len(val_ds) != expected_val["windows"]:
                    raise RuntimeError(
                        f"Validation common-set mismatch L{lookback}: "
                        f"{len(val_ds)} != {expected_val['windows']}"
                    )
                # Count the registered common set without materializing a
                # second multi-million-element copy.
                residual_count = len(val_ds) * args.horizon
                daylight_count = 0
                for gi, t in val_ds.index:
                    daylight_count += int(
                        np.count_nonzero(
                            val_ds.groups[gi]["day"][
                                t:t+args.horizon
                            ] > 0.5
                        )
                    )
                if residual_count != expected_val["residuals"]:
                    raise RuntimeError("Validation residual count mismatch")
                if daylight_count != expected_val["daylight_residuals"]:
                    raise RuntimeError("Validation daylight count mismatch")

            print(
                f"{scope} L{lookback}: train_windows={len(train_ds):,} "
                f"validation_windows={len(val_ds):,}"
            )

            for seed in SEEDS:
                safe_scope = scope.replace("/", "_").replace(" ", "_")
                checkpoint = (
                    out / "checkpoints" / safe_scope /
                    f"L{lookback}_H{args.horizon}_seed{seed}.pth"
                )
                result = train_one_seed(
                    repo=repo,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    lookback=lookback,
                    horizon=args.horizon,
                    seed=seed,
                    checkpoint=checkpoint,
                    y_mean=scope_y_mean,
                    y_scale=scope_y_scale,
                )

                row = {
                    k: v for k, v in result.items()
                    if k not in ("history", "calibration", "correction_params")
                }
                row["scope"] = scope
                row["correction_params_json"] = json.dumps(
                    result["correction_params"], sort_keys=True
                )
                seed_rows.append(row)
                details.append({
                    "scope": scope,
                    "lookback": lookback,
                    "seed": seed,
                    "history": result["history"],
                    "calibration": result["calibration"],
                })

                pd.DataFrame(seed_rows).to_csv(
                    out / "validation_seed_results.csv", index=False
                )
                write_json(out / "training_details.json", details)

    seed_df = pd.DataFrame(seed_rows)
    summary = summarize_seed_rows(seed_df)
    summary.to_csv(out / "validation_config_summary.csv", index=False)

    selected_rows = []
    lock_entries = []
    for scope, g in summary.groupby("scope", sort=True):
        ranked = g.sort_values(
            [
                "daylight_per_kwp_rmse_mean",
                "per_kwp_rmse_mean",
                "lookback",
            ],
            ascending=True,
        )
        best = ranked.iloc[0]
        L = int(best["lookback"])
        selected_rows.append(best.to_dict())

        chosen = seed_df[
            (seed_df["scope"] == scope) &
            (seed_df["lookback"] == L)
        ].sort_values("seed")
        if list(chosen["seed"].astype(int)) != SEEDS:
            raise RuntimeError(f"{scope}: selected config missing seeds 1,2,3")

        scope_y_mean, scope_y_scale = scope_scalers[scope]
        lock_entries.append({
            "scope": scope,
            "horizon": args.horizon,
            "selected_lookback": L,
            "y_scaler_mean": scope_y_mean,
            "y_scaler_scale": scope_y_scale,
            "selection_metric": "daylight_per_kwp_rmse_mean",
            "selection_metric_value": float(
                best["daylight_per_kwp_rmse_mean"]
            ),
            "seeds": [
                {
                    "seed": int(r.seed),
                    "checkpoint": str(r.checkpoint),
                    "checkpoint_sha256": str(r.checkpoint_sha256),
                    "best_epoch": int(r.best_epoch),
                    "selected_correction_method": str(
                        r.selected_correction_method
                    ),
                    "correction_params": json.loads(
                        r.correction_params_json
                    ),
                }
                for r in chosen.itertuples(index=False)
            ],
        })

    pd.DataFrame(selected_rows).to_csv(
        out / "selected_configurations.csv", index=False
    )

    lock = {
        "lock_version": 2,
        "selection_complete": True,
        "selection_used_test": False,
        "selection_command": args.command,
        "horizon": args.horizon,
        "candidate_lookbacks": LOOKBACKS,
        "common_max_lookback": COMMON_MAX_LOOKBACK,
        "seeds": SEEDS,
        "all_seeds_retained": True,
        "sample_sd_ddof": 1,
        "target_col": target_col,
        "strict_feature_cols": feature_cols,
        "scaler_policy": (
            "global_training_target"
            if args.command == "select-global"
            else "per_location_training_target"
        ),
        "official_repo_commit": PINNED_PATCHTST_SHA,
        "architecture_reimplemented": False,
        "input_mode": "S",
        "entries": lock_entries,
        "protocol_code_sha256": sha256(Path(__file__).resolve()),
        "base_code_sha256": sha256(base_code),
        "train_data_sha256": sha256(Path(args.train_data)),
        "valid_data_sha256": sha256(Path(args.val_data)),
        "protocol_config_sha256": sha256(out / "protocol_config.json"),
    }
    write_json(out / "selection_lock.json", lock)

    lines = [
        "ORIGINAL PATCHTST — VALIDATION-ONLY SELECTION",
        "=" * 92,
        f"Scope command: {args.command}",
        f"Horizon: H{args.horizon}",
        f"Lookbacks compared: {LOOKBACKS}",
        f"Seeds retained: {SEEDS}",
        f"Common Validation warm-up: {COMMON_MAX_LOOKBACK} h",
        "Test loaded: NO",
        "Architecture reimplemented: NO",
        f"Official pinned commit: {PINNED_PATCHTST_SHA}",
        "Original input mode: S (strictly past normalized PV target)",
        "Patch/stride: fixed 16/8",
        "Configuration ranking: mean Validation daylight RMSE over seeds 1,2,3",
        "",
        "SELECTED CONFIGURATIONS",
    ]
    for e in lock_entries:
        lines.append(
            f"{e['scope']}: L{e['selected_lookback']} | "
            f"Validation daylight RMSE mean="
            f"{e['selection_metric_value']:.9f}"
        )
    (out / "final_selection_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


def torch_load_state(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def per_horizon(pred, true, day):
    rows = []
    for h in range(pred.shape[1]):
        rows.append({
            "horizon_step": h + 1,
            **metric_dict(
                pred[:, h:h+1],
                true[:, h:h+1],
                day[:, h:h+1],
            )
        })
    return pd.DataFrame(rows)


def by_city(pred, true, day, cities):
    rows = []
    arr = np.asarray(cities, dtype=str)
    for city in sorted(set(arr.tolist())):
        mask = arr == city
        rows.append({
            "city": city,
            **metric_dict(pred[mask], true[mask], day[mask])
        })
    return pd.DataFrame(rows)


def energy_from_overlaps(pred, true, cities, valid_ns):
    n, h = pred.shape
    city_rep = np.repeat(np.asarray(cities, dtype=str), h)
    valid_flat = np.asarray(valid_ns, dtype=np.int64).reshape(-1)
    df = pd.DataFrame({
        "city": city_rep,
        "valid_ns": valid_flat,
        "pred": np.asarray(pred, np.float64).reshape(-1),
        "true": np.asarray(true, np.float64).reshape(-1),
    })
    agg = df.groupby(
        ["city", "valid_ns"], sort=True, as_index=False
    )[["pred", "true"]].mean()

    pred_gwh = float(
        agg["pred"].sum() * REFERENCE_CAPACITY_KWP / 1e6
    )
    true_gwh = float(
        agg["true"].sum() * REFERENCE_CAPACITY_KWP / 1e6
    )
    return {
        "unique_city_hours": int(len(agg)),
        "true_energy_gwh": true_gwh,
        "predicted_energy_gwh": pred_gwh,
        "energy_error_gwh": pred_gwh - true_gwh,
        "energy_error_pct": 100.0 * (pred_gwh - true_gwh) / true_gwh,
    }


def evaluate_checkpoint(
    repo,
    base,
    test_df,
    entry,
    seed_info,
    y_mean,
    y_scale,
    raw_dir,
    city_filter,
):
    L = int(entry["selected_lookback"])
    H = int(entry["horizon"])
    seed = int(seed_info["seed"])

    if city_filter is None:
        eval_df = test_df.copy()
    else:
        eval_df = test_df[
            test_df[base.CITY_COL].astype(str) == str(city_filter)
        ].copy()
        if eval_df.empty:
            raise RuntimeError(f"No Test rows for city: {city_filter}")

    eval_df[base.TARGET_SCALED_COL] = (
        eval_df[base.TARGET_VALUE_COL].to_numpy(np.float64) - float(y_mean)
    ) / float(y_scale)

    ds = SolarPatchTSTDataset(
        eval_df, base, L, H,
        common_start=COMMON_MAX_LOOKBACK,
        city_filter=None,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(repo, L, H, device)

    checkpoint = Path(seed_info["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_sha = sha256(checkpoint)
    if actual_sha != seed_info["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint SHA mismatch: {checkpoint}")

    model.load_state_dict(
        torch_load_state(checkpoint, device), strict=True
    )
    loader = make_loader(ds, False, device)
    pred_s, _, true, day = collect(model, loader, device)
    pred = inverse_scaled(pred_s, y_mean, y_scale)
    pred = apply_correction(
        pred, day, seed_info["correction_params"]
    )
    pred = postprocess(pred, day)

    cities, origins, valid_ns = ds.metadata_arrays()
    e = energy_from_overlaps(pred, true, cities, valid_ns)
    m = metric_dict(pred, true, day)

    safe = (
        "GLOBAL"
        if city_filter is None
        else str(city_filter).replace("/", "_").replace(" ", "_")
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{safe}_H{H}_seed{seed}.npz"
    np.savez_compressed(
        raw_path,
        pred=pred.astype(np.float32),
        true=true.astype(np.float32),
        daylight=(day > 0.5).astype(np.uint8),
        city=np.asarray(cities, dtype=str),
        origin_ns=origins.astype(np.int64),
        valid_ns=valid_ns.astype(np.int64),
        lookback=np.int32(L),
        horizon=np.int32(H),
        seed=np.int32(seed),
    )

    return {
        **m,
        **e,
        "scope": entry["scope"],
        "lookback": L,
        "horizon": H,
        "seed": seed,
        "checkpoint_sha256": actual_sha,
        "prediction_npz": str(raw_path.resolve()),
        "_pred": pred,
        "_true": true,
        "_day": day,
        "_cities": cities,
        "_origins": origins,
        "_valid_ns": valid_ns,
    }



def validate_all_selection_locks_before_test(
    selection_dir: Path,
    current_protocol_sha: str,
    current_base_sha: str,
) -> Dict[str, Any]:
    """Hard core-level gate: Test cannot open until all four Validation locks exist.

    This duplicates the shell-wrapper protection inside the Python protocol so
    direct invocation of evaluate-* cannot bypass the sealed-Test ordering.
    It also verifies that all four locks were produced by the same protocol/base
    code and the same Training/Validation data.
    """
    experiment_root = selection_dir.parent.parent
    required = [
        ("H24_global", experiment_root / "H24" / "global_selection" / "selection_lock.json", 24, "select-global"),
        ("H24_location", experiment_root / "H24" / "location_selection" / "selection_lock.json", 24, "select-location-specific"),
        ("H48_global", experiment_root / "H48" / "global_selection" / "selection_lock.json", 48, "select-global"),
        ("H48_location", experiment_root / "H48" / "location_selection" / "selection_lock.json", 48, "select-location-specific"),
    ]

    audit = {}
    train_hashes = set()
    valid_hashes = set()

    for label, path, horizon, command in required:
        if not path.is_file():
            raise RuntimeError(
                "Independent Test is blocked until ALL FOUR Validation "
                f"selections are frozen. Missing: {path}"
            )

        item = json.loads(path.read_text(encoding="utf-8"))
        if not item.get("selection_complete"):
            raise RuntimeError(f"Incomplete selection lock: {path}")
        if item.get("selection_used_test"):
            raise RuntimeError(f"Selection lock indicates Test leakage: {path}")
        if int(item.get("horizon", -1)) != horizon:
            raise RuntimeError(f"Selection-lock horizon mismatch: {path}")
        if item.get("selection_command") != command:
            raise RuntimeError(f"Selection-lock scope mismatch: {path}")
        if item.get("candidate_lookbacks") != LOOKBACKS:
            raise RuntimeError(f"Selection-lock lookback grid mismatch: {path}")
        if item.get("seeds") != SEEDS or not item.get("all_seeds_retained"):
            raise RuntimeError(f"Selection-lock seed protocol mismatch: {path}")
        if item.get("official_repo_commit") != PINNED_PATCHTST_SHA:
            raise RuntimeError(f"Selection-lock official commit mismatch: {path}")
        if item.get("protocol_code_sha256") != current_protocol_sha:
            raise RuntimeError(
                f"Protocol code differs from frozen selection lock: {path}"
            )
        if item.get("base_code_sha256") != current_base_sha:
            raise RuntimeError(
                f"Base preprocessing code differs from frozen selection lock: {path}"
            )

        train_hashes.add(str(item.get("train_data_sha256")))
        valid_hashes.add(str(item.get("valid_data_sha256")))
        audit[label] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "horizon": horizon,
            "selection_command": command,
        }

    if len(train_hashes) != 1:
        raise RuntimeError(
            "The four selection locks do not share one Training-data SHA-256"
        )
    if len(valid_hashes) != 1:
        raise RuntimeError(
            "The four selection locks do not share one Validation-data SHA-256"
        )

    return {
        "all_four_selection_locks_verified": True,
        "experiment_root": str(experiment_root.resolve()),
        "train_data_sha256": next(iter(train_hashes)),
        "valid_data_sha256": next(iter(valid_hashes)),
        "locks": audit,
    }


def run_evaluation(args):
    selection_dir = Path(args.selection_dir).resolve()
    lock_path = selection_dir / "selection_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    if not lock.get("selection_complete"):
        raise RuntimeError("Selection lock is incomplete")
    if lock.get("selection_used_test"):
        raise RuntimeError("Selection lock indicates Test leakage")
    if lock["candidate_lookbacks"] != LOOKBACKS:
        raise RuntimeError("Selection did not compare registered lookbacks")
    if lock["seeds"] != SEEDS or not lock["all_seeds_retained"]:
        raise RuntimeError("Seed protocol mismatch")
    if lock["official_repo_commit"] != PINNED_PATCHTST_SHA:
        raise RuntimeError("Official commit mismatch in lock")

    expected_selection = (
        "select-global"
        if args.command == "evaluate-global"
        else "select-location-specific"
    )
    if lock["selection_command"] != expected_selection:
        raise RuntimeError(
            f"Scope mismatch: lock={lock['selection_command']} "
            f"evaluation={args.command}"
        )

    H = int(lock["horizon"])
    repo = Path(args.official_repo).resolve()
    base_code = Path(args.base_code).resolve()

    current_protocol_sha = sha256(Path(__file__).resolve())
    if current_protocol_sha != lock.get("protocol_code_sha256"):
        raise RuntimeError(
            "Protocol code changed after Selection; independent Test is blocked"
        )
    current_base_sha = sha256(base_code)
    if current_base_sha != lock.get("base_code_sha256"):
        raise RuntimeError(
            "Base preprocessing code changed after Selection; independent Test is blocked"
        )

    all_locks_audit = validate_all_selection_locks_before_test(
        selection_dir=selection_dir,
        current_protocol_sha=current_protocol_sha,
        current_base_sha=current_base_sha,
    )

    repo_audit = preflight(repo, base_code)
    base = load_base(base_code)

    # Test is opened here, after frozen selection_lock.json and code hashes
    # were validated. Target scaling is applied per frozen entry below.
    test_df = prepare_test_split(
        base=base,
        test_path=Path(args.test_data),
        target_col=lock["target_col"],
        locked_feature_cols=lock["strict_feature_cols"],
    )

    out = Path(args.out_dir).resolve()
    raw_dir = out / "raw_predictions"
    out.mkdir(parents=True, exist_ok=True)

    seed_scope_rows = []
    per_seed_chunks: Dict[int, List[Dict[str, Any]]] = {
        1: [], 2: [], 3: []
    }

    for entry in lock["entries"]:
        city_filter = (
            None
            if entry["scope"] == "__GLOBAL__"
            else entry["scope"]
        )
        for seed_info in entry["seeds"]:
            result = evaluate_checkpoint(
                repo=repo,
                base=base,
                test_df=test_df,
                entry=entry,
                seed_info=seed_info,
                y_mean=float(entry["y_scaler_mean"]),
                y_scale=float(entry["y_scaler_scale"]),
                raw_dir=raw_dir,
                city_filter=city_filter,
            )
            clean = {
                k: v for k, v in result.items()
                if not k.startswith("_")
            }
            seed_scope_rows.append(clean)
            per_seed_chunks[int(seed_info["seed"])].append(result)

    pd.DataFrame(seed_scope_rows).to_csv(
        out / "test_seed_scope_results.csv", index=False
    )

    pooled_rows = []
    ph_tables = []
    city_tables = []
    energy_rows = []

    for seed in SEEDS:
        chunks = per_seed_chunks[seed]
        if not chunks:
            raise RuntimeError(f"No Test chunks for seed {seed}")

        pred = np.concatenate([r["_pred"] for r in chunks], axis=0)
        true = np.concatenate([r["_true"] for r in chunks], axis=0)
        day = np.concatenate([r["_day"] for r in chunks], axis=0)
        cities = np.concatenate(
            [np.asarray(r["_cities"], dtype=str) for r in chunks]
        )
        valid_ns = np.concatenate(
            [r["_valid_ns"] for r in chunks], axis=0
        )

        m = metric_dict(pred, true, day)
        e = energy_from_overlaps(
            pred, true, cities, valid_ns
        )
        pooled_rows.append({
            "seed": seed,
            "horizon": H,
            **m,
            **e,
        })

        ph = per_horizon(pred, true, day)
        ph["seed"] = seed
        ph_tables.append(ph)

        bc = by_city(pred, true, day, cities)
        bc["seed"] = seed
        city_tables.append(bc)

        energy_rows.append({
            "seed": seed,
            "horizon": H,
            **e,
        })

    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(out / "test_pooled_seed_results.csv", index=False)
    pd.concat(ph_tables, ignore_index=True).to_csv(
        out / "test_per_horizon.csv", index=False
    )
    pd.concat(city_tables, ignore_index=True).to_csv(
        out / "test_by_city.csv", index=False
    )
    pd.DataFrame(energy_rows).to_csv(
        out / "test_energy_by_seed.csv", index=False
    )

    expected = EXPECTED[("test", H)]
    for row in pooled.itertuples(index=False):
        if int(row.count) != expected["residuals"]:
            raise RuntimeError(
                f"Residual count mismatch seed {row.seed}: "
                f"{row.count} != {expected['residuals']}"
            )
        if int(row.count // H) != expected["windows"]:
            raise RuntimeError("Window count mismatch")
        if int(row.daylight_count) != expected["daylight_residuals"]:
            raise RuntimeError("Daylight residual count mismatch")
        if int(row.unique_city_hours) != expected["unique_city_hours"]:
            raise RuntimeError("Unique city-hours mismatch")
        if abs(
            float(row.true_energy_gwh) - EXPECTED_TRUE_ENERGY_GWH
        ) > 1e-5:
            raise RuntimeError(
                f"True energy mismatch: {row.true_energy_gwh} vs "
                f"{EXPECTED_TRUE_ENERGY_GWH}"
            )

    summary = {
        "scope": (
            "global"
            if args.command == "evaluate-global"
            else "location-specific"
        ),
        "horizon": H,
        "n_seeds": 3,
        "sample_sd_ddof": 1,
        "selection_used_test": False,
        "architecture_reimplemented": False,
        "official_repo_commit": PINNED_PATCHTST_SHA,
        "input_mode": "S",
        "scaler_policy": lock.get("scaler_policy"),
    }
    summary_cols = METRICS + [
        "true_energy_gwh",
        "predicted_energy_gwh",
        "energy_error_gwh",
        "energy_error_pct",
    ]
    for col in summary_cols:
        x = pd.to_numeric(pooled[col], errors="coerce").to_numpy(float)
        summary[col + "_mean"] = float(np.mean(x))
        summary[col + "_sd"] = float(np.std(x, ddof=1))
    write_json(out / "test_summary.json", summary)

    test_audit = {
        "status": "PASS",
        "command": args.command,
        "horizon": H,
        "selection_lock_sha256": sha256(lock_path),
        "all_four_selection_locks_verified_before_test_open": all_locks_audit,
        "test_loaded_only_after_selection_lock_validation": True,
        "selection_used_test": False,
        "official_model_preflight": repo_audit,
        "expected_common_set": expected,
        "actual_common_set": {
            "windows": int(pooled.iloc[0]["count"] // H),
            "residuals": int(pooled.iloc[0]["count"]),
            "daylight_residuals": int(
                pooled.iloc[0]["daylight_count"]
            ),
            "unique_city_hours": int(
                pooled.iloc[0]["unique_city_hours"]
            ),
        },
        "expected_true_energy_gwh": EXPECTED_TRUE_ENERGY_GWH,
        "actual_true_energy_gwh": float(
            pooled.iloc[0]["true_energy_gwh"]
        ),
        "test_data_sha256": sha256(Path(args.test_data)),
        "protocol_code_sha256": current_protocol_sha,
        "base_code_sha256": current_base_sha,
        "scaler_policy": lock.get("scaler_policy"),
        "sample_sd_ddof": 1,
    }
    write_json(out / "test_audit.json", test_audit)

    lines = [
        "ORIGINAL PATCHTST — INDEPENDENT TEST",
        "=" * 92,
        f"Scope: {summary['scope']}",
        f"Horizon: H{H}",
        "Selection used Test: NO",
        "Architecture reimplemented: NO",
        f"Official pinned commit: {PINNED_PATCHTST_SHA}",
        f"Common windows: {test_audit['actual_common_set']['windows']:,}",
        f"Residuals: {test_audit['actual_common_set']['residuals']:,}",
        f"Daylight residuals: "
        f"{test_audit['actual_common_set']['daylight_residuals']:,}",
        f"Unique city-hours: "
        f"{test_audit['actual_common_set']['unique_city_hours']:,}",
        "",
        f"RMSE: {summary['per_kwp_rmse_mean']:.9f} ± "
        f"{summary['per_kwp_rmse_sd']:.9f}",
        f"MAE: {summary['per_kwp_mae_mean']:.9f} ± "
        f"{summary['per_kwp_mae_sd']:.9f}",
        f"R2: {summary['per_kwp_r2_mean']:.9f} ± "
        f"{summary['per_kwp_r2_sd']:.9f}",
        f"Daylight RMSE: "
        f"{summary['daylight_per_kwp_rmse_mean']:.9f} ± "
        f"{summary['daylight_per_kwp_rmse_sd']:.9f}",
        f"Daylight MAE: "
        f"{summary['daylight_per_kwp_mae_mean']:.9f} ± "
        f"{summary['daylight_per_kwp_mae_sd']:.9f}",
        f"True energy: "
        f"{summary['true_energy_gwh_mean']:.9f} GWh",
        f"Predicted energy: "
        f"{summary['predicted_energy_gwh_mean']:.9f} ± "
        f"{summary['predicted_energy_gwh_sd']:.9f} GWh",
        f"Energy error: "
        f"{summary['energy_error_gwh_mean']:.9f} ± "
        f"{summary['energy_error_gwh_sd']:.9f} GWh",
        "STATUS: PASS",
    ]
    (out / "final_test_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("preflight")
    q.add_argument("--official-repo", required=True)
    q.add_argument("--base-code", required=True)
    q.add_argument("--out", required=True)

    for command in ("select-global", "select-location-specific"):
        q = sub.add_parser(command)
        q.add_argument("--official-repo", required=True)
        q.add_argument("--base-code", required=True)
        q.add_argument("--train-data", required=True)
        q.add_argument("--val-data", required=True)
        q.add_argument(
            "--horizon", required=True, type=int, choices=[24, 48]
        )
        q.add_argument(
            "--lookbacks", nargs="+", type=int, default=LOOKBACKS
        )
        q.add_argument(
            "--seeds", nargs="+", type=int, default=SEEDS
        )
        q.add_argument("--out-dir", required=True)

    for command in ("evaluate-global", "evaluate-location-specific"):
        q = sub.add_parser(command)
        q.add_argument("--official-repo", required=True)
        q.add_argument("--base-code", required=True)
        q.add_argument("--selection-dir", required=True)
        q.add_argument("--test-data", required=True)
        q.add_argument("--out-dir", required=True)

    return p.parse_args()


def main():
    args = parse_args()
    if args.command == "preflight":
        result = preflight(
            Path(args.official_repo).resolve(),
            Path(args.base_code).resolve(),
        )
        write_json(Path(args.out), result)
        print(json.dumps(result, indent=2))
    elif args.command.startswith("select-"):
        run_selection(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
