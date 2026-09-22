#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# Frozen protocol constants
# -----------------------------------------------------------------------------

W = Path("/workspace/lstm")
BASE_CODE = W / "loco_transfer_v2" / "solar_lstm_pv_per_kwp_v3.py"
STRICT_CODE = W / "loco_transfer_v2" / "solar_lstm_strict_protocol_reference.py"
VALID_PATH = W / "data2" / "valid.csv"

A00B_REPORT = W / "output" / "loco_fewshot_A00b_binding" / "fewshot_A00b_binding_report.json"
FINAL_LOCK = W / "FEWSHOT_PROTOCOL_LOCK_FINAL_v1_1.json"

SOURCE_STAGE = W / "output" / "loco_transfer_v2" / "H48" / "cold_start" / "selection"
OUT = W / "output" / "loco_fewshot" / "H48" / "selection_refit"

EXPECTED_BASE_SHA = "41f7892cbdd296f6ea4d6887a5dd92669cd3a1a42ef0fb4695c29e7d79369438"
EXPECTED_STRICT_SHA = "24e6ad5b53f8c8a3160c8772eeb6bf74b931f0496a6674b13346b7fc87952f70"
EXPECTED_VALID_SHA = "86f95e8581b864e2cc6d67640bbcd802fdf7176716b5119aa721d1a64369ca5a"
EXPECTED_LOCK_SHA = "6558a8034efacdbae3b3b4a6002bf7108d9605e89f25bbe661380f1ad07dd351"

HORIZON = 48
LOOKBACK = 24
SEEDS = (1, 2, 3)
CITIES = (
    "Anbar","Babylon","Baghdad","Basra","Dhi_Qar","Diyala","Karbala",
    "Kirkuk","Maysan","Muthanna","Najaf","Nineveh","Qadisiyyah",
    "Salah_al_Din","Wasit",
)
BUDGETS = {
    "30d": {
        "selection_train": ("2023-12-02T00:00:00Z","2023-12-25T23:00:00Z"),
        "selection_validation": ("2023-12-26T00:00:00Z","2023-12-31T23:00:00Z"),
        "full_refit": ("2023-12-02T00:00:00Z","2023-12-31T23:00:00Z"),
        "expected": {"selection_train":529,"selection_validation":97,"full_refit":673},
    },
    "90d": {
        "selection_train": ("2023-10-03T00:00:00Z","2023-12-13T23:00:00Z"),
        "selection_validation": ("2023-12-14T00:00:00Z","2023-12-31T23:00:00Z"),
        "full_refit": ("2023-10-03T00:00:00Z","2023-12-31T23:00:00Z"),
        "expected": {"selection_train":1681,"selection_validation":385,"full_refit":2113},
    },
    "365d": {
        "selection_train": ("2023-01-01T00:00:00Z","2023-10-19T23:00:00Z"),
        "selection_validation": ("2023-10-20T00:00:00Z","2023-12-31T23:00:00Z"),
        "full_refit": ("2023-01-01T00:00:00Z","2023-12-31T23:00:00Z"),
        "expected": {"selection_train":6961,"selection_validation":1705,"full_refit":8713},
    },
}

# Adaptation hyperparameters frozen before any adaptation training.
LR = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
GRAD_CLIP = 1.0
MAX_EPOCHS = 30
EARLY_PATIENCE = 5
MIN_DELTA = 1e-5
HUBER_BETA = 1.0
DAYLIGHT_WEIGHT = 4.0
NIGHT_WEIGHT = 0.25
PRODUCTION_WEIGHT = 1.0
REFERENCE_CAPACITY_KWP = 3370.0
MAX_KW_PER_KWP = 1.20

TRAINABLE_KEYS = (
    "head.0.weight","head.0.bias",
    "head.3.weight","head.3.bias",
    "head.6.weight","head.6.bias",
)
EXPECTED_HEAD_PARAMS = 44272

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def safe_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def json_dump(path: Path, obj: Any) -> None:
    def conv(x):
        if isinstance(x, Path): return str(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        raise TypeError(type(x).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=conv), encoding="utf-8")

def import_file(name: str, path: Path):
    # Register before exec_module: dataclasses and some typing machinery expect
    # cls.__module__ to resolve through sys.modules while the module executes.
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod

def state_sha256(state: Dict[str, torch.Tensor]) -> str:
    # Deterministic content digest independent of torch.save container metadata.
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(t.dtype).encode("ascii"))
        h.update(np.asarray(t.shape, dtype=np.int64).tobytes())
        h.update(t.numpy().tobytes())
    return h.hexdigest()

def load_source_state(path: Path, device: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=device)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Checkpoint is not a state_dict: {path}")
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        obj = obj["model_state_dict"]
    if set(obj) != {
        "lstm.weight_ih_l0","lstm.weight_hh_l0","lstm.bias_ih_l0","lstm.bias_hh_l0",
        "lstm.weight_ih_l1","lstm.weight_hh_l1","lstm.bias_ih_l1","lstm.bias_hh_l1",
        "lstm.weight_ih_l2","lstm.weight_hh_l2","lstm.bias_ih_l2","lstm.bias_hh_l2",
        "head.0.weight","head.0.bias","head.3.weight","head.3.bias","head.6.weight","head.6.bias",
    }:
        raise RuntimeError(f"Unexpected checkpoint state signature: {path}")
    return obj

def build_model(base, n_features: int, device: str):
    return base.SolarLSTM(
        n_features=n_features,
        hidden_size=256,
        num_layers=3,
        dropout=0.20,
        horizon=HORIZON,
        head_hidden=128,
    ).to(device)

def bind_head_only(model) -> Tuple[List[torch.nn.Parameter], List[str], List[str]]:
    trainable = []
    tnames, fnames = [], []
    for name, p in model.named_parameters():
        if name in TRAINABLE_KEYS:
            p.requires_grad = True
            trainable.append(p)
            tnames.append(name)
        else:
            p.requires_grad = False
            fnames.append(name)
    if tuple(tnames) != TRAINABLE_KEYS:
        raise RuntimeError(f"Trainable head key mismatch: {tnames}")
    n = sum(p.numel() for p in trainable)
    if n != EXPECTED_HEAD_PARAMS:
        raise RuntimeError(f"Expected {EXPECTED_HEAD_PARAMS} trainable H48 head parameters, got {n}")
    if not fnames or not all(n.startswith("lstm.") for n in fnames):
        raise RuntimeError(f"Unexpected frozen parameters: {fnames}")
    return trainable, tnames, fnames

def set_adaptation_train_mode(model) -> None:
    # Critical frozen-protocol detail:
    # - LSTM backbone stays in eval mode => recurrent dropout disabled.
    # - MLP forecast head stays in train mode => head dropout active.
    model.eval()
    model.head.train()

def make_loader(ds, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

class TargetContainedDataset(Dataset):
    """One-city cold-start dataset with target-horizon containment.

    Predictor history may precede the labeled-budget start. Only target labels
    t..t+H-1 are required to lie fully inside [start, end]. This is the exact
    budget interpretation frozen in FEWSHOT_PROTOCOL_LOCK_FINAL_v1_1.json.
    """
    def __init__(self, base, prepared_city: pd.DataFrame, feature_cols: Sequence[str],
                 start: str, end: str):
        g = prepared_city.sort_values(base.TIME_COL).reset_index(drop=True)
        self.feature_cols = list(feature_cols)
        self.x = np.ascontiguousarray(g[self.feature_cols].to_numpy(np.float32)).copy()
        self.y_scaled = np.ascontiguousarray(g[base.TARGET_SCALED_COL].to_numpy(np.float32)).copy()
        self.y_raw = np.ascontiguousarray(g[base.TARGET_VALUE_COL].to_numpy(np.float32)).copy()
        self.daylight = np.ascontiguousarray(g[base.DAYLIGHT_COL].to_numpy(np.float32)).copy()
        self.times = pd.DatetimeIndex(pd.to_datetime(g[base.TIME_COL], utc=True))
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.index_map: List[int] = []
        n = len(g)
        for t in range(LOOKBACK, n - HORIZON + 1):
            if self.times[t] >= self.start and self.times[t + HORIZON - 1] <= self.end:
                self.index_map.append(t)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        t = self.index_map[idx]
        x = self.x[t - LOOKBACK:t].copy()
        ys = self.y_scaled[t:t + HORIZON].copy()
        yr = self.y_raw[t:t + HORIZON].copy()
        d = self.daylight[t:t + HORIZON].copy()
        return (
            torch.from_numpy(x),
            torch.from_numpy(ys),
            torch.from_numpy(yr),
            torch.from_numpy(d),
        )

def subset_by_target_containment(base, prepared_city: pd.DataFrame, feature_cols: Sequence[str],
                                 start: str, end: str):
    return TargetContainedDataset(base, prepared_city, feature_cols, start, end)

def prepare_target_city(base, valid_full: pd.DataFrame, target_col: str, city: str,
                        feature_cols: Sequence[str], continuous_cols: Sequence[str],
                        x_scaler, y_scaler) -> pd.DataFrame:
    df = valid_full[valid_full[base.CITY_COL].astype(str) == str(city)].copy()
    if len(df) != 17520:
        raise RuntimeError(f"{city}: expected 17,520 Validation rows, got {len(df)}")
    # Cold-start track: deliberately no target-history feature construction.
    built, built_cont = base.build_feature_columns(df, "basic", False)
    if list(built) != list(feature_cols):
        raise RuntimeError(f"{city}: feature order differs from frozen source fold")
    if list(built_cont) != list(continuous_cols):
        raise RuntimeError(f"{city}: continuous-feature order differs from frozen source fold")
    required = list(feature_cols) + [base.TARGET_VALUE_COL, base.RAW_TARGET_COL, base.DAYLIGHT_COL]
    before = len(df)
    df = df.dropna(subset=required).copy()
    if len(df) != before:
        raise RuntimeError(f"{city}: unexpected NaN drop in cold-start Validation frame: {before-len(df)}")
    if x_scaler is not None:
        df.loc[:, list(continuous_cols)] = x_scaler.transform(
            df[list(continuous_cols)].to_numpy(np.float32)
        )
    df.loc[:, base.TARGET_SCALED_COL] = y_scaler.transform(
        df[[base.TARGET_VALUE_COL]].to_numpy(np.float32)
    ).reshape(-1)
    if not np.isfinite(df[list(feature_cols)].to_numpy(np.float32)).all():
        raise RuntimeError(f"{city}: nonfinite predictor after frozen scaling")
    if not np.isfinite(df[base.TARGET_SCALED_COL].to_numpy(np.float32)).all():
        raise RuntimeError(f"{city}: nonfinite scaled target")
    return df.sort_values(base.TIME_COL).reset_index(drop=True)

def train_head_epoch(base, model, loader, optimizer, horizon_weights, device: str) -> float:
    set_adaptation_train_mode(model)
    loss_sum = 0.0
    count = 0
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    for batch in loader:
        x, y_scaled, y_raw, daylight = batch
        x = x.to(device)
        y_scaled = y_scaled.to(device)
        y_raw = y_raw.to(device)
        daylight = daylight.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred_scaled = model(x)
        loss = base.solar_weighted_loss(
            pred_scaled=pred_scaled,
            true_scaled=y_scaled,
            true_raw=y_raw,
            daylight=daylight,
            horizon_weights=horizon_weights,
            loss_name="huber",
            huber_beta=HUBER_BETA,
            daylight_weight=DAYLIGHT_WEIGHT,
            night_weight=NIGHT_WEIGHT,
            production_weight=PRODUCTION_WEIGHT,
            max_kw_per_kwp=MAX_KW_PER_KWP,
        )
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
        optimizer.step()
        bs = x.size(0)
        loss_sum += float(loss.item()) * bs
        count += bs
    return loss_sum / max(1, count)

@torch.no_grad()
def corrected_metrics(base, model, ds, y_scaler, correction: Dict[str, Any], device: str):
    loader = make_loader(ds, False)
    pred, true, day = base.collect_predictions(
        model=model,
        loader=loader,
        y_scaler=y_scaler,
        device=device,
        use_city_embedding=False,
        clip_per_kwp=True,
        max_kw_per_kwp=MAX_KW_PER_KWP,
        force_night_zero=True,
    )
    pred = base.apply_bias_correction(pred, day, correction)
    pred = base.final_physical_postprocess(
        pred, day, clip_per_kwp=True, max_kw_per_kwp=MAX_KW_PER_KWP, force_night_zero=True
    )
    metrics = base.compute_all_metrics(pred, true, day, REFERENCE_CAPACITY_KWP)
    return metrics, pred, true, day

def run_complete(run_dir: Path) -> bool:
    lock = run_dir / "run_lock.json"
    ckpt = run_dir / "adapted_checkpoint.pt"
    if not lock.is_file() or not ckpt.is_file():
        return False
    try:
        d = safe_json(lock)
        if d.get("run_complete") is not True or d.get("test_dataset_opened") is not False:
            return False
        if sha256(ckpt) != d.get("adapted_checkpoint_sha256"):
            return False
        return True
    except Exception:
        return False

# -----------------------------------------------------------------------------
# Prerequisite verification
# -----------------------------------------------------------------------------

def verify_prerequisites() -> Dict[str, Any]:
    failures = []
    for p, expected, label in (
        (BASE_CODE, EXPECTED_BASE_SHA, "base_code"),
        (STRICT_CODE, EXPECTED_STRICT_SHA, "strict_code"),
        (VALID_PATH, EXPECTED_VALID_SHA, "validation"),
        (FINAL_LOCK, EXPECTED_LOCK_SHA, "fewshot_final_lock"),
    ):
        if not p.is_file():
            failures.append(f"Missing {label}: {p}")
        elif sha256(p) != expected:
            failures.append(f"{label} SHA mismatch")
    if not A00B_REPORT.is_file():
        failures.append(f"Missing A00b PASS report: {A00B_REPORT}")
    else:
        a = safe_json(A00B_REPORT)
        for k, v in (
            ("status","PASS"),
            ("n_failures",0),
            ("test_dataset_opened",False),
            ("adaptation_training_authorized",True),
            ("validation_path_bound",True),
            ("head_parameter_binding_complete",True),
            ("budget_window_binding_complete",True),
        ):
            if a.get(k) != v:
                failures.append(f"A00b prerequisite mismatch {k}={a.get(k)!r}, expected {v!r}")
        if a.get("final_protocol_lock_sha256") != EXPECTED_LOCK_SHA:
            failures.append("A00b final protocol lock SHA mismatch")
    stage_lock = SOURCE_STAGE / "stage_lock.json"
    if not stage_lock.is_file():
        failures.append(f"Missing frozen H48 cold-start source stage lock: {stage_lock}")
    else:
        s = safe_json(stage_lock)
        if s.get("stage_complete") is not True or s.get("selection_used_test") is not False or s.get("test_data_loaded") is not False:
            failures.append("Invalid frozen source stage lock")
        if int(s.get("horizon",-1)) != 48 or s.get("track") != "cold_start":
            failures.append("Source stage is not H48 cold_start")
        if s.get("validation_sha256") != EXPECTED_VALID_SHA:
            failures.append("Source stage Validation SHA mismatch")
        if sorted(s.get("cities",[])) != sorted(CITIES):
            failures.append("Source stage city grid mismatch")
    if failures:
        raise RuntimeError("A20 prerequisite failure:\n- " + "\n- ".join(failures))
    return {
        "a00b_report_sha256": sha256(A00B_REPORT),
        "source_stage_lock_sha256": sha256(SOURCE_STAGE/"stage_lock.json"),
        "final_protocol_lock_sha256": sha256(FINAL_LOCK),
        "validation_sha256": sha256(VALID_PATH),
        "base_code_sha256": sha256(BASE_CODE),
        "strict_code_sha256": sha256(STRICT_CODE),
    }

# -----------------------------------------------------------------------------
# One adaptation run
# -----------------------------------------------------------------------------

def execute_run(base, valid_full, target_col: str, city: str, budget: str, seed: int,
                prereq: Dict[str, Any], device: str) -> Dict[str, Any]:
    run_dir = OUT / "folds" / city / budget / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if run_complete(run_dir):
        print(f"RESUME H48 {city} {budget} seed={seed}: complete")
        return safe_json(run_dir/"run_lock.json")["result_row"]

    source_fold = SOURCE_STAGE / "folds" / city
    source_fold_lock = source_fold / "fold_lock.json"
    fl = safe_json(source_fold_lock)
    if fl.get("selection_complete") is not True or fl.get("target_city") != city:
        raise RuntimeError(f"{city}: invalid source fold lock")
    if city in fl.get("source_cities", []):
        raise RuntimeError(f"{city}: held-out leakage in frozen source fold")
    if int(fl.get("horizon",-1)) != 48 or fl.get("track") != "cold_start":
        raise RuntimeError(f"{city}: wrong source fold protocol")
    if fl.get("validation_sha256") != EXPECTED_VALID_SHA:
        raise RuntimeError(f"{city}: source fold Validation SHA mismatch")

    feature_path = source_fold/"feature_cols.json"
    continuous_path = source_fold/"continuous_cols.json"
    x_path = source_fold/"x_scaler.pkl"
    y_path = source_fold/"y_scaler.pkl"
    source_seed_dir = source_fold/"L24"/f"seed_{seed}"
    source_ckpt = source_seed_dir/"best_checkpoint.pt"
    corr_path = source_seed_dir/"correction.json"

    # Revalidate all source artifacts against the original frozen fold lock.
    seed_art = {int(x["seed"]): x for x in fl["seed_artifacts"]}[seed]
    if sha256(source_ckpt) != seed_art["checkpoint_sha256"]:
        raise RuntimeError(f"{city} seed{seed}: source checkpoint changed")
    if sha256(corr_path) != seed_art["correction_sha256"]:
        raise RuntimeError(f"{city} seed{seed}: source correction changed")
    if sha256(x_path) != fl["x_scaler_sha256"]:
        raise RuntimeError(f"{city}: x_scaler changed")
    if sha256(y_path) != fl["y_scaler_sha256"]:
        raise RuntimeError(f"{city}: y_scaler changed")

    features = json.loads(feature_path.read_text(encoding="utf-8"))
    continuous = json.loads(continuous_path.read_text(encoding="utf-8"))
    if len(features) != 34 or len(continuous) != 20:
        raise RuntimeError(f"{city}: frozen feature/scaler scope mismatch")
    if any(c.startswith("solar_lag") or c.startswith("solar_roll") or c.startswith("solar_delta") or c=="solar_daylight_lag1" for c in features):
        raise RuntimeError(f"{city}: target-history predictor found in cold-start adaptation")

    x_scaler = joblib.load(x_path)
    y_scaler = joblib.load(y_path)
    if int(x_scaler.n_features_in_) != 20 or int(y_scaler.n_features_in_) != 1:
        raise RuntimeError(f"{city}: scaler dimensionality changed")

    correction = safe_json(corr_path)
    if correction.get("method") != "horizon_daylight_bias":
        raise RuntimeError(f"{city} seed{seed}: expected frozen horizon_daylight_bias correction")

    prepared = prepare_target_city(
        base, valid_full, target_col, city, features, continuous, x_scaler, y_scaler
    )

    bd = BUDGETS[budget]
    train_ds = subset_by_target_containment(base, prepared, features, *bd["selection_train"])
    val_ds = subset_by_target_containment(base, prepared, features, *bd["selection_validation"])
    full_ds = subset_by_target_containment(base, prepared, features, *bd["full_refit"])

    counts = {
        "selection_train": len(train_ds),
        "selection_validation": len(val_ds),
        "full_refit": len(full_ds),
    }
    if counts != bd["expected"]:
        raise RuntimeError(f"{city} {budget}: window counts {counts} != expected {bd['expected']}")

    # Strong containment assertion using actual dataset timestamps.
    def assert_containment(ds, interval, label):
        s, e = pd.Timestamp(interval[0]), pd.Timestamp(interval[1])
        for t in ds.index_map:
            if ds.times[t] < s or ds.times[t + HORIZON - 1] > e:
                raise RuntimeError(f"{city} {budget} {label}: target containment failure")
    assert_containment(train_ds, bd["selection_train"], "selection_train")
    assert_containment(val_ds, bd["selection_validation"], "selection_validation")
    assert_containment(full_ds, bd["full_refit"], "full_refit")

    # ------------------------------------------------------------------
    # Selection: source checkpoint -> head-only fine-tune on train portion,
    # monitor target-city adaptation-validation daylight RMSE after the
    # FROZEN source correction and standard postprocessing.
    # ------------------------------------------------------------------
    base.set_seed(seed)
    model = build_model(base, len(features), device)
    source_state = load_source_state(source_ckpt, device)
    model.load_state_dict(source_state)
    trainable, tnames, fnames = bind_head_only(model)

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    horizon_weights = base.build_horizon_weights(HORIZON, device)
    train_loader = make_loader(train_ds, True)

    history = []
    best_state = None
    best_epoch = 0
    best_monitor = float("inf")
    bad_epochs = 0
    start = time.time()

    # Diagnostic only, not eligible as best_epoch=0.
    zero_val_metrics, _, _, _ = corrected_metrics(base, model, val_ds, y_scaler, correction, device)

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_head_epoch(base, model, train_loader, optimizer, horizon_weights, device)
        val_metrics, _, _, _ = corrected_metrics(base, model, val_ds, y_scaler, correction, device)
        monitor = float(val_metrics["daylight_per_kwp_rmse"])
        improved = monitor < (best_monitor - MIN_DELTA)
        if improved:
            best_monitor = monitor
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": LR,
            "val_daylight_per_kwp_rmse": monitor,
            "val_per_kwp_rmse": val_metrics["per_kwp_rmse"],
            "val_per_kwp_mae": val_metrics["per_kwp_mae"],
            "val_daylight_per_kwp_mae": val_metrics["daylight_per_kwp_mae"],
            "improved": bool(improved),
            "bad_epochs": bad_epochs,
        })
        print(
            f"H48 {city} {budget} seed={seed} epoch={epoch:02d} "
            f"train={train_loss:.6f} val_day_RMSE={monitor:.6f}"
        )
        if bad_epochs >= EARLY_PATIENCE:
            break

    if best_state is None or best_epoch < 1:
        raise RuntimeError(f"{city} {budget} seed{seed}: selection failed to choose best epoch >=1")

    selection_ckpt = run_dir/"selection_best_checkpoint.pt"
    torch.save(best_state, selection_ckpt)
    pd.DataFrame(history).to_csv(run_dir/"selection_history.csv", index=False)

    # ------------------------------------------------------------------
    # Refit: restore ORIGINAL source checkpoint, reset same seed, and train
    # on the full labeled budget for exactly best_epoch. No validation is
    # used or evaluated during refit.
    # ------------------------------------------------------------------
    base.set_seed(seed)
    refit = build_model(base, len(features), device)
    refit.load_state_dict(source_state)
    refit_trainable, refit_tnames, refit_fnames = bind_head_only(refit)
    if refit_tnames != tnames or refit_fnames != fnames:
        raise RuntimeError("Selection/refit trainable binding differs")
    refit_optimizer = torch.optim.AdamW(refit_trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    full_loader = make_loader(full_ds, True)
    refit_history = []
    for epoch in range(1, best_epoch + 1):
        tr_loss = train_head_epoch(base, refit, full_loader, refit_optimizer, horizon_weights, device)
        refit_history.append({"epoch":epoch,"train_loss":tr_loss,"learning_rate":LR})
    pd.DataFrame(refit_history).to_csv(run_dir/"refit_history.csv", index=False)

    adapted_ckpt = run_dir/"adapted_checkpoint.pt"
    adapted_state = {k:v.detach().cpu().clone() for k,v in refit.state_dict().items()}
    torch.save(adapted_state, adapted_ckpt)

    # Prove the LSTM backbone is byte-identical to the source checkpoint.
    for k in source_state:
        if k.startswith("lstm."):
            if not torch.equal(source_state[k].detach().cpu(), adapted_state[k]):
                raise RuntimeError(f"{city} {budget} seed{seed}: frozen backbone changed: {k}")

    elapsed = time.time() - start
    result_row = {
        "horizon": HORIZON,
        "city": city,
        "budget": budget,
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_best_daylight_rmse": best_monitor,
        "zero_shot_source_daylight_rmse_on_adaptation_val": float(zero_val_metrics["daylight_per_kwp_rmse"]),
        "selection_train_windows": counts["selection_train"],
        "selection_validation_windows": counts["selection_validation"],
        "full_refit_windows": counts["full_refit"],
        "trainable_parameters": EXPECTED_HEAD_PARAMS,
        "selection_epochs_executed": len(history),
        "refit_epochs": best_epoch,
        "elapsed_seconds": elapsed,
    }

    lock = {
        "run_complete": True,
        "stage": "A20",
        "horizon": HORIZON,
        "city": city,
        "budget": budget,
        "seed": seed,
        "test_dataset_opened": False,
        "test_inference_performed": False,
        "adaptation_uses_target_history_predictors": False,
        "target_label_source": "2023 target-city normalized physics-based AC PV target; not metered plant production",
        "feature_count": len(features),
        "continuous_feature_count": len(continuous),
        "trainable_parameter_names": tnames,
        "frozen_parameter_names": fnames,
        "trainable_parameter_count": EXPECTED_HEAD_PARAMS,
        "backbone_mode_during_adaptation": "eval",
        "head_mode_during_adaptation": "train",
        "fixed_learning_rate": LR,
        "source_correction_method": correction["method"],
        "source_correction_frozen": True,
        "source_scalers_frozen": True,
        "selection_intervals": {
            "train": list(bd["selection_train"]),
            "validation": list(bd["selection_validation"]),
            "full_refit": list(bd["full_refit"]),
        },
        "window_counts": counts,
        "best_epoch": best_epoch,
        "source_fold_lock": str(source_fold_lock),
        "source_fold_lock_sha256": sha256(source_fold_lock),
        "source_checkpoint": str(source_ckpt),
        "source_checkpoint_sha256": sha256(source_ckpt),
        "source_checkpoint_state_digest": state_sha256({k:v.detach().cpu() for k,v in source_state.items()}),
        "source_correction": str(corr_path),
        "source_correction_sha256": sha256(corr_path),
        "x_scaler_sha256": sha256(x_path),
        "y_scaler_sha256": sha256(y_path),
        "selection_checkpoint_sha256": sha256(selection_ckpt),
        "adapted_checkpoint": str(adapted_ckpt),
        "adapted_checkpoint_sha256": sha256(adapted_ckpt),
        "adapted_checkpoint_state_digest": state_sha256(adapted_state),
        "frozen_backbone_identity_verified": True,
        "selection_history_sha256": sha256(run_dir/"selection_history.csv"),
        "refit_history_sha256": sha256(run_dir/"refit_history.csv"),
        **prereq,
        "result_row": result_row,
    }
    json_dump(run_dir/"run_lock.json", lock)
    return result_row


# -----------------------------------------------------------------------------
# A20 runtime smoke gate — no training, no Test
# -----------------------------------------------------------------------------

def run_smoke(base, valid_full, target_col: str, prereq: Dict[str, Any]) -> Dict[str, Any]:
    smoke_dir = W / "output" / "loco_fewshot" / "H48" / "A20_smoke"
    shutil.rmtree(smoke_dir, ignore_errors=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    rows = []

    for city in CITIES:
        source_fold = SOURCE_STAGE / "folds" / city
        fl = safe_json(source_fold / "fold_lock.json")
        features = json.loads((source_fold/"feature_cols.json").read_text(encoding="utf-8"))
        continuous = json.loads((source_fold/"continuous_cols.json").read_text(encoding="utf-8"))
        x_path = source_fold/"x_scaler.pkl"
        y_path = source_fold/"y_scaler.pkl"
        x_scaler = joblib.load(x_path)
        y_scaler = joblib.load(y_path)

        if len(features) != 34 or len(continuous) != 20:
            failures.append(f"{city}: feature/scaler scope mismatch")
            continue
        prepared = prepare_target_city(
            base, valid_full, target_col, city, features, continuous, x_scaler, y_scaler
        )

        for budget, bd in BUDGETS.items():
            ds_train = subset_by_target_containment(base, prepared, features, *bd["selection_train"])
            ds_val = subset_by_target_containment(base, prepared, features, *bd["selection_validation"])
            ds_full = subset_by_target_containment(base, prepared, features, *bd["full_refit"])
            got = {
                "selection_train":len(ds_train),
                "selection_validation":len(ds_val),
                "full_refit":len(ds_full),
            }
            rows.append({"city":city,"budget":budget,**got})
            if got != bd["expected"]:
                failures.append(f"{city} {budget}: {got} != {bd['expected']}")

        # Validate all three frozen source seed artifacts and model loading.
        seed_art = {int(x["seed"]): x for x in fl["seed_artifacts"]}
        for seed in SEEDS:
            sdir = source_fold/"L24"/f"seed_{seed}"
            cp = sdir/"best_checkpoint.pt"
            corr = sdir/"correction.json"
            if sha256(cp) != seed_art[seed]["checkpoint_sha256"]:
                failures.append(f"{city} seed{seed}: source checkpoint hash mismatch")
            if sha256(corr) != seed_art[seed]["correction_sha256"]:
                failures.append(f"{city} seed{seed}: correction hash mismatch")
            state = load_source_state(cp, "cpu")
            model = build_model(base, len(features), "cpu")
            model.load_state_dict(state)
            _, tnames, fnames = bind_head_only(model)
            if tuple(tnames) != TRAINABLE_KEYS or not all(n.startswith("lstm.") for n in fnames):
                failures.append(f"{city} seed{seed}: head/backbone binding mismatch")

        # One forward + weighted-loss calculation, no backward/optimizer/training.
        probe = subset_by_target_containment(base, prepared, features, *BUDGETS["30d"]["selection_train"])
        batch = next(iter(DataLoader(probe, batch_size=min(8, len(probe)), shuffle=False)))
        x, ys, yr, day = batch
        model.eval()
        with torch.no_grad():
            pred = model(x)
            hw = base.build_horizon_weights(HORIZON, "cpu")
            loss = base.solar_weighted_loss(
                pred_scaled=pred, true_scaled=ys, true_raw=yr, daylight=day,
                horizon_weights=hw, loss_name="huber", huber_beta=HUBER_BETA,
                daylight_weight=DAYLIGHT_WEIGHT, night_weight=NIGHT_WEIGHT,
                production_weight=PRODUCTION_WEIGHT, max_kw_per_kwp=MAX_KW_PER_KWP,
            )
        if tuple(pred.shape) != (x.shape[0], HORIZON) or not torch.isfinite(loss):
            failures.append(f"{city}: forward/loss smoke failure")

    pd.DataFrame(rows).to_csv(smoke_dir/"A20_H48_smoke_window_counts.csv", index=False)
    report = {
        "status":"PASS" if not failures else "FAIL",
        "stage":"A20-smoke",
        "training_performed":False,
        "optimizer_created":False,
        "backward_called":False,
        "test_dataset_opened":False,
        "test_inference_performed":False,
        "cities_checked":15,
        "budgets_checked":["30d","90d","365d"],
        "source_seed_checkpoints_checked":45,
        "forward_loss_probes":15,
        "n_failures":len(failures),
        "failures":failures,
        "prerequisites":prereq,
        "window_counts_csv_sha256":sha256(smoke_dir/"A20_H48_smoke_window_counts.csv"),
    }
    json_dump(smoke_dir/"A20_H48_smoke_report.json", report)
    print(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError(f"A20 smoke failed with {len(failures)} issue(s)")
    return report


# -----------------------------------------------------------------------------
# Main stage
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--smoke_only", action="store_true")
    args = parser.parse_args()

    prereq = verify_prerequisites()
    base = import_file("fewshot_base_h48", BASE_CODE)
    # Validation only. Test path is intentionally absent from the executable data flow.
    valid_full, target_col = base.load_split(str(VALID_PATH), "valid", "auto")
    valid_full = base.prepare_physical_target(valid_full, target_col)
    if len(valid_full) != 262800:
        raise RuntimeError(f"Validation row count changed: {len(valid_full)}")
    if sorted(valid_full[base.CITY_COL].astype(str).unique().tolist()) != sorted(CITIES):
        raise RuntimeError("Validation city set changed")

    if args.smoke_only:
        run_smoke(base, valid_full, target_col, prereq)
        print("A20 H48 RUNTIME SMOKE PASS | no training | Test file was not read.")
        return

    # Full training is allowed only after an independently reviewed smoke PASS.
    smoke_report = W / "output" / "loco_fewshot" / "H48" / "A20_smoke" / "A20_H48_smoke_report.json"
    if not smoke_report.is_file() or safe_json(smoke_report).get("status") != "PASS":
        raise RuntimeError("Missing A20 smoke PASS. Run 00_A20_H48_smoke.sh first.")

    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"A20 H48 Few-shot adaptation | device={device}")
    print("Test dataset is NOT read in this stage.")

    rows = []
    failures = []
    for city in CITIES:
        for budget in ("30d","90d","365d"):
            for seed in SEEDS:
                try:
                    rows.append(execute_run(
                        base, valid_full, target_col, city, budget, seed, prereq, device
                    ))
                except Exception as exc:
                    failures.append({
                        "city":city,"budget":budget,"seed":seed,
                        "error":repr(exc),
                    })
                    print(f"FAIL H48 {city} {budget} seed={seed}: {exc}", file=sys.stderr)
                    json_dump(OUT/"failures.json", failures)
                    # Stop immediately: do not continue a scientifically inconsistent stage.
                    raise

    df = pd.DataFrame(rows).sort_values(["city","budget","seed"]).reset_index(drop=True)
    if len(df) != 135:
        raise RuntimeError(f"Expected 135 completed H48 adaptation runs, got {len(df)}")
    df.to_csv(OUT/"A20_H48_all_runs.csv", index=False)

    # Descriptive selection-validation summaries only; these are not Test results.
    summary_rows = []
    for (city,budget), g in df.groupby(["city","budget"], sort=True):
        r = {"city":city,"budget":budget,"horizon":HORIZON,"n_seeds":len(g),"sd_ddof":1}
        for c in ("best_epoch","selection_best_daylight_rmse",
                  "zero_shot_source_daylight_rmse_on_adaptation_val"):
            r[f"{c}_mean"] = float(g[c].mean())
            r[f"{c}_std"] = float(g[c].std(ddof=1))
        summary_rows.append(r)
    pd.DataFrame(summary_rows).to_csv(OUT/"A20_H48_selection_summary_ddof1.csv", index=False)

    # Revalidate every run lock and checkpoint before closing A20.
    manifest = []
    for city in CITIES:
        for budget in ("30d","90d","365d"):
            for seed in SEEDS:
                rd = OUT/"folds"/city/budget/f"seed_{seed}"
                lk = safe_json(rd/"run_lock.json")
                ck = rd/"adapted_checkpoint.pt"
                if lk.get("run_complete") is not True or lk.get("test_dataset_opened") is not False:
                    raise RuntimeError(f"Invalid run lock: {rd}")
                if sha256(ck) != lk["adapted_checkpoint_sha256"]:
                    raise RuntimeError(f"Adapted checkpoint changed: {rd}")
                if lk.get("frozen_backbone_identity_verified") is not True:
                    raise RuntimeError(f"Backbone identity not verified: {rd}")
                manifest.append({
                    "city":city,"budget":budget,"seed":seed,
                    "run_lock":str(rd/"run_lock.json"),
                    "run_lock_sha256":sha256(rd/"run_lock.json"),
                    "adapted_checkpoint":str(ck),
                    "adapted_checkpoint_sha256":sha256(ck),
                    "adapted_checkpoint_state_digest":lk["adapted_checkpoint_state_digest"],
                    "source_checkpoint_sha256":lk["source_checkpoint_sha256"],
                    "source_correction_sha256":lk["source_correction_sha256"],
                })
    pd.DataFrame(manifest).to_csv(OUT/"A20_H48_checkpoint_manifest.csv", index=False)

    stage_lock = {
        "status":"PASS",
        "stage":"A20",
        "stage_complete":True,
        "horizon":HORIZON,
        "n_cities":15,
        "budgets":["30d","90d","365d"],
        "seeds":[1,2,3],
        "selection_runs":135,
        "adapted_checkpoints":135,
        "lookback":LOOKBACK,
        "feature_count":34,
        "target_history_predictors":0,
        "trainable_head_parameters":EXPECTED_HEAD_PARAMS,
        "frozen_backbone":True,
        "source_scalers_frozen":True,
        "source_correction_frozen":True,
        "test_dataset_opened":False,
        "test_inference_performed":False,
        "sample_sd_ddof":1,
        "prerequisites":prereq,
        "all_runs_csv_sha256":sha256(OUT/"A20_H48_all_runs.csv"),
        "selection_summary_sha256":sha256(OUT/"A20_H48_selection_summary_ddof1.csv"),
        "checkpoint_manifest_sha256":sha256(OUT/"A20_H48_checkpoint_manifest.csv"),
    }
    json_dump(OUT/"A20_H48_stage_lock.json", stage_lock)
    print(json.dumps(stage_lock, indent=2))
    print("A20 H48 FEW-SHOT SELECTION+REFIT PASS | Test file was not read.")

if __name__ == "__main__":
    main()
