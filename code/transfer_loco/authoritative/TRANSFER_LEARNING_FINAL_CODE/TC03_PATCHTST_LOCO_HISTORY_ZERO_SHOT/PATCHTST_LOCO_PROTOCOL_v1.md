# PatchTST LOCO history-aware zero-shot protocol v1.0

## Status
FROZEN BEFORE PATCHTST LOCO TRAINING OR TEST.

## Scientific objective
Evaluate whether the final audited official-architecture PatchTST forecasting system can transfer to an Iraqi location that is completely absent from model fitting.

This is a **history-aware zero-shot location-transfer** experiment. The held-out target location contributes no rows to source Training, source Validation, target scaling, checkpoint selection, early stopping, or Validation-only bias-correction fitting. At forecast time, however, PatchTST receives strictly past normalized PV-target history from the held-out location because the final audited PatchTST system is supervised S-mode with one past-target input channel.

Therefore this experiment must **not** be described as pure cold-start transfer.

## Relationship to the final audited PatchTST study
The experiment inherits the final audited PatchTST implementation rather than creating a new Transformer:

- Official repository: `yuqinie98/PatchTST`
- Pinned commit: `204c21efe0b39603ad6e2ca640ef5896646ab1a9`
- Official supervised PatchTST architecture imported unchanged.
- Input mode: S, one channel = strictly past normalized AC PV target.
- Patch length 16, stride 8, end padding.
- RevIN on, affine off, subtract-last off.
- `d_model=128`, 16 heads, 3 encoder layers, `d_ff=256`.
- dropout 0.20, fc_dropout 0.20, head_dropout 0.
- Adam, learning rate 1e-4, MSE, batch 128.
- maximum 100 epochs, patience 20, official type-3 LR rule.
- Validation-only correction candidates: none, global bias, daylight bias, horizon bias, horizon-by-daylight bias.
- Physical post-processing: nighttime zero, nonnegative, cap 1.2 kW/kWp.

The byte-identical audited PatchTST wrapper is preserved inside this package as `solar_patchtst_original_protocol_reference.py` and is imported by the LOCO wrapper for model construction, training, correction, metrics, and energy accounting.

## Fixed lookback
The LOCO experiment uses **L168 at both H24 and H48**. This is inherited from the uploaded audited final PatchTST Test artifacts, whose raw H24 and H48 Global predictions record `lookback=168`. No LOCO lookback search is permitted.

This reduces the final experimental matrix to:

15 held-out locations × 2 horizons × 3 seeds = **90 source-training runs**.

## Data and chronology
- Same 15 representative Iraqi locations.
- Same normalized physics-based AC PV target, kW/kWp.
- Training: 2015–2021.
- Validation/calibration: 2022–2023.
- Independent Test: 2024–2025.
- Horizons: H24 and H48.
- Seeds: 1, 2, 3; all retained.
- Sample SD: `ddof=1`.

The 53-feature strict LSTM preprocessing is rebuilt only to preserve the same split-local 168-hour history loss and common forecast-origin grid. Those 53 predictors are **not** passed to PatchTST.

## Leave-one-location-out source fitting
For target location c:

- Source Train = 14 other locations from 2015–2021.
- Source Validation = the same 14 locations from 2022–2023.
- Target location c is excluded before the source target scaler is fitted.
- A single source target StandardScaler is fitted from the 14-location source Training target only.
- Source Validation uses that frozen source scaler.
- All three seeds train on the same source fold.
- Early stopping uses source Validation MSE only.
- Post-prediction correction is chosen from source Validation only.

No target-location Training or Validation label is used anywhere in fitting.

## Target-location Test inference
Only after both H24 and H48 source-selection stage locks are complete and verified may `test.csv` be hashed and opened.

For the held-out target location:

- the frozen 14-location source scaler normalizes the target location's strictly past PV history;
- the exact frozen seed checkpoint and source-Validation correction are used;
- no target-location recalibration or fine-tuning is allowed;
- Test predictions are saved with rich pairing metadata.

## Evaluation alignment
The same split-local AR-history construction as the audited source removes the first 168 rows per location. Dataset evaluation then starts at `t=168` on the post-cleaning series. This reproduces the established common Test grid:

H24:
- 17,185 windows per target location;
- 257,775 pooled windows per seed;
- 6,186,600 residuals per seed;
- 3,103,207 daylight residuals per seed.

H48:
- 17,161 windows per target location;
- 257,415 pooled windows per seed;
- 12,355,920 residuals per seed;
- 6,199,242 daylight residuals per seed.

Expected true 15-location reference energy: 171.314764839 GWh.

## Raw prediction schema
Each target-city/seed NPZ stores:

- `pred`
- `true`
- `daylight`
- `city`
- `window_start_idx`
- `series_time_ns`
- `origin_ns`
- `valid_ns`
- `lookback`
- `horizon`
- `seed`

The `window_start_idx`, `series_time_ns`, `true`, and `daylight` fields are included specifically so the final PatchTST LOCO outputs can be checked for exact pairing against the already frozen LSTM history-aware LOCO outputs.

## Primary final comparison after both PatchTST Tests
The direct architecture-system transfer comparison will be:

**PatchTST LOCO history-aware zero-shot vs LSTM LOCO history-aware zero-shot**

on exactly paired city × forecast-origin × horizon-step rows.

This remains a forecasting-system comparison rather than a same-input architecture-only ablation because the LSTM history-aware system uses the 53-predictor representation while the audited PatchTST system uses only past target history.

A direct PatchTST-history-aware vs LSTM-cold-start ranking is not a primary comparison because their information policies differ.

## Prohibited changes and claims
- No LOCO lookback tuning.
- No target-city scaler fitting.
- No target-city Validation correction.
- No Test-based checkpoint, seed, or correction choice.
- No PatchTST few-shot extension after this experiment.
- Do not call this pure cold-start transfer.
- Do not claim a new PatchTST architecture.
- Do not describe the target as metered PV-plant production.
