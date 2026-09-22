# PATCHTST–LSTM LOCO PAIRED STATISTICS PROTOCOL v1

## Question
Under the same held-out-city LOCO transfer setting, does the completed history-aware PatchTST
forecasting system differ from the completed history-aware LSTM forecasting system?

This is an end-to-end forecasting-system comparison. The two systems do not use identical internal
input representations, so the result must not be described as a pure architecture-only causal effect.

## Frozen inputs
- PatchTST LOCO history-aware zero-shot:
  `/workspace/lstm/output/patchtst_loco_v1/H{24,48}/test/raw_predictions/`
- LSTM LOCO history-aware zero-shot:
  `/workspace/lstm/output/loco_transfer_v2/H{24,48}/history_aware/test/cities/`
- 15 held-out target cities.
- Seeds 1, 2, and 3.
- H24 and H48.
- PatchTST uses its frozen L168 transfer configuration; LSTM uses its frozen L24 transfer configuration.
- Independent Test period remains 2024–2025; this statistics package never opens `test.csv`.

## Pairing
Exact pairing is required by:
`city × forecast_origin × horizon_step`.

For every city, horizon, and seed, B00 requires exact equality of:
- truth arrays,
- daylight masks,
- window start indices,
- full series timestamp arrays,
- derived forecast-origin timestamps,
- valid timestamps.

Any mismatch stops the analysis.

## Loss aggregation
Predictions are NOT averaged into an ensemble.

For each method and seed:
- RMSE inference uses squared error.
- MAE inference uses absolute error.

At each paired residual, losses are averaged across seeds 1–3. This preserves the previously adopted
multi-seed loss-aggregation rule.

## Primary and supplemental outcomes
- Primary: daylight RMSE and daylight MAE.
- Supplemental: all-hours RMSE and all-hours MAE.
- Per-lead results are descriptive only.

Effect convention:
`delta = PatchTST - LSTM`.
Negative delta favors PatchTST.
Positive percentage improvement means PatchTST has lower error than LSTM.

## Dependence-aware inference
- Paired circular moving-block bootstrap: 5,000 resamples.
- Block length: 7 UTC forecast-origin days.
- Fixed bootstrap base seed: 20260806.
- DM loss-differential test on daily loss differences.
- Bartlett/Newey-West HAC lag: 7 days.
- Harvey-Leybourne-Newbold finite-sample correction with h_days=1 for H24 and 2 for H48.

## Multiplicity
- Pooled tests are pre-specified singleton families within each horizon × daytype × metric.
- City-level tests use Holm correction across the 15 cities within each horizon × daytype × metric.

Strong statistical support requires BOTH:
1. paired 95% bootstrap CI excludes zero; and
2. Holm-adjusted DM-HAC p < 0.05.

## Interpretation constraint
A PatchTST–LSTM difference is a comparison of the two complete transfer forecasting systems.
Because their internal predictor representations differ, do not claim that the result isolates
architecture alone.
