LOCO Few-shot Adaptation — A20 H48 v1
=====================================

Prerequisite:
  A10 H24 must be independently audited PASS/CLOSED.

A20 uses the same frozen few-shot protocol and runs:
  15 target cities × 3 recent-label budgets × 3 seeds = 135 H48 selection+refit runs.

No Test inference is permitted in A20.

Frozen H48 settings:
- Start from the matching frozen H48 cold-start source checkpoint.
- Lookback = 24 h; horizon = 48 h.
- 34 cold-start predictors; zero PV-target-history predictors.
- Source x/y scalers frozen.
- Source horizon_daylight_bias correction frozen.
- LSTM backbone frozen + eval mode.
- Forecast head only trainable = 44,272 parameters.
- AdamW, fixed lr=1e-4, weight_decay=1e-4, batch=512, grad_clip=1.
- Weighted Huber with daylight/night/production weights 4 / 0.25 / 1.
- Selection: max 30 epochs, patience 5, min_delta 1e-5.
- Monitor: target-city adaptation-validation daylight RMSE after frozen correction
  and unchanged physical postprocessing.
- Refit: restore ORIGINAL H48 source checkpoint and train on the full budget for
  exactly best_epoch; target validation is not consulted during refit.

H48 full-target-containment window counts per city:
  30d : train 529  | validation 97   | full refit 673
  90d : train 1681 | validation 385  | full refit 2113
  365d: train 6961 | validation 1705 | full refit 8713

Run the smoke gate FIRST:
  cd /workspace/lstm
  tar -xzf loco_fewshot_A20_H48_v1.tar.gz
  chmod +x loco_fewshot_A20_H48_v1/*.sh
  ./loco_fewshot_A20_H48_v1/00_A20_H48_smoke.sh

Upload for review:
  /workspace/lstm/loco_fewshot_A20_H48_smoke.log
  /workspace/lstm/loco_fewshot_A20_H48_smoke_audit.tar.gz

Do NOT run 20_H48_selection_refit.sh until the A20 smoke gate is reviewed.
After A20 full training is later audited, A30 will freeze all 270 adapted checkpoints
before any adapted Test inference.
