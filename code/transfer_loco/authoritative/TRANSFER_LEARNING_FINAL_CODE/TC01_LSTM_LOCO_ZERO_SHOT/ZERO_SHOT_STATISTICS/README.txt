LOCO Zero-shot Freeze + Paired Statistics v1
=============================================

Purpose
-------
This package does NOT train or tune any model. It freezes the already completed
zero-shot LOCO Test outputs (Stages 50/60/70/80) and then performs a pre-specified,
paired comparison of History-aware versus Cold-start.

Scientific rules
----------------
- H24 and H48 are analyzed separately.
- Pairing key: city × forecast origin × horizon step.
- History-aware and Cold-start must have identical Test grids and identical truth/daylight masks.
- For deep models, loss is computed per seed and then averaged across seeds at each paired residual.
  Predictions are NOT averaged first.
- Daylight is primary; all-hours is supplemental.
- RMSE comparison uses squared loss; MAE uses absolute loss.
- 5000-replicate circular moving-block bootstrap, 7 UTC-origin-day blocks, seed 20260806.
- DM-style daily loss differential with Bartlett/Newey-West HAC lag 7 and HLN correction.
- Holm adjustment across 15 cities within each horizon × daytype × metric.
- Strong support requires BOTH bootstrap 95% CI excluding 0 and Holm-adjusted p < 0.05.
- Effect sign is History-aware minus Cold-start. Negative favors History-aware.
- No adaptation/few-shot experiment is executed by this package.

Run order
---------
1) Run ONLY:
   ./00_freeze_zero_shot.sh

2) Upload:
   /workspace/lstm/loco_zero_shot_freeze_v1.log
   /workspace/lstm/loco_zero_shot_freeze_v1_audit.tar.gz

3) Do NOT run 10_paired_stats.sh until the freeze audit is reviewed.

After authorization:
   ./10_paired_stats.sh

Expected final statistics artifacts:
   /workspace/lstm/loco_zero_shot_paired_stats_v1.log
   /workspace/lstm/loco_zero_shot_paired_stats_v1_audit.tar.gz
