LOCO Few-shot Adapted vs Cold-start Zero-shot — Paired Statistics v1
=====================================================================

Purpose
-------
This package does not train, tune, or run model inference. It first freezes the already
completed A40-A90 adapted Test outputs together with the previously frozen cold-start
zero-shot baseline and the exact B10 analysis code. Only after B00 is reviewed should
B10 execute the pre-specified inferential analysis.

Scientific comparison
---------------------
- Comparison: adapted few-shot vs cold-start zero-shot.
- H24 and H48 are analyzed separately.
- Budgets: 30d, 90d, 365d.
- Pairing: city × forecast origin × horizon step.
- Loss is computed per seed and then averaged across seeds at each paired residual;
  predictions are NOT averaged into an ensemble.
- Daylight is primary; all-hours is supplemental.
- RMSE uses paired squared loss; MAE uses paired absolute loss.
- Circular moving-block bootstrap: 5000 replicates, 7 UTC-origin-day blocks.
- DM-style daily loss differential: Bartlett/Newey-West HAC lag 7 + HLN correction.
- Holm multiplicity: pooled family = 3 budgets; city family = 45 city-budget tests,
  within each horizon × daytype × metric.
- Strong support requires BOTH 95% bootstrap CI excluding 0 and Holm-adjusted p < 0.05.
- Effect = adapted - cold-start zero-shot. Negative favors adaptation.
- No inferential comparison among 30d/90d/365d is performed because budget size and
  seasonal coverage are linked in the frozen recent-history design.

Run order
---------
1) Extract package in /workspace/lstm.
2) Run ONLY B00:
     ./loco_fewshot_paired_stats_v1/00_B00_freeze.sh
3) Upload for review:
     /workspace/lstm/loco_fewshot_stats_B00_freeze.log
     /workspace/lstm/loco_fewshot_stats_B00_freeze_audit.tar.gz
4) Do NOT run 10_B10_paired_stats.sh until B00 is reviewed and explicitly CLOSED.

After B00 authorization
-----------------------
     ./loco_fewshot_paired_stats_v1/10_B10_paired_stats.sh

Then upload:
     /workspace/lstm/loco_fewshot_stats_B10_paired.log
     /workspace/lstm/loco_fewshot_stats_B10_paired_audit.tar.gz
