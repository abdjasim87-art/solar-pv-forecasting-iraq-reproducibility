LOCO Few-shot — A60 H24 / 365-day Adapted Test v1
==================================================

Prerequisites:
- A30 PASS/CLOSED.
- A40 H24/30d PASS/CLOSED.
- A50 H24/90d PASS/CLOSED.
- No training, calibration, correction fitting, scaler fitting, or model selection occurs.

A60 evaluates:
  15 cities × 3 seeds = 45 frozen H24/365d adapted checkpoints.

Before Test is read, A60 verifies:
- frozen A30 protocol/report/manifest/registry,
- exact SHA-256 of all 45 H24/365d adapted checkpoints,
- exact closed A40 stage-lock SHA-256,
- exact closed A50 stage-lock SHA-256.

Test policy is identical to A40/A50 and to the closed Stage-60 H24 cold-start grid:
- exact frozen 2024–2025 test.csv SHA-256,
- trim first 168 rows/city,
- aligned min_start=168 with L24,
- 17,185 windows/city,
- 257,775 pooled windows/seed,
- 6,186,600 residuals/seed,
- 3,103,207 daylight residuals/seed,
- 258,120 unique city-hours/seed,
- true energy 171,314.764839 MWh.

Exact grid / truth / daylight identity is checked against Stage 60.
Raw 45 NPZ outputs are saved locally and independently re-audited; the upload archive
contains hashes and all non-NPZ evidence while excluding the heavy NPZ bytes.

Run:
  cd /workspace/lstm
  tar -xzf loco_fewshot_A60_H24_365d_test_v1.tar.gz
  chmod +x loco_fewshot_A60_H24_365d_test_v1/60_H24_365d_test.sh
  ./loco_fewshot_A60_H24_365d_test_v1/60_H24_365d_test.sh

Upload:
  /workspace/lstm/loco_fewshot_A60_H24_365d_test.log
  /workspace/lstm/loco_fewshot_A60_H24_365d_audit.tar.gz

Stop after A60. Do not run H48 adapted Test until A60 is reviewed.
