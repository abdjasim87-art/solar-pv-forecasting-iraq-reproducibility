LOCO Few-shot — A70 H48 / 30-day Adapted Test v1
=================================================

Prerequisites:
- A30 adaptation freeze PASS/CLOSED.
- H24 adapted Test stages A40/A50/A60 completed and A60 PASS/CLOSED.
- No training, calibration, correction fitting, scaler fitting, or model selection occurs.

A70 evaluates:
  15 cities × 3 seeds = 45 frozen H48/30d adapted checkpoints.

Before Test is read, A70 verifies:
- frozen A30 protocol/report/manifest/registry,
- exact SHA-256 of all 45 H48/30d adapted checkpoints,
- exact closed A60 stage-lock SHA-256.

Test policy is paired to the already-closed Stage-80 H48 cold-start Test:
- exact frozen 2024–2025 test.csv SHA-256,
- trim first 168 rows/city,
- aligned min_start=168 with L24 and H48,
- 17,161 windows/city,
- 257,415 pooled windows/seed,
- 12,355,920 residuals/seed,
- 6,199,242 daylight residuals/seed,
- 258,120 unique city-hours/seed,
- true energy 171,314.764839 MWh.

During Test, exact start-index / series-time / true-target / daylight identity is
checked against Stage 80. Raw 45 NPZ outputs are saved locally and independently
re-audited. The upload archive excludes heavy NPZ bytes but includes their hashes
and every non-NPZ evidence file.

Run:
  cd /workspace/lstm
  tar -xzf loco_fewshot_A70_H48_30d_test_v1.tar.gz
  chmod +x loco_fewshot_A70_H48_30d_test_v1/70_H48_30d_test.sh
  ./loco_fewshot_A70_H48_30d_test_v1/70_H48_30d_test.sh

Upload:
  /workspace/lstm/loco_fewshot_A70_H48_30d_test.log
  /workspace/lstm/loco_fewshot_A70_H48_30d_audit.tar.gz

Stop after A70. Do not run A80 until A70 is reviewed.
