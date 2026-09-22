LOCO Few-shot — A80 H48 / 90-day Adapted Test v1
=================================================

Prerequisites:
- A30 adaptation freeze PASS/CLOSED.
- A70 H48/30d PASS/CLOSED.
- No training, calibration, correction fitting, scaler fitting, or model selection occurs.

A80 evaluates:
  15 cities × 3 seeds = 45 frozen H48/90d adapted checkpoints.

Before Test is read, A80 verifies:
- frozen A30 protocol/report/manifest/registry,
- exact SHA-256 of all 45 H48/90d adapted checkpoints,
- exact closed A70 stage-lock SHA-256.

Test policy remains exactly paired to closed Stage-80 H48 cold-start:
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
re-audited. The upload archive excludes heavy NPZ bytes but includes hashes and
all non-NPZ evidence files.

Run:
  cd /workspace/lstm
  tar -xzf loco_fewshot_A80_H48_90d_test_v1.tar.gz
  chmod +x loco_fewshot_A80_H48_90d_test_v1/80_H48_90d_test.sh
  ./loco_fewshot_A80_H48_90d_test_v1/80_H48_90d_test.sh

Upload:
  /workspace/lstm/loco_fewshot_A80_H48_90d_test.log
  /workspace/lstm/loco_fewshot_A80_H48_90d_audit.tar.gz

Stop after A80. Do not run A90 until A80 is reviewed.
