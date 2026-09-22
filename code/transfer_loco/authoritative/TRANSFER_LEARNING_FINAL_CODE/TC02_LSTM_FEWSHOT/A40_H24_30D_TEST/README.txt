LOCO Few-shot — A40 H24 / 30-day Adapted Test v1
=================================================

Prerequisite: A30 PASS/CLOSED.

This is the FIRST adapted Test stage. Selection/refit is already frozen.
No training, calibration, correction fitting, scaler fitting, or model selection occurs.

A40 evaluates:
  15 cities × 3 seeds = 45 frozen H24/30d adapted checkpoints

Test policy:
- exact frozen 2024–2025 test.csv SHA-256
- same H24 cold-start aligned grid as Stage 60
- trim first 168 rows/city, then min_start=168 with L24
- 17,185 windows/city
- 257,775 pooled windows/seed
- 6,186,600 residuals/seed
- 3,103,207 daylight residuals/seed
- 258,120 unique city-hours/seed
- true energy 171,314.764839 MWh

Before test.csv is read, A40 rehashes the A30 report/registry and all 45 H24/30d
adapted checkpoints. During Test it verifies exact grid, true target, and daylight
identity against the already-closed Stage-60 zero-shot cold-start raw outputs.

Run:
  cd /workspace/lstm
  tar -xzf loco_fewshot_A40_H24_30d_test_v1.tar.gz
  chmod +x loco_fewshot_A40_H24_30d_test_v1/40_H24_30d_test.sh
  ./loco_fewshot_A40_H24_30d_test_v1/40_H24_30d_test.sh

Upload:
  /workspace/lstm/loco_fewshot_A40_H24_30d_test.log
  /workspace/lstm/loco_fewshot_A40_H24_30d_audit.tar.gz

Stop after A40. Do not run A50 until A40 is reviewed.
