LOCO Few-shot Adaptation — A30 Freeze v1
========================================

Prerequisites:
- A10 H24 selection+refit PASS/CLOSED.
- A20 H48 selection+refit PASS/CLOSED.
- No adapted Test inference has been run.

A30 performs NO training and never reads test.csv.

It freezes every file in the completed H24 and H48 selection/refit stages:
- 270 adapted checkpoints
- 270 selection checkpoints
- 270 run locks
- 270 selection histories
- 270 refit histories
- 8 top-level stage summary/lock/manifest files
Total expected frozen files = 1,358.

It also verifies the exact 270 city×horizon×budget×seed run grid, all run-lock
Test flags, frozen-backbone identity flags, and all checkpoint/history hashes.

Run:
  cd /workspace/lstm
  tar -xzf loco_fewshot_A30_freeze_v1.tar.gz
  chmod +x loco_fewshot_A30_freeze_v1/30_freeze_all_adapted.sh
  ./loco_fewshot_A30_freeze_v1/30_freeze_all_adapted.sh

Upload:
  /workspace/lstm/loco_fewshot_A30_freeze.log
  /workspace/lstm/loco_fewshot_A30_freeze_audit.tar.gz

Do not run adapted Test inference until A30 is independently reviewed and CLOSED.
