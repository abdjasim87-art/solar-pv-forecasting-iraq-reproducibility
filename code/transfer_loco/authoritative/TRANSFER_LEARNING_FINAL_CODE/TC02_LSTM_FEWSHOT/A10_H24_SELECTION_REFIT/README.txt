LOCO Few-shot Adaptation — A10 H24 v1.1
=======================================

v1.1 is an implementation-only repair after v1 failed BEFORE training while
dynamically importing the strict reference module. The frozen scientific protocol
is unchanged.

Important:
- v1 produced zero adaptation files/checkpoints and did not open Test.
- v1.1 removes the unnecessary runtime dependency on StrictSolarDataset.
- Window construction is self-contained and explicitly enforces the already-frozen
  full target-horizon containment rule.
- The strict reference file is still SHA-verified as a provenance prerequisite.
- Dynamic imports are also registered in sys.modules before exec_module.

Run ONLY the smoke gate first:
  cd /workspace/lstm
  tar -xzf loco_fewshot_A10_H24_v1_1.tar.gz
  chmod +x loco_fewshot_A10_H24_v1_1/*.sh
  ./loco_fewshot_A10_H24_v1_1/00_A10_H24_smoke.sh

Upload:
  /workspace/lstm/loco_fewshot_A10_H24_smoke.log
  /workspace/lstm/loco_fewshot_A10_H24_smoke_audit.tar.gz

Do NOT run 10_H24_selection_refit.sh until the smoke gate is reviewed.

The smoke gate performs no optimizer step, no backward pass, no training, and no Test
read. It checks all 15 cities, all three budget window counts, all 45 H24 source
checkpoints/corrections, model/head binding, and one forward/loss probe per city.
