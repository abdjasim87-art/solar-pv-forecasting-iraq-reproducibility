LOCO Few-shot A00b Binding v1.1
===============================

This stage performs NO training and does NOT open/read/hash test.csv.

It resolves the one warning from A00 by binding:
- /workspace/lstm/data2/valid.csv to the frozen Validation SHA-256,
- the exact 34 cold-start predictors,
- all six trainable forecast-head parameters,
- all frozen LSTM parameters,
- source x/y scalers and source correction files,
- exact per-city 30/90/365-day H24/H48 selection/refit window counts.

It also finalizes the pre-Test protocol before adaptation training. Two structural
changes are deliberately made before training: the frozen LSTM is held in eval mode,
and the LR scheduler is removed so selection/refit both use a fixed LR=1e-4.

Run only:
  cd /workspace/lstm
  tar -xzf loco_fewshot_A00_binding_v1_1.tar.gz
  chmod +x loco_fewshot_A00_binding_v1_1/00b_bind_and_freeze.sh
  ./loco_fewshot_A00_binding_v1_1/00b_bind_and_freeze.sh

Upload:
  /workspace/lstm/loco_fewshot_A00b_binding.log
  /workspace/lstm/loco_fewshot_A00b_binding_audit.tar.gz

Do not run adaptation training until A00b is reviewed and closed.
