LOCO Few-shot A00b Binding v1.3
=================================

AUDIT-IMPLEMENTATION FIX ONLY. The frozen scientific protocol
FEWSHOT_PROTOCOL_LOCK_FINAL_v1_1.json is unchanged.

Why v1.3 exists
---------------
v1.2 correctly loaded the scaler artifacts with joblib, but the audit incorrectly
expected x_scaler.n_features_in_ == 34. In the authoritative LSTM implementation,
StandardScaler is fitted only to continuous_cols. The cold-start model has 34 total
predictors = 20 continuous scaled predictors + 14 cyclic/binary/no-scale predictors.

Therefore the correct scaler expectations are:
  x_scaler.n_features_in_ = 20
  y_scaler.n_features_in_ = 1

No training and no Test read are performed by this stage.

Run:
  cd /workspace/lstm
  tar -xzf loco_fewshot_A00_binding_v1_3.tar.gz
  chmod +x loco_fewshot_A00_binding_v1_3/00b_bind_and_freeze.sh
  ./loco_fewshot_A00_binding_v1_3/00b_bind_and_freeze.sh

Upload:
  /workspace/lstm/loco_fewshot_A00b_binding.log
  /workspace/lstm/loco_fewshot_A00b_binding_audit.tar.gz

Do not start A10 until v1.3 is reviewed and closed.
