LOCO Few-shot A00b Binding v1.2
=================================

This is an AUDIT-IMPLEMENTATION FIX only. It does not alter the frozen scientific
protocol in FEWSHOT_PROTOCOL_LOCK_FINAL_v1_1.json.

Why v1.2 exists
---------------
v1.1 correctly completed the Validation-grid, budget-window, feature, and forecast-head
binding checks, but then crashed while reading StandardScaler artifacts with pickle.load().
The source pipeline stores these scalers with joblib.dump(), so v1.2 uses joblib.load().

No training was performed by v1.1 and no Test dataset was opened.

Run:
  cd /workspace/lstm
  tar -xzf loco_fewshot_A00_binding_v1_2.tar.gz
  chmod +x loco_fewshot_A00_binding_v1_2/00b_bind_and_freeze.sh
  ./loco_fewshot_A00_binding_v1_2/00b_bind_and_freeze.sh

Upload:
  /workspace/lstm/loco_fewshot_A00b_binding.log
  /workspace/lstm/loco_fewshot_A00b_binding_audit.tar.gz

Do not start A10 until this rerun is reviewed and closed.
