TRANSFER LEARNING FINAL CODE ARCHIVE
Created UTC: 2026-09-22T09:38:56.692533+00:00

PURPOSE
This folder separates the code used for the final solar transfer-learning experiments
from older, failed, duplicate, or not-yet-proven versions. Source files under
/workspace/lstm are NOT deleted or modified.

STRUCTURE
TC01_LSTM_LOCO_ZERO_SHOT
  Final LSTM LOCO history-aware/cold-start protocol, zero-shot statistics, and final raw audits.

TC02_LSTM_FEWSHOT
  Final few-shot preflight, H24/H48 selection/refit, freeze, six Test stages, and final paired statistics.

TC03_PATCHTST_LOCO_HISTORY_ZERO_SHOT
  Final PatchTST LOCO history-aware zero-shot package plus the selection freeze/pre-Test verification gate.

TC04_PATCHTST_VS_LSTM_LOCO_FINAL_STATS
  Final B00/B10 paired transfer-system comparison code.

TC05_SHARED_REFERENCE_CODE
  Shared/reference dependencies found on the pod.

TRACEABILITY_NOT_FINAL
  Older/failed/superseded candidates retained only so provenance is not lost.

REVIEW_REQUIRED
  Files that are preserved but deliberately NOT labeled FINAL_USED because the available
  execution evidence does not identify the accepted version with certainty.

IMPORTANT BINDING NOTE
The old root fewshot_A00b_binding.py is not final: the retained log shows that it failed.
The three directories loco_fewshot_A00_binding_v1_1/v1_2/v1_3 are therefore kept under
REVIEW_REQUIRED rather than silently guessing which one produced the accepted binding.

CONFIRMED VERSION NOTE
The accepted H24 few-shot execution references loco_fewshot_A10_H24_v1_1.
The accepted H48 few-shot execution references loco_fewshot_A20_H48_v1.

STATUS COUNTS
  DUPLICATE_OF_FINAL: 1
  FAILED_OLD_VERSION: 1
  FINAL_USED: 116
  REFERENCE_DEPENDENCY: 3
  REVIEW_REQUIRED_NOT_MARKED_FINAL: 15
  SUPERSEDED_CANDIDATE: 7
