PATCHTST LOCO FREEZE GATE v1

Purpose:
Freeze all H24/H48 Selection artifacts before first access to test.csv, then require re-hashing
before each Test stage.

Order:
1) ./25_selection_freeze.sh
2) STOP and upload freeze log + audit for review.
3) Only after authorization: ./30_H24_test_frozen.sh
4) Review H24 Test.
5) Only after authorization: ./40_H48_test_frozen.sh

Do NOT run the original 30_H24_test.sh or 40_H48_test.sh directly after adopting this gate.
