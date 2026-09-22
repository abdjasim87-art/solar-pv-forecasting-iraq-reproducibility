PATCHTST–LSTM LOCO PAIRED STATS v1

Run order:
1. ./00_B00_freeze.sh
2. STOP and upload B00 log + audit.
3. After review only: ./05_authorize_B10.sh
4. ./10_B10_paired_stats.sh
5. Upload B10 log + audit.

B00 reads only completed prediction NPZ files and package code. It does not train, infer, or open test.csv.
B10 re-hashes every frozen file before statistics.

Frozen model histories differ by design: PatchTST L168; LSTM L24. Pairing is on forecast origin/lead, not lookback.
