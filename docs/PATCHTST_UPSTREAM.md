# Official PatchTST dependency

The retained PatchTST experiments use the official PatchTST implementation rather than a local transformer reimplementation.

- Official repository: https://github.com/yuqinie98/PatchTST
- Pinned commit used by the retained workflow: `204c21efe0b39603ad6e2ca640ef5896646ab1a9`
- Retained input mode: S-mode, one channel of strictly past normalized AC-PV history.
- Retained architecture: patch length 16, stride 8, end padding, RevIN, d_model 128, 16 heads, 3 encoder layers, d_ff 256, dropout 0.20, fc_dropout 0.20.
- Optimization: MSE, Adam, learning rate 1e-4, batch size 128, maximum 100 epochs, early-stopping patience 20, official type-3 learning-rate schedule.
- Validation-selected lookback: 168 h for pooled and all location-specific retained direct models at H24 and H48.

The retained direct PatchTST implementation is included as `code/direct_patchtst/solar_patchtst_original_protocol.py`, with its supporting dependency under `code/direct_patchtst/dependencies/`. The retained wrapper and dependency were recovered from the archived project materials and verified against their recorded SHA-256 values. Large trained-weight and prediction artifacts are included in the full reproducibility deposit and are intentionally omitted from the GitHub subset.
