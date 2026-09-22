# Reproducibility and determinism

The inspected retained LSTM source fixes random seeds for Python, NumPy, and PyTorch, calls `torch.cuda.manual_seed_all` when CUDA is available, sets `torch.backends.cudnn.deterministic = True`, and disables cuDNN benchmarking.

A global call to `torch.use_deterministic_algorithms(True)` was not present in the inspected retained LSTM source. Full global deterministic-algorithm enforcement across every PatchTST/transfer wrapper has not been independently verified from the currently staged source set.

Therefore this repository supports computational traceability through archived configurations, checkpoints, predictions, hashes, and environment information, but **does not claim bitwise-identical reproduction across hardware or software environments**.
