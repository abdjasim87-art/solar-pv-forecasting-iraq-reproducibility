# Original experiment environment

Verified retained records support the following core environment for the final
neural experiments:

- Cloud platform: RunPod
- GPU: NVIDIA GeForce RTX 4090
- Base container:
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Python: 3.11
- PyTorch: 2.4 family
- CUDA: 12.4
- Retained PatchTST runtime record: Python 3.11.10, PyTorch 2.4.1+cu124

The files in `collection_snapshot_2026-09-22/` were captured later while
assembling the repository on a different RunPod GPU (NVIDIA RTX PRO 4000
Blackwell). They are retained only as collection-time provenance. The
collection-time `pip_freeze.txt` is not claimed to be a historical lock of every
package version used in the original experiments.
