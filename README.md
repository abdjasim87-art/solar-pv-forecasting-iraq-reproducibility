# Solar PV Forecasting in Iraq — Reproducibility Repository

This repository provides the browseable source code, documentation, selection
records, compact results, manifests, and environment/provenance notes supporting
the study:

**Multi-Horizon Solar Photovoltaic Power Forecasting at Representative Locations
Across Iraq: Direct Evaluation and Cross-Location Transfer**

## Scope

The study evaluates direct H24/H48 solar PV forecasting and cross-location
transfer at 15 fixed NASA POWER extraction points in Iraq.

The forecast target is a standardized physics-based normalized AC-PV reference
derived from NASA POWER. It is not metered plant production.

## Repository contents

- `code/` — retained forecasting, baseline, and analysis code.
- `docs/` — predictor, target, site, protocol, and reproducibility documentation.
- `selection_records/` — validation/selection and protocol records.
- `results/` — compact authoritative result/audit material suitable for GitHub.
- `manifests/` — component provenance and integrity information.
- `environment/` — documented original experiment environment and separately
  labeled collection-time environment information.

## Full reproducibility archive

Large trained weights, prediction artifacts, and transfer-learning provenance
packages are intentionally not stored in GitHub.

The complete reproducibility deposit is archived separately in Zenodo.

**Zenodo DOI (Version 1.0.0):** https://doi.org/10.5281/zenodo.22903000

**Zenodo Concept DOI (all versions):** https://doi.org/10.5281/zenodo.22902999

## Interpretation boundary

The retained LSTM and official PatchTST systems do not use matched predictor
representations. The LSTM uses 53 leakage-controlled predictors, whereas the
official PatchTST S-mode uses one channel of strictly past normalized AC-PV.
Their reported comparison is therefore an end-to-end forecasting-system
comparison rather than a same-input architecture ablation.

## Reproducibility boundary

Random seeds were fixed for Python/NumPy/PyTorch and deterministic cuDNN mode
was used where documented. Global deterministic-algorithm enforcement across
the complete neural workflow was not verified, so bitwise-identical results
across all hardware/software environments are not claimed.

## Licensing

Author-created code is licensed under the MIT License. Author-created non-code
research materials are released under CC BY 4.0. Third-party components retain
their original upstream licenses. See `LICENSES.md`.
