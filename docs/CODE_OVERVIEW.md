# Code overview

## `rnadgg/`

Reusable Python package for new work.

- `sequence.py`: one-hot encoding, decoding and simple sequence statistics.
- `data.py`: CSV loading helpers for sequence-function tables.
- `models.py`: shared Oracle CNN and 1D U-Net modules.
- `diffusion.py`: minimal Gaussian diffusion wrapper with optional Oracle guidance.
- `smoke.py`: package-level smoke test.

## `scripts/`

Small command-line utilities intended as clean public entry points.

The main public scripts are intentionally small and parameterized. Historical standalone scripts were moved out of this upload-ready tree because they duplicated model definitions and included one-off local configuration.

## Excluded from this release

Debug, sandbox and historical standalone scripts were not included in the clean GitHub-ready tree because they duplicate formal analyses, contain exploratory code, or include hard-coded local paths.
