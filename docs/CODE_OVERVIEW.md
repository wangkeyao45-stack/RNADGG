# Code overview

## `rnadgg/`

Python package for model development and small reproduction workflows.

- `sequence.py`: one-hot encoding, decoding and simple sequence statistics.
- `data.py`: CSV loading helpers for sequence-function tables.
- `models.py`: shared Oracle CNN and 1D U-Net modules.
- `diffusion.py`: minimal Gaussian diffusion wrapper with optional Oracle guidance.
- `smoke.py`: package-level smoke test.

## `scripts/`

Small command-line utilities for dataset inspection, smoke tests and example runs.

The public scripts are parameterized and use the shared package modules.

## Excluded from this release

Debug and sandbox scripts are not included because they duplicate formal analyses, contain exploratory code, or include hard-coded local paths.
