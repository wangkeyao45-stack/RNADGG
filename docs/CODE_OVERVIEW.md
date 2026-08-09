# Code overview

## `rnadgg/`

Python package for model development and small reproduction workflows.

- `sequence.py`: one-hot encoding, decoding and simple sequence statistics.
- `data.py`: CSV loading helpers for sequence-function tables.
- `models.py`: shared Oracle CNN and 1D U-Net modules.
- `diffusion.py`: Gaussian diffusion wrapper implementing the manuscript predicted-noise guidance update.
- `smoke.py`: package-level smoke test.
- `../tests/test_diffusion_guidance.py`: regression tests for the guidance equation and batch-size invariance.

## `scripts/`

Small command-line utilities for dataset inspection, smoke tests and example runs.

The public scripts are parameterized and use the shared package modules.

## Current scope

The public package provides shared components and a compact RBS example. It does not yet contain the complete frozen drivers, configurations and outputs required to reproduce every manuscript benchmark.
