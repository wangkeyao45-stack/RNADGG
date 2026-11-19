# Code overview

## `rnadgg/`

Python package for model development and small reproduction workflows.

- `sequence.py`: one-hot encoding, decoding and simple sequence statistics.
- `data.py`: CSV loading helpers for sequence-function tables.
- `models.py`: shared Oracle CNN and 1D U-Net modules.
- `diffusion.py`: Gaussian diffusion wrapper implementing the manuscript predicted-noise guidance update.
- `training.py`: predictor fitting and Gaussian noise-prediction training.
- `smoke.py`: package-level smoke test.
- `benchmarks/models.py`: manuscript U-Net, ported with the archived toehold checkpoint parameter names and final residual block.
- `benchmarks/utr.py`: UTR partitions, predictor training, guidance sweep, scoring and multi-run comparison.
- `benchmarks/gan.py`: GRU generator and CNN discriminator from the archived UTR comparison.
- `benchmarks/toehold.py`: original grouped-split two-output predictor and scaffold-constrained weight series.

`tests/` covers guidance scaling, forward and reverse diffusion, sequence encoding, CSV input handling, exported sequence scores and release-file checks. `.github/workflows/tests.yml` runs CPU checks on Python 3.10 and 3.12.

## `scripts/`

Small command-line utilities for dataset inspection, smoke tests and example runs.

The public scripts are parameterized and use the shared package modules.

`run_diffusion_rbs.py` writes candidate sequences and scores, two model checkpoints and `run_config.json`. Scores are evaluated on the exported nucleotide sequences. `check_release.py` checks tracked and candidate files without scanning ignored local data or checkpoints.

## Current scope

The RBS example retains its compact architecture. The UTR and grouped-toehold drivers use the manuscript architecture and record their effective settings separately. See [EXPERIMENTS.md](EXPERIMENTS.md) for the method-to-code mapping and remaining historical reproduction gaps.
