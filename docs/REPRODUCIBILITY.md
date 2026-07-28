# Reproducibility

This document records the recommended order for checking and reproducing RNADGG analyses.

## 1. Environment check

```bash
python -m rnadgg.smoke
```

The smoke test does not load data and should finish in seconds.

## 2. Data preparation

Copy the processed task CSVs into `data/processed/`. Scripts in `scripts/` accept explicit paths and write outputs to `results/`.

## 3. Representative workflow

Compact RBS diffusion guidance run:

```bash
python scripts/run_diffusion_rbs.py \
  --data data/processed/rbs_data.csv \
  --label-column rl \
  --guidance-scale 1.0 \
  --output-dir results/rbs_diffusion_g1
```

## 4. Manuscript-level reproduction

This repository excludes large generated outputs. Full manuscript-level reproduction should be paired with an external archive containing exact processed datasets, generated sequence libraries, benchmark tables and checkpoints.

## 5. Output policy

Write generated checkpoints, figures, logs and sequence libraries to `models/` or `results/`. These directories are excluded from git. For manuscript-level reproducibility, release the exact generated sequence libraries and benchmark tables through an external archive with a DOI.

## Known limitations of this code release

The repository uses reusable modules and parameterized scripts. One-off development scripts are not included because they duplicated model definitions and depended on local configuration.
