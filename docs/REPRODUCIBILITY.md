# Reproducibility

This document records the recommended order for checking and reproducing RNADGG analyses.

## 1. Environment check

```bash
python codes/tests/run_smoke_test.py
```

The smoke test does not load data and should finish in seconds.

## 2. Data preparation

Copy the processed task CSVs into `data/processed/`. Some historical scripts read files from the current working directory through a `CSV_FILE` variable. If a script reports a missing CSV, either run it from `data/processed/` or edit the local `CSV_FILE` value to point to the corresponding file.

## 3. Representative experiments

RBS diffusion and GA comparison:

```bash
python codes/experiments/rbs/experiment_rbs_diffusion_ga_comparison.py
```

Unified diffusion experiment:

```bash
python codes/experiments/diffusion/experiment_diffusion_rl_unified.py
```

RNN-GAN baseline:

```bash
python codes/experiments/rnn_gan/experiment_rnn_gan_rl_unified.py
```

## 4. Hyperparameter searches

```bash
python codes/search/search_oracle_hyperparams.py
python codes/search/search_diffusion_hyperparams_v3.py
```

## 5. Output policy

Write generated checkpoints, figures, logs and sequence libraries to `models/` or `results/`. These directories are excluded from git. For manuscript-level reproducibility, release the exact generated sequence libraries and benchmark tables through an external archive with a DOI.

## Known limitations of this code release

Several scripts are historical standalone experiment drivers rather than a fully packaged Python API. They have been kept to preserve the original computational workflow. Before rerunning large experiments on a new system, check each script's top-level configuration block for dataset filename, sequence length, output directory, seed, number of diffusion steps and guidance strength.
