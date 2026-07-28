# Code overview

## `codes/experiments`

Main experiment drivers.

- `diffusion/`: diffusion-based sequence generation and task-specific guidance experiments.
- `rnn_gan/`: RNN-GAN baseline and related comparative analyses.
- `rbs/`: RBS-specific diffusion and GA comparison scripts.

## `codes/search`

Hyperparameter search scripts for Oracle and diffusion components.

## `codes/preprocess`

Utilities for preparing raw or intermediate CSV files.

## `codes/analysis`

Scripts for summarizing training curves, generated-sequence diagnostics and final model assessments.

## `codes/tests`

Lightweight tests that can run without full datasets.

## Excluded from this release

Debug and sandbox scripts were not included in the clean GitHub-ready tree because they duplicate formal analyses, contain exploratory code, or include hard-coded local paths.
