# Manuscript experiments

The UTR and grouped-toehold drivers implement the settings in the current
manuscript Methods and Supplementary Table S2. The toehold model and sampling
functions were ported from `toehold_grouped_constrained.py`, retaining parameter
names so the archived predictor and denoiser checkpoints can be loaded directly.
UTR comparison and GAN settings were also checked against the archived revision
drivers. Differences between those historical settings and the current text are
listed below.

## Protocols

| Setting | UTR main comparison | UTR guidance sweep | Grouped toehold series |
|---|---|---|---|
| Input | 280,000 unique 50-nt sequences, `r1` labels | Same | 44,859 rows, 44,660 full-sequence groups, `ON` and `OFF` labels |
| Partition | 80/10/10, two seeded `train_test_split` calls | Same | Two seeded `GroupShuffleSplit` calls, grouped by full sequence |
| Seeds | 42, 43, 44, 123, 456 | 42 | 42, 43, 44 |
| Predictor | 64/128-channel CNN, kernels 7/5, pooling, FC 64, dropout 0.3, scalar output | Same | Shared CNN trunk, FC 64, dropout 0.3, two outputs |
| Predictor fitting | 30 epochs, Adam 0.001, batch 256, validation-MSE checkpoint | 30 epochs, batch 128 | 35 epochs, batch 256, validation-MSE checkpoint |
| Denoiser | 64/128/256 channels, two bottleneck residual blocks, final residual block | Same | Same |
| Diffusion fitting | 80 epochs, Adam 0.0001, batch 32 | Same | 60 epochs, batch 64 |
| Diffusion steps | 500 | 500 | 400 |
| Noise schedule | Linear beta from 0.0001 to 0.02 | Same | Same |
| Guidance | 1.0 | 0, 0.25, 0.5, 0.75, 1.0 | 0.5 with scale `gamma * t / T` |
| Generation budget | 2,048 records per method and seed | 1,024 per guidance setting | 1,024 per weight setting and seed |
| Gradient | Per-sequence objectives summed; elementwise clip to [-1, 1] | Same | Weighted ON maximization minus OFF minimization; gradient through scaffold projection |
| Decoding | Argmax, followed by re-encoding before scoring | Same | Argmax after scaffold projection |
| Reported activity | Predictor output clipped to training-label range | Same | Generated predictions clipped to each output's training range |

The UTR guidance value can be changed explicitly. Use `--guidance-levels 0.5`
with the main profile for the comparison reported at moderate guidance in
Supplementary Table S8. Main-profile generation includes the GRU/CNN RNN-GAN
baseline unless `--skip-gan` is passed. GAN training uses 40 epochs, batch 128,
Adam 0.0002 with betas (0.5, 0.999), a 32-dimensional normal input and a
128-dimensional GRU. The discriminator uses label targets 0.9 and 0.1. The GAN
seed is the run seed plus 99, as in the archived comparison driver.

The toehold series includes an unguided library and weights (1, 0), (0.75, 0.25),
(0.5, 0.5), (0.25, 0.75) and (0, 1). Fixed positions use the loop `AACAGAGGAGA`
and start codon `ATG`. Stem 1 is the reverse complement of switch positions
25-30 and stem 2 of positions 13-21, in one-based coordinates. Projection is
applied to both guided and unguided samples.

## Commands

Install the package as described in the README. Place the original processed
data locally; full datasets are not included in Git.

```bash
# Validate the full data and write split manifests without training.
python scripts/run_diffusion_utr.py --data data/processed/processed_data.csv --dry-run
python scripts/run_diffusion_toehold.py --data data/processed/toehold_data_f.csv --dry-run

# Main five-run UTR comparison and single-run guidance sweep.
python scripts/run_diffusion_utr.py --data data/processed/processed_data.csv --profile main
python scripts/run_diffusion_utr.py --data data/processed/processed_data.csv --profile guidance

# Moderate-guidance five-run comparison (use a separate output directory).
python scripts/run_diffusion_utr.py --data data/processed/processed_data.csv --profile main --guidance-levels 0.5 --output results/utr_main_g05

# Grouped, scaffold-constrained toehold series, seeds 42, 43 and 44.
python scripts/run_diffusion_toehold.py --data data/processed/toehold_data_f.csv

# Small execution checks using at most 80 input rows. These are not benchmark runs.
python scripts/run_diffusion_utr.py --data data/processed/processed_data.csv --smoke --seeds 42 --device cpu
python scripts/run_diffusion_toehold.py --data data/processed/toehold_data_f.csv --smoke --seeds 42 --device cpu --output results/toehold_smoke
```

Use `--seeds 42` for one full run. Device selection is automatic unless `--device`
is supplied. Full defaults fit models on all training rows; there is no implicit
8,000-row training cap. Smoke and dry-run defaults have separate output
directories from the full runs. Use dedicated output directories when changing
settings explicitly.

## Outputs and checks

Each seed directory contains the effective `run_config.json`, a row-level
`split_manifest.csv`, training histories, held-out predictions, model checkpoints,
generated sequences and result tables. Raw and clipped generated predictions are
both exported. Counts are computed over sampled records; unique counts are
reported separately, and generated libraries are not silently deduplicated.
UTR runs also write per-seed and cross-seed aggregate tables. Toehold per-seed
tables contain each constraint-valid fraction and objective change from the
corresponding unguided library.

`--smoke` uses one epoch, four diffusion steps, eight-channel denoisers and eight
generated records per setting. Output metadata sets `validation_only` to true.
Smoke metrics must not be substituted for manuscript results.

Regression tests cover the published defaults, full UTR partition sizes, grouped
split disjointness, gradient propagation through tied stems, scaffold validity,
checkpoint export and post-decoding scoring. The port was additionally compared
with the original toehold source: identically seeded models, forward outputs and
guided/unguided samples matched exactly in the same environment, and the
archived seed-42 predictor and denoiser checkpoints loaded without key conversion.

## Historical UTR discrepancies

The archived `run_revision_utr_benchmark` and
`run_revision_gan_diffusion_compare_utr` implementations differ from the current
manuscript in these settings:

| Setting | Current manuscript/default | Archived UTR comparison |
|---|---|---|
| Predictor dropout | 0.3 | No dropout |
| Diffusion training minibatch | 32 | 64 |
| Prediction-clipping reference | Training-label range | Complete encoded dataset's label range |

The `--legacy-utr` option with the main profile applies the archived settings
and records `settings_basis: archived_utr_comparison`. It uses a separate default
output directory. For example:

```bash
python scripts/run_diffusion_utr.py --data data/processed/processed_data.csv --legacy-utr --guidance-levels 0.5
```

This option makes the historical settings inspectable. It does not certify
bitwise reproduction of the archived numerical tables: library versions, device,
model initialization, data order and RNG consumption also affect training and
sampling. The author must resolve the manuscript-versus-historical settings
before attributing newly generated numbers to a particular published table.

## Remaining reproduction artifacts

The grouped-toehold driver targets Supplementary Figure S2 and Tables S5/S14.
Figure 5 uses a separate, unprojected 200-sequence equal-weight analysis. Its
frozen sequence sets, predictor checkpoints and reference used for z-score
normalization must be recovered to reproduce that panel exactly. Do not replace
its libraries with the scaffold-constrained series.

Independent-predictor rescoring, all original figure libraries and a permanent
data/checkpoint archive are also still required for a complete frozen
reproduction of every manuscript result. Existing manuscript files and
archived experimental results are not changed by these code ports.
