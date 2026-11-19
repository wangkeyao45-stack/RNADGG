# Reproducibility

This document records the recommended order for checking and reproducing RNADGG analyses.

## 1. Environment check

```bash
python -m rnadgg.smoke
python -m unittest discover -s tests -v
python scripts/check_release.py
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

Keep `run_config.json` alongside the two model checkpoints and `generated_sequences.csv`. It records the supplied arguments, sequence length, noise-schedule endpoints, device and PyTorch version. The example uses a seeded 80/10/10 row split. It does not provide the frozen split manifests used in the manuscript.

The exported `oracle_score` is evaluated on the decoded nucleotide sequence after re-encoding it as one-hot, rather than on the continuous final diffusion state. This example reports raw predictor scores without the benchmark-specific clipping described in the manuscript. Predictor scores estimate activity and are not new experimental measurements.

## 4. Guidance implementation

The implementation in `rnadgg/diffusion.py` applies the manuscript update to the predicted noise:

```text
guided_noise = predicted_noise - sqrt(1 - alpha_bar_t) * gamma * clipped_oracle_gradient
```

Predictor gradients are computed from a sum of per-sequence objectives, so their scale does not shrink with batch size. Gradients are clipped elementwise to `[-1, 1]` by default. See [DIFFUSION.md](DIFFUSION.md) for the forward and reverse distributions and noise-prediction objective.

## 5. Manuscript-level reproduction

The RBS driver is a compact example. UTR training, guidance sweeps and RNN-GAN comparisons, and the grouped scaffold-constrained toehold series are available as task-specific drivers. [EXPERIMENTS.md](EXPERIMENTS.md) maps their settings to the manuscript and records the historical UTR discrepancies. Every run saves its effective configuration and data partition manifest. Exact historical reproduction requires the original processed data and frozen generated libraries and checkpoints.

## 6. Output policy

Write generated checkpoints, figures, logs and sequence libraries to `models/` or `results/`. These directories are excluded from git. For manuscript-level reproducibility, release the exact generated sequence libraries and benchmark tables through an external archive with a DOI.

## Known limitations of this code release

The unprojected 200-sequence Figure 5 analysis, independent-predictor rescoring and all frozen source libraries are not yet packaged as complete reproduction pipelines. The historical UTR settings disagree with the current manuscript in three documented places. The public RBS script uses compact defaults intended for code inspection and small runs. See EXPERIMENTS.md before comparing new runs with published tables.
