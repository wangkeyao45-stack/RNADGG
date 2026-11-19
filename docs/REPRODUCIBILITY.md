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

The current RBS driver is a compact example and does not freeze the complete manuscript protocol. Full manuscript-level reproduction requires an external archive containing the exact processed datasets, split manifests, task-specific configurations and drivers, generated sequence libraries, benchmark tables and checkpoints.

## 6. Output policy

Write generated checkpoints, figures, logs and sequence libraries to `models/` or `results/`. These directories are excluded from git. For manuscript-level reproducibility, release the exact generated sequence libraries and benchmark tables through an external archive with a DOI.

## Known limitations of this code release

The repository does not yet include the complete UTR and constrained-toehold workflows, RNN-GAN comparisons, independent-Oracle rescoring, or the frozen source libraries used for every manuscript figure. The public RBS script also uses compact defaults intended for code inspection and small runs. These limitations should be resolved before describing the repository as a complete reproduction package.
