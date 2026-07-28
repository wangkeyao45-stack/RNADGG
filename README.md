# RNADGG

Gradient-guided diffusion for functional regulatory RNA sequence design.

RNADGG couples a diffusion sequence generator with assay-trained functional predictors, referred to as Oracles, to guide RNA sequence generation toward desired reporter-assay readouts. The repository contains code used for regulatory RNA design experiments on ribosome binding sites, 5-prime untranslated regions, and toehold switches.

<p align="center">
  <img src="docs/graphic_abstract.png" alt="RNADGG graphical abstract" width="850"/>
</p>

## Repository layout

```text
RNADGG/
  codes/
    experiments/      Main experiment scripts for diffusion, RNN-GAN and RBS baselines
    search/           Hyperparameter search scripts for Oracles and diffusion models
    preprocess/       Dataset preprocessing utilities
    analysis/         Result summarization and diagnostic scripts
    tests/            Lightweight smoke tests
  data/
    raw/              Raw data files, not tracked by git
    processed/        Processed task CSV files, not tracked by git
  docs/               Project documentation and graphical abstract
  models/             Model checkpoints, not tracked by git
  results/            Generated outputs, not tracked by git
```

## Installation

Python 3.8 or newer is recommended. GPU support is recommended for full training, but the smoke test can run on CPU.

```bash
git clone https://github.com/<your-user>/RNADGG.git
cd RNADGG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA-enabled PyTorch, install the appropriate PyTorch build from the official PyTorch instructions before installing the remaining dependencies.

## Data

The full CSV datasets are not tracked in git. Place the files below under `data/processed/` before running full experiments:

```text
data/processed/rbs_data.csv
data/processed/rbs_data_f.csv
data/processed/utr_data.csv
data/processed/toehold_data.csv
data/processed/toehold_data_f.csv
```

The raw 5-prime UTR library can be placed at:

```text
data/raw/GSM3130443_designed_library.csv
```

See [docs/DATA.md](docs/DATA.md) for dataset provenance and expected filenames.

## Quick check

Run the smoke test first. It checks that PyTorch and the core model-shape assumptions work without loading datasets.

```bash
python codes/tests/run_smoke_test.py
```

## Main scripts

Representative entry points are:

```bash
python codes/experiments/rbs/experiment_rbs_diffusion_ga_comparison.py
python codes/experiments/diffusion/experiment_diffusion_rl_unified.py
python codes/experiments/rnn_gan/experiment_rnn_gan_rl_unified.py
python codes/search/search_oracle_hyperparams.py
python codes/search/search_diffusion_hyperparams_v3.py
```

Most historical experiment scripts were originally run as standalone scripts. Some expect the required task CSV to be present in the working directory or under `data/processed/`. For clean reproduction, copy or symlink the relevant CSV into the run directory, or adapt the `CSV_FILE` variable near the top of each script.

## Reproducibility notes

- Random seeds are set inside the major experiment scripts where possible.
- Full training can be GPU-intensive and may take hours depending on the task and sampling settings.
- Generated checkpoints, logs, sequence libraries and figures should be written to `models/` or `results/`, both of which are excluded from git.
- Revision-specific analyses are documented in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Citation

If you use this code, please cite the associated RNADGG manuscript and the original dataset papers listed in [CITATION.cff](CITATION.cff).

## License

This repository is released under the GNU General Public License v3.0. See [LICENSE](LICENSE).
