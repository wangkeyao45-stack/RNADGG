# RNADGG

Gradient-guided diffusion for functional regulatory RNA sequence design.

RNADGG combines a diffusion sequence generator with assay-trained sequence-to-function predictors, referred to as Oracles, to guide regulatory RNA sequence generation toward desired reporter-assay readouts. This repository contains the shared model components, the manuscript gradient-guidance update, data-handling utilities and a compact RBS example.

<p align="center">
  <img src="docs/graphic_abstract.png" alt="RNADGG graphical abstract" width="850"/>
</p>

## Repository structure

```text
RNADGG/
  rnadgg/       Python package for sequence utilities, models, data loading and diffusion helpers
  scripts/      Small command-line utilities for smoke tests and dataset inspection
  configs/      Example experiment configuration files
  data/         Local data directory, not tracked by git
  models/       Local checkpoints, not tracked by git
  results/      Local generated outputs, not tracked by git
  docs/         Data, reproducibility and code organization notes
```

The `rnadgg/` package is the main entry point. Development notebooks, debug scripts, generated outputs and local checkpoints are not included in the repository.

## Installation

Python 3.10 or newer is required. A CUDA-capable GPU is recommended for training, although the smoke test and unit tests can run on CPU.

```bash
git clone https://github.com/wangkeyao45-stack/RNADGG.git
cd RNADGG
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell, use:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

For CUDA-enabled PyTorch, install the appropriate PyTorch build from the official PyTorch instructions before installing the package.

## Quick checks

Run the package smoke test:

```bash
python -m rnadgg.smoke
```

or:

```bash
python scripts/run_smoke_test.py
```

Run the guidance regression tests:

```bash
python -m unittest discover -s tests -v
```

Inspect a dataset after placing it under `data/processed/`:

```bash
python scripts/inspect_dataset.py data/processed/rbs_data.csv --label-column rl
```

## Data

Full CSV datasets are not tracked in git. Place the processed files below under `data/processed/` before running full experiments:

```text
rbs_data.csv
rbs_data_f.csv
utr_data.csv
toehold_data.csv
toehold_data_f.csv
```

The raw 5-prime UTR library can be placed at:

```text
data/raw/GSM3130443_designed_library.csv
```

See [docs/DATA.md](docs/DATA.md) for dataset provenance.

## RBS example workflow

The package provides shared building blocks for sequence encoding, Oracle training, the 1D U-Net denoiser and guided diffusion sampling. A small RBS workflow is provided as:

```bash
python scripts/run_diffusion_rbs.py \
  --data data/processed/rbs_data.csv \
  --label-column rl \
  --guidance-scale 1.0 \
  --output-dir results/rbs_diffusion_g1
```

The guidance scale, number of diffusion steps, number of generated sequences and output directory can be changed from the command line.

## Scope of this release

The RBS command is a compact example, not a frozen reproduction of every manuscript benchmark. Full reproduction additionally requires the task-specific UTR and toehold drivers, exact data partitions, generated sequence libraries, benchmark tables, configurations and model checkpoints. Until those artifacts are deposited, the repository should be interpreted as an illustrative implementation of the shared method rather than a complete manuscript reproduction package.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for details.

## Output policy

Generated checkpoints, logs, figures and sequence libraries should be written to `models/` or `results/`. These directories are ignored by git. For publication-level reproducibility, archive generated sequence libraries and benchmark tables in a data repository such as Zenodo, Figshare, OSF or an institutional repository.

## Citation

If you use this code, please cite the associated RNADGG manuscript and the original dataset papers listed in [CITATION.cff](CITATION.cff).

## License

This repository is released under the GNU General Public License v3.0. See [LICENSE](LICENSE).
