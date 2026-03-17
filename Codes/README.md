# Codes Directory

Main scripts for 5′ UTR / RBS sequence modeling and generation.

Recommended: run commands from the **repository root** so relative paths resolve consistently.

---

## Data Requirements

Data lives under `Data/`. For backward-compatibility, the repository root contains symlinks like `rbs_data.csv` that point into `Data/`.

| File | Description |
|------|-------------|
| `processed_data.csv` | Preprocessed UTR sequences + r1 scores (from GEO or `preprocess_utr_data.py`) |
| `rbs_data.csv` | RBS sequences + rl scores |
| `utr_data.csv` | UTR sequences (from `preprocess_utr_data.py`) |
| `toehold_data.csv` | Toehold switch data (optional) |
| `GSM3130443_designed_library.csv` | Raw GEO library (for preprocessing) |

---

## Script Categories

### Preprocessing

| Script | Command | Description |
|--------|---------|-------------|
| `preprocess_utr_data.py` | `python Codes/preprocess/preprocess_utr_data.py` | Extract UTR + rl from GEO library |
| `preprocess_csv_generic.py` | `python Codes/preprocess/preprocess_csv_generic.py` | Compute ON/OFF ratio for toehold data |
| `analyze_sequence_counts.py` | `python Codes/preprocess/analyze_sequence_counts.py` | Count rows in a CSV |
| `count_sequences.py` | `python Codes/preprocess/count_sequences.py` | Count rows in a CSV |

### Core Experiments (full training + evaluation)

| Script | Command | Description |
|--------|---------|-------------|
| `experiment_rnn_gan_rl_unified.py` | `python Codes/experiments/experiment_rnn_gan_rl_unified.py` | RNN-GAN + RL fine-tuning, full evaluation |
| `experiment_diffusion_rl_unified.py` | `python Codes/experiments/experiment_diffusion_rl_unified.py` | Diffusion + RL-guided sampling, full evaluation |
| `experiment_rbs_diffusion_ga_comparison.py` | `python Codes/experiments/experiment_rbs_diffusion_ga_comparison.py` | RBS: Diffusion + GA + RL comparison |
| `experiment_rnn_gan_baseline_comprehensive_v2.py` | `python Codes/experiments/rnn_gan/experiment_rnn_gan_baseline_comprehensive_v2.py` | RNN-GAN baseline (comprehensive) |
| `experiment_rbs_diffusion_ga_variant2.py` | `python Codes/experiments/rbs/experiment_rbs_diffusion_ga_variant2.py` | RBS diffusion variant (26k data) |
| `experiment_rbs_diffusion_ga_variant3.py` | `python Codes/experiments/rbs/experiment_rbs_diffusion_ga_variant3.py` | RBS diffusion variant (strict train/val/test split) |

### Diffusion Experiments (by dataset & guidance)

| Script | Description |
|--------|-------------|
| `Codes/experiments/diffusion/utr/experiment_utr_diffusion_guidance*.py` | UTR diffusion (guidance 0.1, 0.5, 1.0) |
| `Codes/experiments/diffusion/rbs/experiment_rbs_diffusion_guidance*.py` | RBS diffusion (guidance 0.1, 0.5, 1.0, hyperparam variants) |
| `Codes/experiments/diffusion/toehold/experiment_toehold_diffusion*.py` | Toehold diffusion |
| `Codes/experiments/diffusion/two_stage/experiment_two_stage_diffusion*.py` | Two-stage diffusion |

### Hyperparameter Search

| Script | Command | Description |
|--------|---------|-------------|
| `search_oracle_hyperparams.py` | `python Codes/search/search_oracle_hyperparams.py` | Oracle CNN: channels, LR, dropout |
| `search_diffusion_hyperparams_v1.py` | `python Codes/search/search_diffusion_hyperparams_v1.py` | Diffusion: LR, schedule, guidance, T |
| `search_diffusion_hyperparams_v2.py` | `python Codes/search/search_diffusion_hyperparams_v2.py` | Diffusion (uses variant2 base) |
| `search_diffusion_hyperparams_v3.py` | `python Codes/search/search_diffusion_hyperparams_v3.py` | Diffusion (uses variant3 base, multi-seed) |

### Analysis & Evaluation

| Script | Command | Description |
|--------|---------|-------------|
| `analyze_oracle_search_results.py` | `python Codes/analysis/analyze_oracle_search_results.py` | Plot Oracle MSE trends |
| `analyze_generation_results_overview.py` | `python Codes/analysis/analyze_generation_results_overview.py` | Plot hyperparam training curves |
| `analyze_generation_results_rigorous.py` | `python Codes/analysis/analyze_generation_results_rigorous.py` | Plot generalization gap |
| `evaluate_final_models.py` | `python Codes/analysis/evaluate_final_models.py` | Final evaluation with trained UNet |
| `evaluate_complete_assessment.py` | `python Codes/analysis/evaluate_complete_assessment.py` | Full assessment over trained models |

### Sandbox (debug / exploratory)

| Script | Note |
|--------|------|
| `sandbox_debug_*.py`, `sandbox_experiment_code8.py` | Kept for reference; not part of main pipeline |

---

## Output Directories

Scripts write outputs to folders such as:

- `output_plots_*` – sequence logos, violin plots, etc.
- `results_hyperparam_*` – training metrics from hyperparameter search
- `oracle_search_results/` – Oracle search metrics
- `hyperparam_results_*` – diffusion hyperparameter results

These are ignored by `.gitignore`; regenerate by re-running the scripts.

---

## Quick Test

Verify the environment (runs in seconds):

```bash
python Codes/tests/run_smoke_test.py
```

Or syntax-check all scripts:

```bash
python -m compileall -q .
```

To run a full experiment (may take hours depending on data size and GPU):

```bash
python experiment_rnn_gan_rl_unified.py
```
