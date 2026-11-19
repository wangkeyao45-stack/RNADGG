#!/usr/bin/env python3
"""Grouped-split, structure-constrained toehold RNADGG benchmark.

This script replaces the row-level split used by the earlier exploratory run. It
keeps duplicate full sequences in the same partition, projects every sampled
state onto the known 59-nt toehold scaffold, exports all generated sequences,
and reports held-out prediction metrics for the two-output Oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .models import UNet1D


BASES = "ACGT"
BASE_TO_INDEX = {base: index for index, base in enumerate(BASES)}
COMPLEMENT_CHANNELS = torch.tensor([3, 2, 1, 0], dtype=torch.long)
SEQUENCE_LENGTH = 59
LOOP2 = "AACAGAGGAGA"
START_CODON = "ATG"
WEIGHTS = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]


@dataclass(frozen=True)
class Config:
    seed: int = 42
    oracle_epochs: int = 35
    diffusion_epochs: int = 60
    oracle_batch_size: int = 256
    diffusion_batch_size: int = 64
    generation_batch_size: int = 128
    generation_n: int = 1024
    diffusion_steps: int = 400
    base_channels: int = 64
    guidance_scale: float = 0.5
    oracle_lr: float = 1e-3
    diffusion_lr: float = 1e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_sequences(sequences: list[str]) -> np.ndarray:
    encoded = np.zeros((len(sequences), 4, SEQUENCE_LENGTH), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        sequence = sequence.upper().replace("U", "T")
        if len(sequence) != SEQUENCE_LENGTH or any(base not in BASE_TO_INDEX for base in sequence):
            raise ValueError(f"Invalid 59-nt sequence at row {row}: {sequence!r}")
        for position, base in enumerate(sequence):
            encoded[row, BASE_TO_INDEX[base], position] = 1.0
    return encoded


def decode_sequences(one_hot: torch.Tensor) -> list[str]:
    indices = one_hot.argmax(dim=1).detach().cpu().numpy()
    return ["".join(BASES[index] for index in row) for row in indices]


def fixed_one_hot(sequence: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.zeros((4, len(sequence)), device=device, dtype=dtype)
    for position, base in enumerate(sequence):
        tensor[BASE_TO_INDEX[base], position] = 1.0
    return tensor


def project_toehold_scaffold(x: torch.Tensor) -> torch.Tensor:
    """Project continuous 4x59 tensors onto the measured toehold scaffold.

    Layout: switch[0:30], loop2[30:41], stem1[41:47], ATG[47:50],
    stem2[50:59]. Stem1 is reverse-complementary to switch[24:30], and
    stem2 is reverse-complementary to switch[12:21].
    """
    projected = x.clone()
    complement = COMPLEMENT_CHANNELS.to(x.device)
    projected[:, :, 30:41] = fixed_one_hot(LOOP2, x.device, x.dtype).unsqueeze(0)
    projected[:, :, 41:47] = x[:, complement, 24:30].flip(dims=[2])
    projected[:, :, 47:50] = fixed_one_hot(START_CODON, x.device, x.dtype).unsqueeze(0)
    projected[:, :, 50:59] = x[:, complement, 12:21].flip(dims=[2])
    return projected


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def validity_metrics(sequences: list[str]) -> dict[str, float | int]:
    checks = {
        "length_59": [len(sequence) == 59 for sequence in sequences],
        "fixed_loop2": [sequence[30:41] == LOOP2 for sequence in sequences],
        "fixed_atg": [sequence[47:50] == START_CODON for sequence in sequences],
        "stem1_complementarity": [sequence[41:47] == reverse_complement(sequence[24:30]) for sequence in sequences],
        "stem2_complementarity": [sequence[50:59] == reverse_complement(sequence[12:21]) for sequence in sequences],
    }
    result: dict[str, float | int] = {"n": len(sequences), "n_unique": len(set(sequences))}
    for name, values in checks.items():
        result[f"{name}_fraction"] = float(np.mean(values)) if values else math.nan
    valid = np.logical_and.reduce([np.asarray(values, dtype=bool) for values in checks.values()])
    result["all_constraints_fraction"] = float(valid.mean()) if len(valid) else math.nan
    return result












class ToeholdOracle(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 64, 7, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        self.outputs = nn.Sequential(
            nn.Linear(128 * (SEQUENCE_LENGTH // 4), 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.outputs(self.features(x))


class Diffusion:
    def __init__(self, model: nn.Module, steps: int, device: torch.device):
        self.model = model
        self.steps = steps
        self.beta = torch.linspace(1e-4, 0.02, steps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def training_loss(self, clean: torch.Tensor) -> torch.Tensor:
        timestep = torch.randint(0, self.steps, (clean.shape[0],), device=clean.device)
        noise = torch.randn_like(clean)
        alpha_bar = self.alpha_bar[timestep, None, None]
        noisy = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise
        return F.mse_loss(self.model(noisy, timestep.float()), noise)


def grouped_split(data: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = data["full_sequence"].astype(str).to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_index, remainder_index = next(outer.split(data, groups=groups))
    remainder = data.iloc[remainder_index]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    validation_rel, test_rel = next(inner.split(remainder, groups=remainder["full_sequence"].astype(str)))
    return train_index, remainder_index[validation_rel], remainder_index[test_rel]


def overlap_counts(sequences: np.ndarray, split_indices: dict[str, np.ndarray]) -> dict[str, int]:
    sets = {name: set(sequences[index]) for name, index in split_indices.items()}
    return {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_test": len(sets["train"] & sets["test"]),
        "validation_test": len(sets["validation"] & sets["test"]),
    }


def prediction_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, dict[str, float]]:
    names = ["on", "off"]
    result = {}
    for column, name in enumerate(names):
        truth = observed[:, column]
        estimate = predicted[:, column]
        result[name] = {
            "mse": float(mean_squared_error(truth, estimate)),
            "mae": float(mean_absolute_error(truth, estimate)),
            "r2": float(r2_score(truth, estimate)),
            "pearson_r": float(pearsonr(truth, estimate).statistic),
            "spearman_rho": float(spearmanr(truth, estimate).statistic),
        }
    return result


def train_oracle(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: Config,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.oracle_lr)
    best_loss = math.inf
    best_state = None
    history = []
    for epoch in range(1, config.oracle_epochs + 1):
        model.train()
        train_losses = []
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(sequences), labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for sequences, labels in validation_loader:
                validation_losses.append(F.mse_loss(model(sequences.to(device)), labels.to(device)).item())
        row = {
            "epoch": epoch,
            "train_mse": float(np.mean(train_losses)),
            "validation_mse": float(np.mean(validation_losses)),
        }
        history.append(row)
        if row["validation_mse"] < best_loss:
            best_loss = row["validation_mse"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == config.oracle_epochs:
            print(f"[oracle] epoch {epoch}/{config.oracle_epochs} val_mse={row['validation_mse']:.6f}")
    if best_state is None:
        raise RuntimeError("Oracle training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, history


def train_diffusion(
    diffusion: Diffusion,
    train_loader: DataLoader,
    config: Config,
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=config.diffusion_lr)
    history = []
    for epoch in range(1, config.diffusion_epochs + 1):
        losses = []
        diffusion.model.train()
        for (sequences,) in train_loader:
            sequences = sequences.to(device)
            optimizer.zero_grad()
            loss = diffusion.training_loss(sequences)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        row = {"epoch": epoch, "mean_noise_mse": float(np.mean(losses))}
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == config.diffusion_epochs:
            print(f"[diffusion] epoch {epoch}/{config.diffusion_epochs} noise_mse={row['mean_noise_mse']:.6f}")
    return history


def sample_batch(
    diffusion: Diffusion,
    oracle: nn.Module,
    batch_size: int,
    weight_on: float,
    weight_off: float,
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    state = project_toehold_scaffold(torch.randn(batch_size, 4, SEQUENCE_LENGTH, device=device))
    for timestep in tqdm(reversed(range(diffusion.steps)), total=diffusion.steps, leave=False):
        time_tensor = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
        if guidance_scale:
            with torch.enable_grad():
                state = state.detach().requires_grad_(True)
                projected = project_toehold_scaffold(state)
                predictions = oracle(projected)
                objective = weight_on * predictions[:, 0].sum() - weight_off * predictions[:, 1].sum()
                gradient = torch.autograd.grad(objective, state)[0].clamp(-1.0, 1.0)
        else:
            gradient = torch.zeros_like(state)
        with torch.no_grad():
            projected = project_toehold_scaffold(state.detach())
            predicted_noise = diffusion.model(projected, time_tensor.float())
            alpha = diffusion.alpha[timestep]
            alpha_bar = diffusion.alpha_bar[timestep]
            scale_t = guidance_scale * (timestep / diffusion.steps)
            guided_noise = predicted_noise - torch.sqrt(1.0 - alpha_bar) * gradient * scale_t
            mean = (projected - (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar) * guided_noise) / torch.sqrt(alpha)
            state = mean + torch.sqrt(diffusion.beta[timestep]) * torch.randn_like(state) if timestep else mean
            state = project_toehold_scaffold(state)
    return F.one_hot(state.argmax(dim=1), num_classes=4).float().permute(0, 2, 1)


def generate_library(
    diffusion: Diffusion,
    oracle: nn.Module,
    config: Config,
    weight_on: float,
    weight_off: float,
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    batches = []
    remaining = config.generation_n
    while remaining:
        current = min(config.generation_batch_size, remaining)
        batches.append(
            sample_batch(diffusion, oracle, current, weight_on, weight_off, guidance_scale, device).cpu()
        )
        remaining -= current
    return torch.cat(batches, dim=0)


def predict(model: nn.Module, features: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(features), batch_size=batch_size)
    outputs = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            outputs.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def run_experiment(args: argparse.Namespace, config: Config) -> None:
    set_seed(config.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output = args.output / f"seed_{config.seed}"
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    data = pd.read_csv(args.data)
    required = {"full_sequence", "ON", "OFF"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")
    data = data.dropna(subset=sorted(required)).reset_index(drop=True)
    if args.smoke:
        data = data.head(80).reset_index(drop=True)
    if not args.smoke and (len(data) != 44859 or data["full_sequence"].nunique() != 44660):
        raise ValueError("The manuscript toehold table requires 44,859 rows and 44,660 full-sequence groups.")
    features = torch.from_numpy(encode_sequences(data["full_sequence"].astype(str).tolist()))
    labels = torch.from_numpy(data[["ON", "OFF"]].to_numpy(dtype=np.float32))
    train_index, validation_index, test_index = grouped_split(data, config.seed)
    split_indices = {"train": train_index, "validation": validation_index, "test": test_index}
    overlaps = overlap_counts(data["full_sequence"].astype(str).to_numpy(), split_indices)
    if any(overlaps.values()):
        raise RuntimeError(f"Grouped split contains sequence overlap: {overlaps}")

    split_manifest = pd.DataFrame({
        "row_index": np.concatenate([train_index, validation_index, test_index]),
        "split": np.repeat(
            ["train", "validation", "test"],
            [len(train_index), len(validation_index), len(test_index)],
        ),
    }).sort_values("row_index")
    split_manifest["full_sequence"] = data.loc[split_manifest["row_index"], "full_sequence"].to_numpy()
    split_manifest.to_csv(output / "split_manifest.csv", index=False)
    (output / "run_config.json").write_text(
        json.dumps({
            "config": asdict(config),
            "validation_only": args.smoke,
            "dry_run": args.dry_run,
            "analysis": "grouped_scaffold_weight_series",
            "data": str(args.data),
            "split_rows": {name: len(index) for name, index in split_indices.items()},
            "cross_split_exact_sequence_overlap": overlaps,
            "torch_version": torch.__version__,
            "device": str(device),
        }, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print(f"Validated grouped split and saved manifest to {output}")
        return

    oracle_train = DataLoader(
        TensorDataset(features[train_index], labels[train_index]),
        batch_size=config.oracle_batch_size,
        shuffle=True,
    )
    oracle_validation = DataLoader(
        TensorDataset(features[validation_index], labels[validation_index]),
        batch_size=config.oracle_batch_size,
    )
    oracle, oracle_history = train_oracle(
        ToeholdOracle().to(device), oracle_train, oracle_validation, config, device
    )
    torch.save(oracle.state_dict(), output / "toehold_oracle.pt")
    pd.DataFrame(oracle_history).to_csv(output / "oracle_training_history.csv", index=False)

    test_prediction = predict(oracle, features[test_index], config.oracle_batch_size, device)
    test_truth = labels[test_index].numpy()
    test_metrics = prediction_metrics(test_truth, test_prediction)
    pd.DataFrame({
        "row_index": test_index,
        "full_sequence": data.loc[test_index, "full_sequence"].to_numpy(),
        "observed_on": test_truth[:, 0],
        "observed_off": test_truth[:, 1],
        "predicted_on": test_prediction[:, 0],
        "predicted_off": test_prediction[:, 1],
    }).to_csv(output / "heldout_predictions.csv", index=False)

    diffusion_loader = DataLoader(
        TensorDataset(features[train_index]),
        batch_size=config.diffusion_batch_size,
        shuffle=True,
    )
    diffusion = Diffusion(UNet1D(config.base_channels).to(device), config.diffusion_steps, device)
    diffusion_history = train_diffusion(diffusion, diffusion_loader, config, device)
    torch.save(diffusion.model.state_dict(), output / "toehold_diffusion.pt")
    pd.DataFrame(diffusion_history).to_csv(output / "diffusion_training_history.csv", index=False)

    train_min = labels[train_index].min(dim=0).values.numpy()
    train_max = labels[train_index].max(dim=0).values.numpy()
    rows = []
    libraries = [("unguided", math.nan, math.nan, 0.0)] + [
        (f"w=({weight_on:.2f},{weight_off:.2f})", weight_on, weight_off, config.guidance_scale)
        for weight_on, weight_off in WEIGHTS
    ]
    for name, weight_on, weight_off, scale in libraries:
        print(f"[generate] {name}")
        generated = generate_library(
            diffusion,
            oracle,
            config,
            0.0 if math.isnan(weight_on) else weight_on,
            0.0 if math.isnan(weight_off) else weight_off,
            scale,
            device,
        )
        sequences = decode_sequences(generated)
        predictions = predict(oracle, generated, config.generation_batch_size, device)
        clipped = np.clip(predictions, train_min, train_max)
        pd.DataFrame({
            "sequence": sequences,
            "predicted_on_raw": predictions[:, 0],
            "predicted_off_raw": predictions[:, 1],
            "predicted_on_clipped": clipped[:, 0],
            "predicted_off_clipped": clipped[:, 1],
        }).to_csv(output / f"generated_{name.replace('=', '').replace('(', '').replace(')', '').replace(',', '_')}.csv", index=False)
        validity = validity_metrics(sequences)
        row = {
            "method": name,
            "weight_on": weight_on,
            "weight_off": weight_off,
            "guidance_scale": scale,
            "mean_on": float(clipped[:, 0].mean()),
            "std_on": float(clipped[:, 0].std(ddof=1)),
            "mean_off": float(clipped[:, 1].mean()),
            "std_off": float(clipped[:, 1].std(ddof=1)),
            **validity,
        }
        if not math.isnan(weight_on):
            row["scalar_objective"] = float(weight_on * row["mean_on"] - weight_off * row["mean_off"])
        rows.append(row)

    results = pd.DataFrame(rows)
    unguided = results.loc[results["method"] == "unguided"].iloc[0]
    for index, row in results.iterrows():
        if pd.notna(row["weight_on"]):
            reference = row["weight_on"] * unguided["mean_on"] - row["weight_off"] * unguided["mean_off"]
            results.loc[index, "guided_minus_unguided_objective"] = row["scalar_objective"] - reference
    results.to_csv(output / "toehold_grouped_constrained_results.csv", index=False)

    summary = {
        "validation_only": args.smoke,
        "config": asdict(config),
        "device": str(device),
        "data": {
            "path": str(args.data),
            "n_rows": int(len(data)),
            "n_unique_sequences": int(data["full_sequence"].nunique()),
            "split_rows": {name: int(len(index)) for name, index in split_indices.items()},
            "split_unique_sequences": {
                name: int(data.loc[index, "full_sequence"].nunique()) for name, index in split_indices.items()
            },
            "cross_split_exact_sequence_overlap": overlaps,
        },
        "label_scale": "raw assay values; generated predictions clipped to training ranges for summaries",
        "training_label_ranges": {
            "on": [float(train_min[0]), float(train_max[0])],
            "off": [float(train_min[1]), float(train_max[1])],
        },
        "heldout_oracle_metrics": test_metrics,
        "architecture": {
            "oracle": "shared 2-layer 1D-CNN trunk with two outputs (ON and OFF)",
            "unet": f"two downsampling and two upsampling levels; {config.base_channels}/{config.base_channels * 2}/{config.base_channels * 4} channels; single-head attention in each level and bottleneck",
            "guidance": "predicted-noise correction; elementwise gradient clipping to [-1,1]; gamma_t=gamma*t/T",
        },
        "elapsed_seconds": time.time() - started,
    }
    with open(output / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the grouped toehold experiments in Tables S5 and S14.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true", help="Use at most 80 rows and tiny models for validation only.")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and save splits without training.")
    args = parser.parse_args(argv)
    if args.output is None:
        suffix = "_smoke" if args.smoke else ("_audit" if args.dry_run else "")
        args.output = Path("results") / ("toehold_grouped" + suffix)
    config = Config()
    if args.smoke:
        config = replace(
            config, oracle_epochs=1, diffusion_epochs=1, generation_n=8,
            diffusion_steps=4, base_channels=8, oracle_batch_size=8,
            diffusion_batch_size=8, generation_batch_size=8,
        )
    for seed in args.seeds:
        run_experiment(args, replace(config, seed=seed))


if __name__ == "__main__":
    main()
