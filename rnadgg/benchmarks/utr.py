"""UTR guidance and RNADGG/RNN-GAN experiments with explicit protocol profiles."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from ..data import COMMON_SEQUENCE_COLUMNS, find_column
from ..diffusion import GaussianDiffusion
from ..sequence import decode_one_hot, sequences_to_tensor
from ..utils import set_seed
from .gan import sample_gan, train_gan
from .models import UNet1D
from .toehold import Diffusion, overlap_counts, train_diffusion


@dataclass(frozen=True)
class Config:
    seed: int = 42
    oracle_epochs: int = 30
    diffusion_epochs: int = 80
    oracle_batch_size: int = 256
    diffusion_batch_size: int = 32
    generation_batch_size: int = 64
    generation_n: int = 2048
    diffusion_steps: int = 500
    base_channels: int = 64
    predictor_dropout: float = 0.3
    oracle_lr: float = 1e-3
    diffusion_lr: float = 1e-4
    gan_epochs: int = 40
    gan_batch_size: int = 128
    clipping_reference: str = "train"


class ScalarPredictor(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 64, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten(),
        )
        layers = [nn.Linear(128 * (50 // 4), 64), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(64, 1))
        self.fc_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_net(self.conv_net(x))


def split_indices(count, seed):
    train, remainder = train_test_split(np.arange(count), test_size=0.2, random_state=seed)
    validation, test = train_test_split(remainder, test_size=0.5, random_state=seed)
    return {"train": train, "validation": validation, "test": test}


def load_data(path, sequence_column=None, label_column="r1", smoke=False):
    frame = pd.read_csv(path, nrows=80 if smoke else None)
    sequence_column = sequence_column or find_column(frame, COMMON_SEQUENCE_COLUMNS, "sequence")
    if label_column not in frame:
        raise ValueError(f"Missing label column {label_column!r}; the manuscript table uses 'r1'.")
    sequences = frame[sequence_column].astype("string").str.strip().str.upper().str.replace("U", "T", regex=False)
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    valid = sequences.notna() & sequences.str.fullmatch("[ACGT]{50}", na=False) & np.isfinite(labels)
    if not valid.all():
        raise ValueError("UTR rows must have 50 A/C/G/T/U nucleotides and finite numeric labels.")
    if not smoke and len(frame) != 280000:
        raise ValueError("The manuscript UTR table requires 280,000 rows from processed_data.csv.")
    if sequences.duplicated().any():
        raise ValueError("Duplicate UTR sequences would violate the manuscript sequence-level split.")
    return sequences.tolist(), torch.tensor(labels.to_numpy(), dtype=torch.float32)


@torch.no_grad()
def predict(model, features, batch_size, device):
    """Score fixed batches without advancing the training or sampling RNG."""
    model.eval()
    return np.concatenate([
        model(features[start:start + batch_size].to(device)).cpu().numpy()
        for start in range(0, len(features), batch_size)
    ])


def fit_predictor(features, labels, splits, config, device):
    model = ScalarPredictor(config.predictor_dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.oracle_lr)
    best_loss = float("inf")
    best_state = None
    history = []
    train = splits["train"]
    validation = splits["validation"]
    # The archived scalar predictor iterated over fixed-order training batches.
    for epoch in range(config.oracle_epochs):
        model.train()
        for start in range(0, len(train), config.oracle_batch_size):
            index = train[start:start + config.oracle_batch_size]
            optimizer.zero_grad()
            loss = F.mse_loss(model(features[index].to(device)).squeeze(-1), labels[index].to(device))
            loss.backward()
            optimizer.step()
        estimated = predict(model, features[validation], config.oracle_batch_size, device)[:, 0]
        validation_loss = float(np.mean((estimated - labels[validation].numpy()) ** 2))
        history.append({"epoch": epoch + 1, "validation_mse": validation_loss})
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"[predictor] epoch {epoch + 1}/{config.oracle_epochs} val_mse={validation_loss:.6f}", flush=True)
    if best_state is None:
        raise RuntimeError("Predictor training produced no checkpoint.")
    model.load_state_dict(best_state)
    return model.eval(), history


def clipping_bounds(labels, train_indices, reference):
    selected = labels[train_indices] if reference == "train" else labels
    return float(selected.min()), float(selected.max())


def export_library(name, samples, predictor, bounds, test_mean, output, batch_size, device):
    raw = predict(predictor, samples, batch_size, device)[:, 0]
    clipped = np.clip(raw, *bounds)
    sequences = decode_one_hot(samples)
    pd.DataFrame({
        "sequence": sequences, "predicted_activity_raw": raw,
        "predicted_activity_clipped": clipped,
    }).to_csv(output / f"generated_{name}.csv", index=False)
    return {
        "method": name, "n_records": len(sequences), "n_unique": len(set(sequences)),
        "mean_predictor_score": float(clipped.mean()),
        "std_predictor_score": float(clipped.std()),
        "max_predictor_score": float(clipped.max()),
        "test_mean_predictor_score": test_mean,
        "relative_lift_vs_test_pct": float(100 * (clipped.mean() - test_mean) / abs(test_mean))
        if abs(test_mean) > 1e-8 else None,
    }


def run_experiment(args, config, guidance_levels):
    set_seed(config.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output = args.output / f"seed_{config.seed}"
    output.mkdir(parents=True, exist_ok=True)
    sequences, labels = load_data(args.data, args.sequence_column, args.label_column, args.smoke)
    splits = split_indices(len(sequences), config.seed)
    overlaps = overlap_counts(np.asarray(sequences), splits)
    if any(overlaps.values()):
        raise RuntimeError(f"Exact sequence overlap between partitions: {overlaps}")
    rows = np.concatenate(list(splits.values()))
    manifest = pd.DataFrame({
        "row_index": rows,
        "split": np.repeat(list(splits), [len(index) for index in splits.values()]),
        "sequence": np.asarray(sequences)[rows],
    }).sort_values("row_index")
    manifest.to_csv(output / "split_manifest.csv", index=False)
    bounds = clipping_bounds(labels, splits["train"], config.clipping_reference)
    configuration = {
        "config": asdict(config), "profile": args.profile,
        "settings_basis": "archived_utr_comparison" if args.legacy_utr else "manuscript_table_S2",
        "validation_only": args.smoke, "dry_run": args.dry_run,
        "data": str(args.data), "label_column": args.label_column,
        "guidance_levels": list(guidance_levels), "training_label_ranges": list(bounds),
        "split_rows": {name: len(index) for name, index in splits.items()},
        "cross_split_exact_sequence_overlap": overlaps,
        "torch_version": torch.__version__, "device": str(device),
        "with_gan": args.profile == "main" and not args.skip_gan,
    }
    (output / "run_config.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(f"Validated UTR data and saved split manifest to {output}")
        return []

    features = sequences_to_tensor(sequences)
    predictor, history = fit_predictor(features, labels, splits, config, device)
    torch.save(predictor.state_dict(), output / "predictor.pt")
    pd.DataFrame(history).to_csv(output / "predictor_training_history.csv", index=False)
    test_predictions = predict(predictor, features[splits["test"]], config.oracle_batch_size, device)[:, 0]
    pd.DataFrame({
        "row_index": splits["test"], "observed": labels[splits["test"]].numpy(),
        "predicted_raw": test_predictions, "predicted_clipped": np.clip(test_predictions, *bounds),
    }).to_csv(output / "heldout_predictions.csv", index=False)
    test_mean = float(np.clip(test_predictions, *bounds).mean())

    denoiser = UNet1D(config.base_channels).to(device)
    diffusion = Diffusion(denoiser, config.diffusion_steps, device)
    loader = DataLoader(
        TensorDataset(features[splits["train"]]), batch_size=config.diffusion_batch_size,
        shuffle=True, drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("Diffusion training needs at least one full minibatch.")
    diffusion_history = train_diffusion(diffusion, loader, config, device)
    torch.save(denoiser.state_dict(), output / "denoiser.pt")
    pd.DataFrame(diffusion_history).to_csv(output / "diffusion_training_history.csv", index=False)
    sampler = GaussianDiffusion(timesteps=config.diffusion_steps, device=device)
    results = []
    for guidance in guidance_levels:
        batches = []
        for start in range(0, config.generation_n, config.generation_batch_size):
            count = min(config.generation_batch_size, config.generation_n - start)
            continuous = sampler.sample(
                denoiser, (count, 4, 50), oracle=predictor, guidance_scale=guidance,
            )
            tokens = continuous.argmax(dim=1)
            batches.append(F.one_hot(tokens, num_classes=4).float().permute(0, 2, 1).cpu())
        results.append(export_library(
            f"diffusion_g{guidance:g}", torch.cat(batches), predictor, bounds, test_mean,
            output, config.generation_batch_size, device,
        ))

    if args.profile == "main" and not args.skip_gan:
        generator, discriminator = train_gan(
            features[splits["train"]], device, config.seed, config.gan_epochs, config.gan_batch_size,
        )
        torch.save(generator.state_dict(), output / "gan_generator.pt")
        torch.save(discriminator.state_dict(), output / "gan_discriminator.pt")
        generated = sample_gan(generator, config.generation_n, config.generation_batch_size, device)
        results.append(export_library(
            "rnn_gan", generated, predictor, bounds, test_mean, output,
            config.generation_batch_size, device,
        ))
    for row in results:
        row["seed"] = config.seed
    pd.DataFrame(results).to_csv(output / "results.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps({"protocol": configuration, "results": results}, indent=2) + "\n", encoding="utf-8"
    )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run UTR experiments using explicit manuscript or archived settings.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sequence-column")
    parser.add_argument("--label-column", default="r1")
    parser.add_argument("--profile", choices=["main", "guidance"], default="main")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--guidance-levels", type=float, nargs="+")
    parser.add_argument("--skip-gan", action="store_true")
    parser.add_argument("--legacy-utr", action="store_true", help="Use the archived comparison's dropout, batch and clipping settings.")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.output is None:
        name = "utr_archived_comparison" if args.legacy_utr else f"utr_{args.profile}"
        suffix = "_smoke" if args.smoke else ("_audit" if args.dry_run else "")
        args.output = Path("results") / (name + suffix)
    config = Config()
    seeds = args.seeds or ([42] if args.profile == "guidance" else [42, 43, 44, 123, 456])
    levels = args.guidance_levels or ([0, 0.25, 0.5, 0.75, 1] if args.profile == "guidance" else [1.0])
    if any(not np.isfinite(level) or level < 0 for level in levels):
        parser.error("Guidance levels must be finite and non-negative.")
    if args.profile == "guidance":
        config = replace(config, generation_n=1024, oracle_batch_size=128)
    if args.legacy_utr:
        if args.profile != "main":
            parser.error("--legacy-utr is defined for the archived main comparison only.")
        config = replace(config, predictor_dropout=0.0, diffusion_batch_size=64, clipping_reference="encoded_dataset")
    if args.smoke:
        config = replace(
            config, oracle_epochs=1, diffusion_epochs=1, generation_n=8,
            diffusion_steps=4, base_channels=8, oracle_batch_size=8,
            diffusion_batch_size=8, generation_batch_size=8, gan_epochs=1, gan_batch_size=8,
        )
    results = []
    for seed in seeds:
        results.extend(run_experiment(args, replace(config, seed=seed), levels))
    if results:
        frame = pd.DataFrame(results)
        frame.to_csv(args.output / "per_seed_results.csv", index=False)
        frame.groupby("method")["mean_predictor_score"].agg(["count", "mean", "std"]).to_csv(
            args.output / "aggregate_results.csv"
        )


if __name__ == "__main__":
    main()
