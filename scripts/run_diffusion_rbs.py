"""Run a compact RBS diffusion-guidance workflow.

This script is intended as a clean, parameterized replacement for multiple
historical RBS scripts that differed mainly in guidance scale and output path.
It trains an Oracle and denoiser from a processed RBS CSV, then exports guided
candidate sequences.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rnadgg.data import load_sequence_table
from rnadgg.diffusion import GaussianDiffusion
from rnadgg.models import OracleCNN, UNet1D
from rnadgg.sequence import decode_one_hot, sequences_to_tensor
from rnadgg.training import train_denoiser, train_oracle
from rnadgg.utils import ensure_dir, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and sample an RBS RNADGG model.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/rbs_data.csv"))
    parser.add_argument("--sequence-column", default=None)
    parser.add_argument("--label-column", default="rl")
    parser.add_argument("--output-dir", type=Path, default=Path("results/rbs_diffusion"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--oracle-epochs", type=int, default=30)
    parser.add_argument("--diffusion-epochs", type=int, default=80)
    parser.add_argument("--oracle-batch-size", type=int, default=256)
    parser.add_argument("--diffusion-batch-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}. Place the processed RBS CSV under data/processed/ "
            "or pass --data explicitly."
        )

    set_seed(args.seed)
    device = get_device()
    output_dir = ensure_dir(args.output_dir)

    sequences, labels, _ = load_sequence_table(
        args.data,
        sequence_column=args.sequence_column,
        label_column=args.label_column,
        nrows=args.nrows,
    )
    sequence_length = len(sequences[0])
    x = sequences_to_tensor(sequences, device=device)
    y = torch.tensor(labels.to_numpy(), dtype=torch.float32, device=device)

    idx_train, idx_holdout = train_test_split(
        range(len(sequences)), test_size=0.2, random_state=args.seed
    )
    idx_val, idx_test = train_test_split(
        idx_holdout, test_size=0.5, random_state=args.seed
    )
    x_train, y_train = x[idx_train], y[idx_train]
    x_val, y_val = x[idx_val], y[idx_val]
    x_test, y_test = x[idx_test], y[idx_test]

    oracle = OracleCNN(sequence_length=sequence_length).to(device)
    oracle = train_oracle(
        oracle,
        x_train,
        y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=args.oracle_epochs,
        batch_size=args.oracle_batch_size,
    )

    denoiser = UNet1D(channels=args.channels).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)
    denoiser = train_denoiser(
        denoiser,
        diffusion,
        x_train,
        epochs=args.diffusion_epochs,
        batch_size=args.diffusion_batch_size,
    )

    generated = diffusion.sample(
        denoiser,
        shape=(args.num_samples, 4, sequence_length),
        oracle=oracle,
        guidance_scale=args.guidance_scale,
    )
    generated_sequences = decode_one_hot(generated)
    with torch.no_grad():
        scores = oracle(generated).squeeze(-1).detach().cpu().numpy()
        test_mse = torch.mean((oracle(x_test).squeeze(-1) - y_test) ** 2).item()

    out = pd.DataFrame({"sequence": generated_sequences, "oracle_score": scores})
    out.to_csv(output_dir / "generated_sequences.csv", index=False)
    torch.save(oracle.state_dict(), output_dir / "oracle.pt")
    torch.save(denoiser.state_dict(), output_dir / "denoiser.pt")
    print(f"Held-out test MSE: {test_mse:.6f}")
    print(f"Saved generated sequences and checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
