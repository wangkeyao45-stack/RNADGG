"""Package-level smoke test."""

from __future__ import annotations

import torch

from .diffusion import GaussianDiffusion
from .models import OracleCNN, UNet1D
from .sequence import decode_one_hot
from .utils import get_device, set_seed


def run_smoke_test() -> None:
    """Run a lightweight shape and sampling check without external data."""
    set_seed(42)
    device = get_device()
    sequence_length = 17
    batch_size = 2
    print(f"Device: {device}")

    denoiser = UNet1D(alphabet_size=4, channels=8).to(device)
    oracle = OracleCNN(sequence_length=sequence_length).to(device)
    diffusion = GaussianDiffusion(timesteps=4, device=device)

    x = torch.randn(batch_size, 4, sequence_length, device=device)
    t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)
    assert denoiser(x, t).shape == x.shape
    assert oracle(x).shape == (batch_size, 1)

    generated = diffusion.sample(denoiser, (batch_size, 4, sequence_length), oracle=oracle, guidance_scale=0.01)
    decoded = decode_one_hot(generated)
    assert len(decoded) == batch_size
    assert all(len(seq) == sequence_length for seq in decoded)
    print("RNADGG smoke test passed.")


if __name__ == "__main__":
    run_smoke_test()
