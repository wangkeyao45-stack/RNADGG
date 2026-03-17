"""
Standalone smoke test: verifies PyTorch and core model components without
loading data or running full training. Runs in seconds.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

SEQUENCE_LENGTH = 17
NUCLEOTIDES = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MinimalUNet(nn.Module):
    """Minimal UNet for smoke test (matches experiment architecture)."""
    def __init__(self, n_channels=64):
        super().__init__()
        self.in_conv = nn.Conv1d(NUCLEOTIDES, n_channels, 3, padding=1)
        self.out_conv = nn.Conv1d(n_channels, NUCLEOTIDES, 1)

    def forward(self, x, t):
        h = self.in_conv(x)
        h = torch.relu(h)
        return self.out_conv(h)


class MinimalOracle(nn.Module):
    """Minimal Oracle CNN for smoke test."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(NUCLEOTIDES, 32, 5, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * SEQUENCE_LENGTH, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    print("Running smoke test (standalone, no data loading)...")
    print(f"Device: {device}")

    unet = MinimalUNet().to(device)
    oracle = MinimalOracle().to(device)

    # Diffusion-like forward
    x = torch.randn(2, NUCLEOTIDES, SEQUENCE_LENGTH, device=device)
    t = torch.randint(0, 100, (2,), device=device)
    pred = unet(x, t.float())
    assert pred.shape == x.shape, f"UNet shape mismatch: {pred.shape}"

    # Oracle forward
    scores = oracle(x)
    assert scores.shape == (2, 1), f"Oracle shape mismatch: {scores.shape}"

    print("All checks passed. Smoke test OK.")


if __name__ == "__main__":
    main()
