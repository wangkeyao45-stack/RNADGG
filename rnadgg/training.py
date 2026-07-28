"""Training helpers for lightweight RNADGG workflows."""

from __future__ import annotations

import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def train_oracle(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> nn.Module:
    """Train an Oracle model with mean squared error loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x_train, y_train.float()), batch_size=batch_size, shuffle=True)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = F.mse_loss(model(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()

        if x_val is None or y_val is None:
            best_state = copy.deepcopy(model.state_dict())
            continue

        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(x_val).squeeze(-1), y_val.float()).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


def train_denoiser(
    model: nn.Module,
    diffusion,
    x_train: torch.Tensor,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-4,
) -> nn.Module:
    """Train a diffusion denoiser to predict Gaussian noise."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for (xb,) in loader:
            t = torch.randint(0, diffusion.timesteps, (xb.shape[0],), device=xb.device)
            noise = torch.randn_like(xb)
            noisy = diffusion.q_sample(xb, t, noise=noise)
            predicted = model(noisy, t)
            loss = F.mse_loss(predicted, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model
