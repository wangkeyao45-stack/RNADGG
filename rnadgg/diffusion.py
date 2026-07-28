"""Minimal Gaussian diffusion utilities with optional Oracle guidance."""

from __future__ import annotations

import torch
from torch import nn


class GaussianDiffusion:
    """Gaussian diffusion wrapper for 1D one-hot sequence tensors."""

    def __init__(
        self,
        timesteps: int = 500,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | None = None,
    ):
        self.timesteps = timesteps
        self.device = device or torch.device("cpu")
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Diffuse clean samples `x0` to timestep `t`."""
        noise = torch.randn_like(x0) if noise is None else noise
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    @torch.no_grad()
    def denoise_step(self, model: nn.Module, x: torch.Tensor, t: int) -> torch.Tensor:
        """Single reverse step without Oracle guidance."""
        t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        predicted_noise = model(x, t_batch)
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]
        mean = (x - beta / (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
        if t == 0:
            return mean
        return mean + beta.sqrt() * torch.randn_like(x)

    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int],
        oracle: nn.Module | None = None,
        guidance_scale: float = 0.0,
    ) -> torch.Tensor:
        """Sample sequences, optionally guided by gradients from an Oracle."""
        x = torch.randn(shape, device=self.device)
        model.eval()
        if oracle is not None:
            oracle.eval()

        for t in reversed(range(self.timesteps)):
            if oracle is not None and guidance_scale > 0:
                x = x.detach().requires_grad_(True)
                score = oracle(x).mean()
                grad = torch.autograd.grad(score, x)[0]
                x = x.detach() + guidance_scale * grad.detach()
            x = self.denoise_step(model, x, t)
        return x
