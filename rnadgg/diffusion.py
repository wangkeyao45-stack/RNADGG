"""Minimal Gaussian diffusion utilities with optional Oracle guidance."""

from __future__ import annotations

from collections.abc import Callable

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
        return self._reverse_step(x, predicted_noise, t)

    def _reverse_step(self, x: torch.Tensor, predicted_noise: torch.Tensor, t: int) -> torch.Tensor:
        """Apply one Gaussian reverse step from a supplied noise prediction."""
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]
        mean = (x - beta / (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
        if t == 0:
            return mean
        return mean + beta.sqrt() * torch.randn_like(x)

    def _oracle_gradient(
        self,
        oracle: nn.Module,
        x: torch.Tensor,
        objective_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        clip_value: float,
    ) -> torch.Tensor:
        """Return a batch-size-invariant gradient of the per-sequence objective."""
        x_for_grad = x.detach().requires_grad_(True)
        oracle_output = oracle(x_for_grad)
        per_sequence = (
            objective_fn(oracle_output)
            if objective_fn is not None
            else oracle_output.reshape(x.shape[0], -1).sum(dim=1)
        )
        if per_sequence.shape != (x.shape[0],):
            raise ValueError("objective_fn must return one scalar per sequence")
        gradient = torch.autograd.grad(per_sequence.sum(), x_for_grad)[0]
        return gradient.detach().clamp(-clip_value, clip_value)

    def _guided_noise_prediction(
        self,
        predicted_noise: torch.Tensor,
        gradient: torch.Tensor,
        t: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Apply the manuscript guidance update to the predicted noise."""
        time_scale = (1.0 - self.alpha_bars[t]).sqrt()
        return predicted_noise - time_scale * guidance_scale * gradient

    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int],
        oracle: nn.Module | None = None,
        guidance_scale: float = 0.0,
        objective_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        gradient_clip: float = 1.0,
    ) -> torch.Tensor:
        """Sample sequences with the gradient-guidance update used in the manuscript."""
        if gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")

        x = torch.randn(shape, device=self.device)
        model.eval()
        if oracle is not None:
            oracle.eval()

        for t in reversed(range(self.timesteps)):
            if oracle is not None and guidance_scale > 0:
                with torch.enable_grad():
                    gradient = self._oracle_gradient(
                        oracle, x, objective_fn=objective_fn, clip_value=gradient_clip
                    )
                with torch.no_grad():
                    t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
                    predicted_noise = model(x.detach(), t_batch)
                    guided_noise = self._guided_noise_prediction(
                        predicted_noise, gradient, t, guidance_scale
                    )
                    x = self._reverse_step(x.detach(), guided_noise, t)
            else:
                x = self.denoise_step(model, x, t)
        return x
