import unittest
from unittest.mock import patch

import torch
from torch import nn

from rnadgg.diffusion import GaussianDiffusion
from rnadgg.sequence import sequences_to_tensor


class ZeroDenoiser(nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


class DiffusionProcessTests(unittest.TestCase):
    def test_forward_process_uses_the_supplied_gaussian_noise(self):
        diffusion = GaussianDiffusion(timesteps=4)
        clean = sequences_to_tensor(["ACGT", "AAAA"])
        steps = torch.tensor([0, 3])
        noise = torch.linspace(-2.0, 2.0, clean.numel()).reshape_as(clean)
        noisy = diffusion.q_sample(clean, steps, noise=noise)
        alpha_bar = diffusion.alpha_bars[steps, None, None]
        recovered_noise = (noisy - alpha_bar.sqrt() * clean) / (1 - alpha_bar).sqrt()
        torch.testing.assert_close(recovered_noise, noise, atol=1e-5, rtol=1e-5)

    def test_reverse_noise_has_the_scheduled_scale(self):
        diffusion = GaussianDiffusion(timesteps=4)
        state = torch.zeros((1, 4, 3))
        with patch("rnadgg.diffusion.torch.randn_like", return_value=torch.ones_like(state)):
            previous = diffusion.denoise_step(ZeroDenoiser(), state, t=2)
        torch.testing.assert_close(previous, torch.ones_like(state) * diffusion.betas[2].sqrt())

    def test_final_reverse_step_does_not_draw_noise(self):
        diffusion = GaussianDiffusion(timesteps=4)
        state = torch.ones((1, 4, 3))
        with patch("rnadgg.diffusion.torch.randn_like") as noise:
            previous = diffusion.denoise_step(ZeroDenoiser(), state, t=0)
        noise.assert_not_called()
        torch.testing.assert_close(previous, state / diffusion.alphas[0].sqrt())


if __name__ == "__main__":
    unittest.main()
