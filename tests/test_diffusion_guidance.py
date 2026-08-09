import unittest

import torch
from torch import nn

from rnadgg.diffusion import GaussianDiffusion


class SumOracle(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1).sum(dim=1, keepdim=True)


class DiffusionGuidanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.diffusion = GaussianDiffusion(timesteps=4)

    def test_guided_noise_matches_manuscript_equation(self) -> None:
        predicted_noise = torch.full((2, 4, 3), 0.25)
        gradient = torch.full_like(predicted_noise, 0.5)
        t = 2
        gamma = 0.75

        actual = self.diffusion._guided_noise_prediction(
            predicted_noise, gradient, t=t, guidance_scale=gamma
        )
        expected = predicted_noise - (
            (1.0 - self.diffusion.alpha_bars[t]).sqrt() * gamma * gradient
        )

        torch.testing.assert_close(actual, expected)

    def test_oracle_gradient_does_not_shrink_with_batch_size(self) -> None:
        oracle = SumOracle()
        one = torch.zeros((1, 4, 3))
        four = torch.zeros((4, 4, 3))

        grad_one = self.diffusion._oracle_gradient(
            oracle, one, objective_fn=None, clip_value=1.0
        )
        grad_four = self.diffusion._oracle_gradient(
            oracle, four, objective_fn=None, clip_value=1.0
        )

        torch.testing.assert_close(grad_one[0], grad_four[0])
        torch.testing.assert_close(grad_four, torch.ones_like(grad_four))


if __name__ == "__main__":
    unittest.main()
