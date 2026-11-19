from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch
from torch import nn

from scripts import run_diffusion_rbs


class FirstChannelPredictor(nn.Module):
    def forward(self, x):
        return x[:, 0, :].sum(dim=1, keepdim=True)


class RBSWorkflowTests(unittest.TestCase):
    def test_scores_describe_exported_nucleotides(self):
        continuous = torch.full((2, 4, 17), -0.5)
        continuous[:, 0, :] = 2.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "input.csv"
            output = root / "output"
            pd.DataFrame({"sequence": ["A" * 17] * 20, "rl": range(20)}).to_csv(
                data, index=False
            )
            arguments = [
                "run_diffusion_rbs.py", "--data", str(data),
                "--output-dir", str(output), "--num-samples", "2", "--timesteps", "4",
            ]
            with (
                patch("sys.argv", arguments),
                patch.object(run_diffusion_rbs, "get_device", return_value=torch.device("cpu")),
                patch.object(run_diffusion_rbs, "OracleCNN", return_value=FirstChannelPredictor()),
                patch.object(run_diffusion_rbs, "UNet1D", return_value=nn.Identity()),
                patch.object(run_diffusion_rbs, "train_oracle", side_effect=lambda model, *a, **k: model),
                patch.object(run_diffusion_rbs, "train_denoiser", side_effect=lambda model, *a, **k: model),
                patch.object(run_diffusion_rbs.GaussianDiffusion, "sample", return_value=continuous),
                redirect_stdout(io.StringIO()),
            ):
                run_diffusion_rbs.main()

            exported = pd.read_csv(output / "generated_sequences.csv")
            self.assertEqual(exported["sequence"].tolist(), ["A" * 17] * 2)
            self.assertEqual(exported["oracle_score"].tolist(), [17.0, 17.0])
            configuration = json.loads((output / "run_config.json").read_text())
            self.assertEqual(configuration["seed"], 42)
            self.assertEqual(configuration["sequence_length"], 17)
            self.assertEqual(configuration["timesteps"], 4)


if __name__ == "__main__":
    unittest.main()
