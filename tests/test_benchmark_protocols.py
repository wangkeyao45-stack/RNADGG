import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from rnadgg.benchmarks import toehold, utr
from rnadgg.benchmarks.models import UNet1D
from rnadgg.sequence import decode_one_hot


class BenchmarkProtocolTests(unittest.TestCase):
    def test_utr_main_defaults_match_manuscript(self):
        with patch.object(utr, "run_experiment", return_value=[]) as run:
            utr.main(["--data", "processed_data.csv", "--dry-run"])
        self.assertEqual([call.args[1].seed for call in run.call_args_list], [42, 43, 44, 123, 456])
        config = run.call_args.args[1]
        self.assertEqual((config.diffusion_steps, config.diffusion_epochs, config.diffusion_batch_size), (500, 80, 32))
        self.assertEqual((config.oracle_epochs, config.predictor_dropout, config.generation_n), (30, 0.3, 2048))
        self.assertEqual(run.call_args.args[2], [1.0])
        self.assertEqual(config.clipping_reference, "train")

    def test_legacy_utr_profile_is_explicit_and_separate(self):
        with patch.object(utr, "run_experiment", return_value=[]) as run:
            utr.main(["--data", "processed_data.csv", "--legacy-utr", "--dry-run", "--seeds", "42"])
        config = run.call_args.args[1]
        self.assertEqual((config.predictor_dropout, config.diffusion_batch_size), (0.0, 64))
        self.assertEqual(config.clipping_reference, "encoded_dataset")
        self.assertEqual(utr.Config().predictor_dropout, 0.3)

    def test_guidance_sweep_has_the_published_levels_and_budget(self):
        with patch.object(utr, "run_experiment", return_value=[]) as run:
            utr.main(["--data", "processed_data.csv", "--profile", "guidance", "--dry-run"])
        run.assert_called_once()
        self.assertEqual(run.call_args.args[2], [0, 0.25, 0.5, 0.75, 1])
        self.assertEqual(run.call_args.args[1].generation_n, 1024)

    def test_toehold_defaults_match_manuscript(self):
        with patch.object(toehold, "run_experiment") as run:
            toehold.main(["--data", "toehold.csv", "--dry-run"])
        self.assertEqual([call.args[1].seed for call in run.call_args_list], [42, 43, 44])
        config = run.call_args.args[1]
        self.assertEqual((config.oracle_epochs, config.diffusion_epochs, config.diffusion_steps), (35, 60, 400))
        self.assertEqual((config.base_channels, config.generation_n, config.guidance_scale), (64, 1024, 0.5))
        self.assertEqual(len(toehold.WEIGHTS), 5)
        self.assertTrue(all(sum(weights) == 1 for weights in toehold.WEIGHTS))

    def test_utr_split_cardinalities_and_disjointness(self):
        splits = utr.split_indices(280000, 42)
        self.assertEqual([len(splits[name]) for name in ["train", "validation", "test"]], [224000, 28000, 28000])
        combined = np.concatenate(list(splits.values()))
        self.assertEqual(len(np.unique(combined)), 280000)

    def test_grouped_split_keeps_duplicates_together(self):
        frame = pd.DataFrame({"full_sequence": [f"group_{i}" for i in range(30)] * 2})
        indices = toehold.grouped_split(frame, 42)
        splits = dict(zip(["train", "validation", "test"], indices))
        self.assertEqual(toehold.overlap_counts(frame.full_sequence.to_numpy(), splits), {
            "train_validation": 0, "train_test": 0, "validation_test": 0,
        })

    def test_test_labels_cannot_change_training_clipping_range(self):
        labels = torch.tensor([1.0, 2.0, 100.0])
        self.assertEqual(utr.clipping_bounds(labels, [0, 1], "train"), (1.0, 2.0))
        self.assertEqual(utr.clipping_bounds(labels, [0, 1], "encoded_dataset"), (1.0, 100.0))

    def test_utr_evaluation_does_not_advance_the_rng(self):
        predictor = utr.ScalarPredictor().eval()
        state = torch.get_rng_state().clone()
        utr.predict(predictor, torch.zeros(3, 4, 50), 2, torch.device("cpu"))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_utr_loader_rejects_the_wrong_benchmark_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            pd.DataFrame({"sequence": ["A" * 49], "r1": [2]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "50"):
                utr.load_data(path, smoke=True)

    def test_manuscript_model_has_a_final_residual_block(self):
        model = UNet1D(8)
        self.assertIsNotNone(model.output_residual)
        self.assertEqual(model.output.in_channels, 8)
        self.assertEqual(model(torch.zeros(2, 4, 50), torch.zeros(2)).shape, (2, 4, 50))


class ScaffoldTests(unittest.TestCase):
    def test_projection_preserves_all_scaffold_constraints(self):
        state = torch.randn(8, 4, 59)
        projected = toehold.project_toehold_scaffold(state)
        torch.testing.assert_close(toehold.project_toehold_scaffold(projected), projected)
        sequences = decode_one_hot(projected)
        self.assertEqual(toehold.validity_metrics(sequences)["all_constraints_fraction"], 1.0)

    def test_stem_gradient_is_propagated_to_the_switch(self):
        state = torch.zeros(1, 4, 59, requires_grad=True)
        projected = toehold.project_toehold_scaffold(state)
        gradient = torch.autograd.grad(projected[:, 0, 41:47].sum(), state)[0]
        self.assertEqual(gradient[:, 3, 24:30].sum().item(), 6)
        self.assertEqual(gradient[:, :, 30:41].abs().sum().item(), 0)

    def test_sampling_is_constraint_valid_for_guided_and_unguided_modes(self):
        class ZeroDenoiser(nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)

        class Predictor(nn.Module):
            def forward(self, x):
                return torch.stack([x[:, 0].sum(1), x[:, 1].sum(1)], dim=1)

        diffusion = toehold.Diffusion(ZeroDenoiser(), 4, torch.device("cpu"))
        for guidance in [0.0, 0.5]:
            sampled = toehold.sample_batch(diffusion, Predictor(), 3, 0.5, 0.5, guidance, torch.device("cpu"))
            self.assertEqual(toehold.validity_metrics(decode_one_hot(sampled))["all_constraints_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
