from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from rnadgg.benchmarks import toehold, utr
from rnadgg.sequence import decode_one_hot, sequences_to_tensor


class BenchmarkWorkflowTests(unittest.TestCase):
    def test_utr_smoke_trains_samples_and_scores_both_generators(self):
        rng = np.random.default_rng(11)
        sequences = ["".join(rng.choice(list("ACGT"), size=50)) for _ in range(80)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "utr.csv"
            output = root / "result"
            pd.DataFrame({"sequence": sequences, "r1": rng.uniform(0, 12, 80)}).to_csv(data, index=False)
            with redirect_stdout(io.StringIO()):
                utr.main([
                    "--data", str(data), "--output", str(output), "--smoke",
                    "--seeds", "42", "--device", "cpu", "--guidance-levels", "0", "0.5",
                ])
            run = output / "seed_42"
            config = json.loads((run / "run_config.json").read_text())
            self.assertTrue(config["validation_only"])
            self.assertEqual(config["split_rows"], {"train": 64, "validation": 8, "test": 8})
            self.assertTrue((run / "gan_generator.pt").is_file())
            predictor = utr.ScalarPredictor().eval()
            predictor.load_state_dict(torch.load(run / "predictor.pt", map_location="cpu", weights_only=True))
            for name in ["diffusion_g0", "diffusion_g0.5", "rnn_gan"]:
                generated = pd.read_csv(run / f"generated_{name}.csv")
                self.assertEqual(len(generated), 8)
                with torch.no_grad():
                    raw = predictor(sequences_to_tensor(generated.sequence)).squeeze(-1).numpy()
                np.testing.assert_allclose(generated.predicted_activity_raw, raw, atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(
                    generated.predicted_activity_clipped,
                    np.clip(raw, *config["training_label_ranges"]), atol=1e-6, rtol=1e-6,
                )

    def test_toehold_smoke_exports_all_weight_settings_and_constraints(self):
        generator = torch.Generator().manual_seed(11)
        projected = toehold.project_toehold_scaffold(torch.randn(80, 4, 59, generator=generator))
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "toehold.csv"
            output = root / "result"
            pd.DataFrame({
                "full_sequence": decode_one_hot(projected),
                "ON": rng.uniform(0, 1, 80), "OFF": rng.uniform(0, 1, 80),
            }).to_csv(data, index=False)
            with (
                patch.object(toehold, "tqdm", side_effect=lambda iterable, **kwargs: iterable),
                redirect_stdout(io.StringIO()),
            ):
                toehold.main([
                    "--data", str(data), "--output", str(output),
                    "--seeds", "42", "--device", "cpu", "--smoke",
                ])
            run = output / "seed_42"
            config = json.loads((run / "run_config.json").read_text())
            self.assertTrue(config["validation_only"])
            self.assertTrue(all(value == 0 for value in config["cross_split_exact_sequence_overlap"].values()))
            results = pd.read_csv(run / "toehold_grouped_constrained_results.csv")
            self.assertEqual(len(results), 6)
            self.assertTrue(results["n"].eq(8).all())
            self.assertTrue(results["all_constraints_fraction"].eq(1).all())
            self.assertTrue((run / "toehold_diffusion.pt").is_file())


if __name__ == "__main__":
    unittest.main()
