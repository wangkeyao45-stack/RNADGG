from pathlib import Path
import tempfile
import unittest

import pandas as pd

from rnadgg.data import load_sequence_table


class DataTests(unittest.TestCase):
    def test_missing_sequences_and_nonfinite_labels_are_excluded(self):
        frame = pd.DataFrame({
            "sequence": [" acgu ", None, " ", "ACGT", "AAAA", "CCCC", "GGGG"],
            "rl": [1.5, 2, 3, "invalid", float("inf"), -float("inf"), 4],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            frame.to_csv(path, index=False)
            sequences, labels, retained = load_sequence_table(path, label_column="rl")

        self.assertEqual(sequences, ["ACGU", "GGGG"])
        self.assertEqual(labels.tolist(), [1.5, 4.0])
        self.assertEqual(labels.index.tolist(), retained.index.tolist())
        self.assertEqual(retained.index.tolist(), [0, 6])

    def test_empty_valid_dataset_has_a_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            pd.DataFrame({"sequence": [None], "rl": [1]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "No rows"):
                load_sequence_table(path, label_column="rl")


if __name__ == "__main__":
    unittest.main()
