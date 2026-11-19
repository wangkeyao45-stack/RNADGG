import unittest

import numpy as np

from rnadgg.constants import RNA_ALPHABET
from rnadgg.sequence import decode_one_hot, one_hot_encode, sequences_to_tensor


class SequenceTests(unittest.TestCase):
    def test_rna_input_uses_the_implementation_t_channel(self):
        np.testing.assert_array_equal(one_hot_encode("acgu"), one_hot_encode("ACGT"))

    def test_rna_alphabet_accepts_dna_input(self):
        encoded = sequences_to_tensor(["ACGT"], alphabet=RNA_ALPHABET)
        self.assertEqual(decode_one_hot(encoded, alphabet=RNA_ALPHABET), ["ACGU"])

    def test_one_hot_has_one_nucleotide_per_position(self):
        encoded = one_hot_encode("AAACGU")
        np.testing.assert_array_equal(encoded.sum(axis=0), np.ones(6))
        self.assertEqual(float(encoded.mean()), 0.25)

    def test_ambiguous_bases_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported base"):
            one_hot_encode("ACGN")


if __name__ == "__main__":
    unittest.main()
