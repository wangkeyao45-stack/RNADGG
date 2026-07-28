"""Sequence encoding and decoding helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from .constants import DNA_ALPHABET


def one_hot_encode(sequence: str, alphabet: str = DNA_ALPHABET) -> np.ndarray:
    """Encode a DNA or RNA sequence as an array with shape `(alphabet, length)`."""
    lookup = {base: i for i, base in enumerate(alphabet)}
    encoded = np.zeros((len(alphabet), len(sequence)), dtype=np.float32)
    for pos, base in enumerate(sequence.upper()):
        if base not in lookup:
            raise ValueError(f"Unsupported base {base!r}; expected one of {alphabet!r}.")
        encoded[lookup[base], pos] = 1.0
    return encoded


def sequences_to_tensor(
    sequences: Iterable[str],
    alphabet: str = DNA_ALPHABET,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert sequences to a float tensor with shape `(n, alphabet, length)`."""
    array = np.stack([one_hot_encode(seq, alphabet) for seq in sequences])
    tensor = torch.tensor(array, dtype=torch.float32)
    return tensor.to(device) if device is not None else tensor


def decode_one_hot(one_hot: np.ndarray | torch.Tensor, alphabet: str = DNA_ALPHABET) -> list[str]:
    """Decode one-hot arrays with shape `(n, alphabet, length)` into strings."""
    if isinstance(one_hot, torch.Tensor):
        one_hot = one_hot.detach().cpu().numpy()
    indices = np.argmax(one_hot, axis=1)
    return ["".join(alphabet[i] for i in row) for row in indices]


def gc_fraction(sequence: str) -> float:
    """Return the GC fraction of a sequence."""
    sequence = sequence.upper()
    if not sequence:
        return 0.0
    return (sequence.count("G") + sequence.count("C")) / len(sequence)
