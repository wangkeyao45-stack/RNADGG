"""Inspect a sequence-function CSV table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rnadgg.data import load_sequence_table
from rnadgg.sequence import gc_fraction


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a regulatory RNA dataset.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--sequence-column")
    parser.add_argument("--label-column")
    parser.add_argument("--nrows", type=int)
    args = parser.parse_args()

    sequences, labels, _ = load_sequence_table(
        args.csv,
        sequence_column=args.sequence_column,
        label_column=args.label_column,
        nrows=args.nrows,
    )
    lengths = [len(seq) for seq in sequences]
    gc_values = [gc_fraction(seq) for seq in sequences]

    print(f"Rows: {len(sequences)}")
    print(f"Sequence length range: {min(lengths)}-{max(lengths)}")
    print(f"Mean GC fraction: {sum(gc_values) / len(gc_values):.3f}")
    print(f"Label mean: {labels.mean():.4f}")


if __name__ == "__main__":
    main()
