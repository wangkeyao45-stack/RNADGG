"""Count data rows in a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def count_data_rows(file_path: str | Path) -> int:
    """Return the number of data rows in a CSV file, excluding the header."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    return int(pd.read_csv(path).shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Count data rows in a CSV file.")
    parser.add_argument("csv", type=Path, help="Path to the CSV file.")
    args = parser.parse_args()

    n_rows = count_data_rows(args.csv)
    print(f"{args.csv}: {n_rows} rows")


if __name__ == "__main__":
    main()
