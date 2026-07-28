"""Count sequence rows in one or more CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def count_rows_in_csv(file_path: str | Path) -> int:
    """Return the number of rows in a CSV file, excluding the header."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    return int(pd.read_csv(path).shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Count sequence rows in CSV files.")
    parser.add_argument("csv", nargs="+", type=Path, help="CSV file(s) to inspect.")
    args = parser.parse_args()

    for csv_path in args.csv:
        print(f"{csv_path}: {count_rows_in_csv(csv_path)} rows")


if __name__ == "__main__":
    main()
