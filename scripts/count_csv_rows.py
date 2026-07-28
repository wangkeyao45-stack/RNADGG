"""Count rows in CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Count rows in one or more CSV files.")
    parser.add_argument("csv", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.csv:
        print(f"{path}: {pd.read_csv(path).shape[0]} rows")


if __name__ == "__main__":
    main()
