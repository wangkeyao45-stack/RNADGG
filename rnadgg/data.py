"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


COMMON_SEQUENCE_COLUMNS = ("sequence", "Sequence", "seq", "Seq", "utr", "rbs", "trigger", "\u5e8f\u5217")
COMMON_LABEL_COLUMNS = ("label", "score", "rl", "MRL", "mrl", "ON", "OFF", "on", "off")


def find_column(frame: pd.DataFrame, candidates: tuple[str, ...], role: str) -> str:
    """Find the first candidate column present in a DataFrame."""
    for name in candidates:
        if name in frame.columns:
            return name
    raise ValueError(f"Could not infer {role} column from columns: {list(frame.columns)}")


def load_sequence_table(
    path: str | Path,
    sequence_column: str | None = None,
    label_column: str | None = None,
    nrows: int | None = None,
) -> tuple[list[str], pd.Series, pd.DataFrame]:
    """Load a sequence-function CSV table.

    Returns `(sequences, labels, dataframe)`. Rows without a sequence or a
    finite numeric label are excluded.
    """
    table = pd.read_csv(path, nrows=nrows)
    seq_col = sequence_column or find_column(table, COMMON_SEQUENCE_COLUMNS, "sequence")
    label_col = label_column or find_column(table, COMMON_LABEL_COLUMNS, "label")
    sequences = table[seq_col].astype("string").str.strip().str.upper()
    labels = pd.to_numeric(table[label_col], errors="coerce")
    valid = sequences.notna() & sequences.ne("") & np.isfinite(labels)
    if not valid.any():
        raise ValueError("No rows with a non-empty sequence and a finite numeric label.")
    return sequences[valid].tolist(), labels[valid].astype(float), table[valid]
