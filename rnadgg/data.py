"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


COMMON_SEQUENCE_COLUMNS = ("sequence", "Sequence", "seq", "Seq", "utr", "rbs", "trigger", "搴忓垪")
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

    Returns `(sequences, labels, dataframe)`.
    """
    table = pd.read_csv(path, nrows=nrows)
    seq_col = sequence_column or find_column(table, COMMON_SEQUENCE_COLUMNS, "sequence")
    label_col = label_column or find_column(table, COMMON_LABEL_COLUMNS, "label")
    sequences = table[seq_col].astype(str).str.upper().tolist()
    labels = pd.to_numeric(table[label_col], errors="coerce")
    valid = labels.notna()
    return [seq for seq, keep in zip(sequences, valid) if keep], labels[valid].astype(float), table[valid]
