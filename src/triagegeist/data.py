"""Dataset loading helpers."""

from __future__ import annotations

from zipfile import ZipFile

import pandas as pd

from .config import DATA_DIR, ID_COLUMN, TEXT_COLUMN


def load_raw_tables() -> dict[str, pd.DataFrame]:
    return {
        "train": pd.read_csv(DATA_DIR / "train.csv"),
        "test": pd.read_csv(DATA_DIR / "test.csv"),
        "chief_complaints": pd.read_csv(DATA_DIR / "chief_complaints.csv"),
        "patient_history": pd.read_csv(DATA_DIR / "patient_history.csv"),
        "sample_submission": pd.read_csv(DATA_DIR / "sample_submission.csv"),
    }


def load_merged(split: str) -> pd.DataFrame:
    tables = load_raw_tables()
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    merged = tables[split].merge(
        tables["chief_complaints"][[ID_COLUMN, TEXT_COLUMN]],
        on=ID_COLUMN,
        how="left",
    ).merge(
        tables["patient_history"],
        on=ID_COLUMN,
        how="left",
    )
    return merged
