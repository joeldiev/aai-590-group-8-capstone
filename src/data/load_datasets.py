"""
Dataset loader — reads the cleaned CSV produced by the data pipeline notebooks.

The CSV is expected to have columns: text, label (0=benign, 1=malicious).
It is produced by notebooks/data_pipeline/1_data_cleaning.ipynb.

Usage:
    from src.data.load_datasets import load_dataset_csv
    df = load_dataset_csv()
"""

import pandas as pd

from src.config import DATASET_CSV


def load_dataset_csv(path: str | None = None) -> pd.DataFrame:
    """Load the cleaned binary dataset from CSV.

    Args:
        path: Path to CSV file. If None, uses DATASET_CSV from config.

    Returns:
        pd.DataFrame with columns [text, label].
    """
    csv_path = path or DATASET_CSV

    from pathlib import Path
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at {csv_path}\n"
            f"Run notebooks/data_pipeline/1_data_cleaning.ipynb first to produce it,\n"
            f"or pass --csv to point to a different file."
        )

    df = pd.read_csv(csv_path)

    # Validate expected columns
    if "text" not in df.columns:
        # Try common alternative column names
        text_cols = [c for c in df.columns if c.lower() in ("text", "prompt", "content")]
        if text_cols:
            df = df.rename(columns={text_cols[0]: "text"})
        else:
            raise ValueError(
                f"CSV must have a 'text' column. Found: {list(df.columns)}"
            )

    if "label" not in df.columns:
        raise ValueError(
            f"CSV must have a 'label' column (0=benign, 1=malicious). Found: {list(df.columns)}"
        )

    # Keep only what we need
    df = df[["text", "label"]].copy()

    # Ensure label is int
    df["label"] = df["label"].astype(int)

    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    return df
