"""
I/O helpers: save/load models, results, and checkpoints.
"""

import json
from pathlib import Path

import pandas as pd


def save_json(data: dict, path: str | Path) -> None:
    """Save dict as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> dict:
    """Load JSON file to dict."""
    with open(path) as f:
        return json.load(f)


def load_split(split_name: str, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load a dataset split from parquet."""
    from src.config import PROCESSED_DIR
    if data_dir is None:
        data_dir = PROCESSED_DIR
    return pd.read_parquet(Path(data_dir) / f"{split_name}.parquet")
