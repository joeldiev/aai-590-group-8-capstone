"""
Build the AGL dataset: load CSV, deduplicate, balance, split.

Output: data/processed/{train,val,test}.parquet

Usage:
    from src.data.build_dataset import build_dataset
    build_dataset()
"""

import hashlib
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DIR,
    TRAIN_RATIO,
    SEED,
    LABEL_NAMES,
    LABEL2ID,
    ID2LABEL,
    MAX_SAMPLES_PER_CLASS,
)
from src.data.load_datasets import load_dataset_csv


def build_dataset(csv_path: str | None = None) -> dict[str, pd.DataFrame]:
    """Build the full AGL dataset pipeline.

    Args:
        csv_path: Path to cleaned CSV. If None, uses default from config.

    Returns:
        Dict with keys "train", "val", "test" — each a pd.DataFrame.
    """
    print(f"\n{'='*60}")
    print(f"Building AGL dataset (binary classification)")
    print(f"{'='*60}\n")

    # 1. Load cleaned CSV
    df = load_dataset_csv(csv_path)

    # 2. Add label name column for reporting
    df["label_name"] = df["label"].map(ID2LABEL)

    # 3. Deduplicate on text content
    df = _deduplicate(df)
    print(f"\nAfter dedup: {len(df)} samples")

    # 4. Clean text
    df = _clean_text(df)

    # 5. Balance classes (downsample majority class)
    df = _balance_classes(df)
    print(f"\nAfter balancing: {len(df)} samples")
    print(f"Label distribution:\n{df['label_name'].value_counts().to_string()}")

    # 6. Split: 70/15/15 stratified
    splits = _stratified_split(df)

    # 7. Save to parquet
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in splits.items():
        path = PROCESSED_DIR / f"{split_name}.parquet"
        split_df.to_parquet(path, index=False)
        print(f"Saved {split_name}: {len(split_df)} samples -> {path}")

    # 8. Save metadata
    _save_metadata(splits)

    return splits


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact text duplicates, keeping first occurrence."""
    df["text_hash"] = df["text"].apply(
        lambda t: hashlib.md5(str(t).strip().lower().encode()).hexdigest()
    )
    before = len(df)
    df = df.drop_duplicates(subset=["text_hash"], keep="first")
    df = df.drop(columns=["text_hash"])
    print(f"  Dedup removed {before - len(df)} duplicates")
    return df


def _clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Basic text cleaning."""
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    # Remove empty/very short texts
    df = df[df["text"].str.len() >= 5]
    return df


def _balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Downsample majority classes to cap."""
    cap = MAX_SAMPLES_PER_CLASS
    balanced_frames = []

    for label_val in df["label"].unique():
        subset = df[df["label"] == label_val]
        if len(subset) > cap:
            subset = subset.sample(n=cap, random_state=SEED)
        balanced_frames.append(subset)

    return pd.concat(balanced_frames, ignore_index=True)


def _stratified_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """70/15/15 stratified split."""
    val_test_ratio = 1 - TRAIN_RATIO
    train_df, valtest_df = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df["label"],
        random_state=SEED,
    )

    val_df, test_df = train_test_split(
        valtest_df,
        test_size=0.5,
        stratify=valtest_df["label"],
        random_state=SEED,
    )

    return {"train": train_df, "val": val_df, "test": test_df}


def _save_metadata(splits: dict[str, pd.DataFrame]) -> None:
    """Save dataset statistics as JSON."""
    meta = {}
    for split_name, df in splits.items():
        meta[split_name] = {
            "total": len(df),
            "label_counts": df["label_name"].value_counts().to_dict(),
        }

    path = PROCESSED_DIR / "dataset_metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata -> {path}")
