"""
Build the unified AGL dataset: merge sources, deduplicate, balance, split.

Output: data/processed/{train,val,test}.parquet

Usage:
    from src.data.build_dataset import build_dataset
    build_dataset(phase="mvp")   # Phase A: deepset + jailbreakv28k only
    build_dataset(phase="full")  # Phase B: all sources + synthetic exfiltration
"""

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    SEED,
    LABEL_NAMES,
    LABEL2ID,
)
from src.data.load_datasets import MVP_LOADERS, ALL_LOADERS
from src.data.label_mapping import unify_labels
from src.data.synthetic_exfiltration import load_synthetic_exfiltration


def build_dataset(phase: str = "mvp") -> dict[str, pd.DataFrame]:
    """Build the full AGL dataset pipeline.

    Args:
        phase: "mvp" (3-class, Phase A) or "full" (4-class, Phase B).

    Returns:
        Dict with keys "train", "val", "test" — each a pd.DataFrame.
    """
    print(f"\n{'='*60}")
    print(f"Building AGL dataset — phase={phase}")
    print(f"{'='*60}\n")

    # 1. Load raw data from all sources
    loaders = MVP_LOADERS if phase == "mvp" else ALL_LOADERS
    frames = []
    for name, loader_fn in loaders.items():
        print(f"Loading {name}...")
        try:
            df = loader_fn()
            print(f"  → {len(df)} samples")
            frames.append(df)
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # Add synthetic exfiltration for full phase
    if phase == "full":
        print("Loading synthetic exfiltration...")
        try:
            synth = load_synthetic_exfiltration()
            frames.append(synth)
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # 2. Merge
    if not frames:
        raise RuntimeError(
            "No datasets loaded successfully. Check network access to HuggingFace Hub, "
            "or run on Colab/Kaggle where HF is accessible."
        )
    raw = pd.concat(frames, ignore_index=True)
    print(f"\nMerged: {len(raw)} total samples")
    print(f"Sources: {raw['source'].value_counts().to_dict()}")

    # 3. Unify labels
    df = unify_labels(raw)
    print(f"\nAfter label mapping: {len(df)} samples")
    print(f"Label distribution:\n{df['unified_label'].value_counts().to_string()}")

    # 4. Deduplicate on text content
    df = _deduplicate(df)
    print(f"\nAfter dedup: {len(df)} samples")

    # 5. Clean text
    df = _clean_text(df)

    # 6. Balance classes (downsample majority classes)
    df = _balance_classes(df, phase=phase)
    print(f"\nAfter balancing: {len(df)} samples")
    print(f"Label distribution:\n{df['unified_label'].value_counts().to_string()}")

    # 7. Split: 70/15/15 stratified
    splits = _stratified_split(df)

    # 8. Save to parquet
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in splits.items():
        path = PROCESSED_DIR / f"{split_name}.parquet"
        split_df.to_parquet(path, index=False)
        print(f"Saved {split_name}: {len(split_df)} samples → {path}")

    # 9. Save metadata
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


def _balance_classes(
    df: pd.DataFrame,
    phase: str = "mvp",
) -> pd.DataFrame:
    """Downsample majority classes to target sizes.

    MVP (3-class): cap each class at 7,000
    Full (4-class): Benign/Injection/Jailbreak at 7,000; Exfiltration uncapped
    """
    cap = 7000
    balanced_frames = []

    for label_name in df["unified_label"].unique():
        subset = df[df["unified_label"] == label_name]
        if len(subset) > cap and label_name != "Exfiltration":
            subset = subset.sample(n=cap, random_state=SEED)
        balanced_frames.append(subset)

    return pd.concat(balanced_frames, ignore_index=True)


def _stratified_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """70/15/15 stratified split."""
    # First split: train vs (val+test)
    val_test_ratio = 1 - TRAIN_RATIO
    train_df, valtest_df = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df["label"],
        random_state=SEED,
    )

    # Second split: val vs test (50/50 of remaining)
    val_df, test_df = train_test_split(
        valtest_df,
        test_size=0.5,
        stratify=valtest_df["label"],
        random_state=SEED,
    )

    return {"train": train_df, "val": val_df, "test": test_df}


def _save_metadata(splits: dict[str, pd.DataFrame]) -> None:
    """Save dataset statistics as JSON."""
    import json

    meta = {}
    for split_name, df in splits.items():
        meta[split_name] = {
            "total": len(df),
            "label_counts": df["unified_label"].value_counts().to_dict(),
            "sources": df["source"].value_counts().to_dict(),
        }

    path = PROCESSED_DIR / "dataset_metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata → {path}")
