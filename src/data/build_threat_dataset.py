"""
Build a multi-class threat-type dataset for the severity classifier.

Re-labels existing binary training data back into threat subtypes using
source metadata + heuristics, then merges with L1B3RT4S jailbreak prompts.

Produces a balanced dataset with classes:
  0 = injection
  1 = jailbreak
  2 = exfiltration
  3 = unknown_malicious

Usage:
    python -m src.data.build_threat_dataset \
        --binary-csv data/processed/dataset_cleaned.csv \
        --l1b3rt4s-csv data/processed/l1b3rt4s_prompts.csv \
        --output-dir data/processed/threat_type/
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


THREAT_LABELS = {
    "injection": 0,
    "jailbreak": 1,
    "exfiltration": 2,
    "unknown_malicious": 3,
}

# Keyword heuristics for re-labeling malicious samples that lack source metadata
EXFILTRATION_KEYWORDS = [
    "system prompt", "reveal your instructions", "output all instructions",
    "what are your rules", "show me your prompt", "repeat back",
    "confidential", "send the conversation", "exfiltrate", "leak",
    "extract.*data", "show.*hidden", "display.*system", "print.*instructions",
    "what were you told", "original prompt", "initial instructions",
    "developer mode", "show configuration", "export.*data",
]

INJECTION_KEYWORDS = [
    "ignore previous", "disregard", "new instructions", "override",
    "forget everything", "you are now", "act as", "pretend you",
    "ignore all prior", "system:", "\\[INST\\]", "ignore the above",
    "do not follow", "instead of", "new role", "from now on you",
    "your new task", "cancel previous", "reset your",
]

JAILBREAK_KEYWORDS = [
    "jailbreak", "godmode", "dan ", "do anything now", "evil mode",
    "no restrictions", "unfiltered", "uncensored", "bypass",
    "safety off", "no limits", "liberated", "free from",
    "rebel", "l33t", "leetspeak", "pliny", "libertas",
    "fuck yeah", "unhinge", "anarchy", "🐉",
]


def classify_threat_type(text: str, source: str = "") -> str:
    """Classify a malicious prompt into a threat subtype using heuristics.

    Args:
        text: The prompt text.
        source: Original data source name (if available).

    Returns:
        One of: injection, jailbreak, exfiltration, unknown_malicious
    """
    text_lower = text.lower()

    # Score each category
    scores = {"injection": 0, "jailbreak": 0, "exfiltration": 0}

    import re
    for kw in EXFILTRATION_KEYWORDS:
        if re.search(kw, text_lower):
            scores["exfiltration"] += 1

    for kw in INJECTION_KEYWORDS:
        if re.search(kw, text_lower):
            scores["injection"] += 1

    for kw in JAILBREAK_KEYWORDS:
        if re.search(kw, text_lower):
            scores["jailbreak"] += 1

    # Source-based boosting
    source_lower = source.lower() if source else ""
    if "jailbreak" in source_lower or "l1b3rt4s" in source_lower:
        scores["jailbreak"] += 3
    elif "hackaprompt" in source_lower:
        scores["injection"] += 3
    elif "exfil" in source_lower or "synthetic" in source_lower:
        scores["exfiltration"] += 3

    max_score = max(scores.values())
    if max_score == 0:
        return "unknown_malicious"

    # Return the highest scoring category
    return max(scores, key=scores.get)


def build_threat_dataset(
    binary_csv: str | Path,
    l1b3rt4s_csv: str | Path | None = None,
    output_dir: str | Path = "data/processed/threat_type",
    max_per_class: int = 5000,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Build the multi-class threat type dataset.

    Args:
        binary_csv: Path to the binary-labeled dataset CSV.
        l1b3rt4s_csv: Path to the parsed L1B3RT4S prompts CSV.
        output_dir: Where to save train/val/test parquet files.
        max_per_class: Maximum samples per class for balancing.
        test_size: Fraction for test split.
        val_size: Fraction for validation split.
        seed: Random seed.

    Returns:
        Dict of {split_name: DataFrame}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading binary dataset...")
    df = pd.read_csv(binary_csv)

    # Ensure required columns
    if "text" not in df.columns:
        # Try common alternatives
        for col in ["prompt", "content", "input"]:
            if col in df.columns:
                df = df.rename(columns={col: "text"})
                break

    if "label" not in df.columns:
        raise ValueError("CSV must have a 'label' column")

    # Filter to malicious only (label == 1 or label == "malicious"/"Malicious")
    malicious_mask = df["label"].apply(
        lambda x: x == 1 or str(x).strip().lower() in ("malicious", "1")
    )
    mal_df = df[malicious_mask].copy()
    print(f"  Malicious samples from binary dataset: {len(mal_df)}")

    # Source column (may not exist)
    if "source" not in mal_df.columns:
        mal_df["source"] = "binary_dataset"

    # Classify threat types
    mal_df["threat_type"] = mal_df.apply(
        lambda row: classify_threat_type(row["text"], row.get("source", "")),
        axis=1,
    )

    # Add L1B3RT4S data
    if l1b3rt4s_csv and Path(l1b3rt4s_csv).exists():
        print("Loading L1B3RT4S prompts...")
        l1b3rt4s_df = pd.read_csv(l1b3rt4s_csv)
        l1b3rt4s_df = l1b3rt4s_df[["text", "threat_type", "source"]].copy()
        mal_df = pd.concat(
            [mal_df[["text", "threat_type", "source"]], l1b3rt4s_df],
            ignore_index=True,
        )
        print(f"  After adding L1B3RT4S: {len(mal_df)} total samples")

    # Map threat types to numeric labels
    mal_df["threat_label"] = mal_df["threat_type"].map(THREAT_LABELS)

    # Drop any unmapped
    mal_df = mal_df.dropna(subset=["threat_label"])
    mal_df["threat_label"] = mal_df["threat_label"].astype(int)

    # Class distribution
    print("\nClass distribution (before balancing):")
    for threat_type, count in mal_df["threat_type"].value_counts().items():
        print(f"  {threat_type}: {count}")

    # Balance classes
    balanced_dfs = []
    for threat_type in THREAT_LABELS:
        class_df = mal_df[mal_df["threat_type"] == threat_type]
        if len(class_df) > max_per_class:
            class_df = class_df.sample(n=max_per_class, random_state=seed)
        balanced_dfs.append(class_df)

    balanced = pd.concat(balanced_dfs, ignore_index=True)

    print(f"\nClass distribution (after balancing, cap={max_per_class}):")
    for threat_type, count in balanced["threat_type"].value_counts().items():
        print(f"  {threat_type}: {count}")

    # Deduplicate
    balanced = balanced.drop_duplicates(subset=["text"])
    print(f"After dedup: {len(balanced)} samples")

    # Split
    train_df, temp_df = train_test_split(
        balanced, test_size=(test_size + val_size), random_state=seed,
        stratify=balanced["threat_label"],
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=test_size / (test_size + val_size), random_state=seed,
        stratify=temp_df["threat_label"],
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}

    for name, split_df in splits.items():
        path = output_dir / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        print(f"  {name}: {len(split_df)} samples → {path}")

    # Save label mapping
    import json
    mapping_path = output_dir / "threat_label_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump({
            "label2id": THREAT_LABELS,
            "id2label": {v: k for k, v in THREAT_LABELS.items()},
        }, f, indent=2)
    print(f"  Label mapping → {mapping_path}")

    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Build multi-class threat type dataset"
    )
    parser.add_argument("--binary-csv", type=str, required=True)
    parser.add_argument("--l1b3rt4s-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="data/processed/threat_type")
    parser.add_argument("--max-per-class", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_threat_dataset(
        binary_csv=args.binary_csv,
        l1b3rt4s_csv=args.l1b3rt4s_csv,
        output_dir=args.output_dir,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
