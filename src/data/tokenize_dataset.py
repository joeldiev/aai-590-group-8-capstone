"""
Tokenize dataset for RoBERTa training.

Converts parquet DataFrames → HuggingFace Dataset objects with
input_ids, attention_mask, and labels.
"""

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from src.config import MODEL_NAME, MAX_SEQ_LENGTH, PROCESSED_DIR


def tokenize_for_roberta(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer | None = None,
    max_length: int = MAX_SEQ_LENGTH,
) -> Dataset:
    """Tokenize a DataFrame for RoBERTa fine-tuning.

    Args:
        df: Must have columns [text, label].
        tokenizer: If None, loads the default tokenizer.
        max_length: Max sequence length for tokenization.

    Returns:
        HuggingFace Dataset with input_ids, attention_mask, labels.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    ds = ds.rename_column("label", "labels")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    return ds


def load_tokenized_splits(
    tokenizer: AutoTokenizer | None = None,
    max_length: int = MAX_SEQ_LENGTH,
) -> dict[str, Dataset]:
    """Load all splits from parquet and tokenize them.

    Returns:
        Dict with keys "train", "val", "test" — each a HF Dataset.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    splits = {}
    for split_name in ["train", "val", "test"]:
        path = PROCESSED_DIR / f"{split_name}.parquet"
        df = pd.read_parquet(path)
        splits[split_name] = tokenize_for_roberta(df, tokenizer, max_length)
        print(f"Tokenized {split_name}: {len(splits[split_name])} samples")

    return splits
