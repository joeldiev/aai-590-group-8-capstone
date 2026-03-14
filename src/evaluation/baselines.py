"""
Baseline models for comparison against the AGL pipeline.

1. Keyword blocklist (regex)
2. TF-IDF + LinearSVM
3. MSP (Maximum Softmax Probability) anomaly baseline
"""

import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline as SkPipeline

from src.config import LABEL2ID, LABEL_NAMES


# ── 1. Keyword Blocklist Baseline ──────────────────────────────────────────

# Patterns per class (case-insensitive)
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?(prior|previous|above)",
    r"forget\s+(all\s+)?(prior|previous|your)\s+instructions",
    r"new\s+instructions?\s*:",
    r"you\s+are\s+now\s+(an?\s+)?unrestricted",
    r"override\s+(safety|content|system)",
    r"do\s+not\s+follow\s+(your|the)\s+(rules|guidelines)",
]

_JAILBREAK_PATTERNS = [
    r"do\s+anything\s+now",
    r"\bDAN\b",
    r"act\s+as\s+(an?\s+)?(evil|unrestricted|unfiltered)",
    r"jailbreak",
    r"pretend\s+(you\s+)?(are|have)\s+no\s+(restrictions|filters|rules)",
    r"developer\s+mode",
    r"bypass\s+(safety|content|filter)",
    r"opposite\s+mode",
    r"no\s+ethical\s+(guidelines|restrictions)",
]

_EXFILTRATION_PATTERNS = [
    r"(show|reveal|display|print|output|repeat)\s+(your\s+)?(system\s+)?prompt",
    r"system\s+(message|instructions|prompt)",
    r"(what|show)\s+(are\s+)?(your|the)\s+(initial\s+)?instructions",
    r"(api|secret)\s+key",
    r"(database|db)\s+connection",
    r"(export|extract|leak)\s+(all\s+)?(user\s+)?data",
    r"(proprietary|confidential|private)\s+(data|information)",
]


def keyword_blocklist_baseline(test_df: pd.DataFrame) -> np.ndarray:
    """Classify prompts using regex keyword matching.

    Args:
        test_df: DataFrame with 'text' column.

    Returns:
        Array of predicted label IDs.
    """
    predictions = []
    for text in test_df["text"]:
        text_lower = str(text).lower()
        pred = _classify_by_keywords(text_lower)
        predictions.append(pred)
    return np.array(predictions)


def _classify_by_keywords(text: str) -> int:
    """Classify a single text using keyword patterns. Priority: Exfil > Jailbreak > Injection > Benign."""
    for pattern in _EXFILTRATION_PATTERNS:
        if re.search(pattern, text):
            return LABEL2ID["Exfiltration"]
    for pattern in _JAILBREAK_PATTERNS:
        if re.search(pattern, text):
            return LABEL2ID["Jailbreak"]
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, text):
            return LABEL2ID["Injection"]
    return LABEL2ID["Benign"]


# ── 2. TF-IDF + LinearSVM Baseline ────────────────────────────────────────

def tfidf_svm_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int = 10000,
    ngram_range: tuple = (1, 3),
) -> tuple[np.ndarray, SkPipeline]:
    """Train TF-IDF + LinearSVM and predict on test set.

    Args:
        train_df: Training data with 'text' and 'label' columns.
        test_df: Test data with 'text' column.

    Returns:
        Tuple of (predictions array, fitted pipeline).
    """
    pipeline = SkPipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            strip_accents="unicode",
            sublinear_tf=True,
        )),
        ("svm", LinearSVC(max_iter=5000, class_weight="balanced")),
    ])

    pipeline.fit(train_df["text"].astype(str), train_df["label"])
    predictions = pipeline.predict(test_df["text"].astype(str))

    return predictions, pipeline


# ── 3. MSP Anomaly Baseline ───────────────────────────────────────────────

def msp_baseline(
    model,
    dataset,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Maximum Softmax Probability anomaly scores.

    Uses 1 - max(softmax(logits)) as anomaly score.
    Higher score = more anomalous.

    Args:
        model: Fine-tuned classifier.
        dataset: HF Dataset with input_ids and attention_mask.
        device: Device for inference.
        batch_size: Batch size.

    Returns:
        (N,) array of anomaly scores.
    """
    import torch
    from torch.utils.data import DataLoader

    model = model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size)
    all_scores = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            max_probs = probs.max(dim=-1).values
            anomaly_scores = 1.0 - max_probs
            all_scores.append(anomaly_scores.cpu().numpy())

    return np.concatenate(all_scores)
