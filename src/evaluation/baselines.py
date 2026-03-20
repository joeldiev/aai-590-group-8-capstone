"""
Baseline models for comparison against the AGL pipeline.

1. Keyword blocklist (regex) — binary: benign vs malicious
2. TF-IDF + LinearSVM
"""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline as SkPipeline

from src.config import LABEL2ID


# ── 1. Keyword Blocklist Baseline ──────────────────────────────────────────

# Combined malicious patterns (injection, jailbreak, exfiltration)
_MALICIOUS_PATTERNS = [
    # Injection patterns
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?(prior|previous|above)",
    r"forget\s+(all\s+)?(prior|previous|your)\s+instructions",
    r"new\s+instructions?\s*:",
    r"you\s+are\s+now\s+(an?\s+)?unrestricted",
    r"override\s+(safety|content|system)",
    r"do\s+not\s+follow\s+(your|the)\s+(rules|guidelines)",
    # Jailbreak patterns
    r"do\s+anything\s+now",
    r"\bDAN\b",
    r"act\s+as\s+(an?\s+)?(evil|unrestricted|unfiltered)",
    r"jailbreak",
    r"pretend\s+(you\s+)?(are|have)\s+no\s+(restrictions|filters|rules)",
    r"developer\s+mode",
    r"bypass\s+(safety|content|filter)",
    r"opposite\s+mode",
    r"no\s+ethical\s+(guidelines|restrictions)",
    # Exfiltration patterns
    r"(show|reveal|display|print|output|repeat)\s+(your\s+)?(system\s+)?prompt",
    r"system\s+(message|instructions|prompt)",
    r"(what|show)\s+(are\s+)?(your|the)\s+(initial\s+)?instructions",
    r"(api|secret)\s+key",
    r"(database|db)\s+connection",
    r"(export|extract|leak)\s+(all\s+)?(user\s+)?data",
    r"(proprietary|confidential|private)\s+(data|information)",
]


def keyword_blocklist_baseline(test_df: pd.DataFrame) -> np.ndarray:
    """Classify prompts using regex keyword matching (binary).

    Args:
        test_df: DataFrame with 'text' column.

    Returns:
        Array of predicted label IDs (0=benign, 1=malicious).
    """
    predictions = []
    for text in test_df["text"]:
        text_lower = str(text).lower()
        pred = _classify_by_keywords(text_lower)
        predictions.append(pred)
    return np.array(predictions)


def _classify_by_keywords(text: str) -> int:
    """Classify a single text: any malicious pattern match → Malicious, else Benign."""
    for pattern in _MALICIOUS_PATTERNS:
        if re.search(pattern, text):
            return LABEL2ID["Malicious"]
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
