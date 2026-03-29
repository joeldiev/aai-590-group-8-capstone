"""
Lightweight threat type classifier using TF-IDF + XGBoost.

Classifies malicious prompts into subtypes:
  - injection: attempts to override system instructions
  - jailbreak: attempts to bypass safety/content filters
  - exfiltration: attempts to extract data or system prompts
  - unknown_malicious: doesn't match known patterns

No LLM required. Inference latency: ~1-5ms.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThreatClassification:
    threat_type: str
    threat_type_id: int
    confidence: float
    class_probabilities: dict[str, float]


class ThreatTypeClassifier:
    """TF-IDF + XGBoost threat type classifier."""

    def __init__(
        self,
        vectorizer,
        model,
        label_mapping: dict[str, int],
    ):
        self.vectorizer = vectorizer
        self.model = model
        self.label2id = label_mapping
        self.id2label = {v: k for k, v in label_mapping.items()}

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "ThreatTypeClassifier":
        """Load classifier from saved artifacts.

        Expected files:
            - tfidf_vectorizer.joblib
            - model.joblib
            - label_mapping.json
        """
        artifact_dir = Path(artifact_dir)

        vectorizer = joblib.load(artifact_dir / "tfidf_vectorizer.joblib")
        model = joblib.load(artifact_dir / "model.joblib")

        with open(artifact_dir / "label_mapping.json") as f:
            mapping_data = json.load(f)

        label_mapping = mapping_data.get("label2id", mapping_data)

        logger.info(
            "Loaded ThreatTypeClassifier from %s (%d classes)",
            artifact_dir, len(label_mapping),
        )

        return cls(vectorizer, model, label_mapping)

    def predict(self, text: str) -> ThreatClassification:
        """Classify a single prompt into a threat subtype.

        Args:
            text: The malicious prompt text.

        Returns:
            ThreatClassification with type, confidence, and probabilities.
        """
        features = self.vectorizer.transform([text])

        # Get probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
        else:
            # Fallback for models without predict_proba
            pred = self.model.predict(features)[0]
            probs = np.zeros(len(self.id2label))
            probs[int(pred)] = 1.0

        predicted_id = int(np.argmax(probs))
        confidence = float(probs[predicted_id])

        class_probabilities = {
            self.id2label[i]: float(probs[i])
            for i in range(len(probs))
        }

        return ThreatClassification(
            threat_type=self.id2label[predicted_id],
            threat_type_id=predicted_id,
            confidence=confidence,
            class_probabilities=class_probabilities,
        )

    def predict_batch(self, texts: list[str]) -> list[ThreatClassification]:
        """Classify a batch of prompts."""
        return [self.predict(text) for text in texts]
