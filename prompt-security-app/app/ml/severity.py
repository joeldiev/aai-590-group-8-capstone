"""
Severity evaluation service — orchestrates threat classification,
severity scoring, and threat intelligence lookup.

Activated ONLY when both classifiers agree a prompt is malicious.
No LLM involved.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.ml.threat_classifier import ThreatTypeClassifier, ThreatClassification
from app.ml.threat_intel import ThreatIntelService
from app.schemas.prediction import PredictionResponse
from app.schemas.classification import ClassificationResponse

logger = logging.getLogger(__name__)


# Severity tiers
SEVERITY_TIERS = {
    "critical": {"min_score": 8, "color": "#dc2626"},
    "high": {"min_score": 5, "color": "#ea580c"},
    "medium": {"min_score": 3, "color": "#ca8a04"},
    "low": {"min_score": 0, "color": "#65a30d"},
}

# Threat type risk weights
THREAT_TYPE_WEIGHTS = {
    "exfiltration": 2,
    "injection": 2,
    "jailbreak": 1,
    "unknown_malicious": 1,
}


@dataclass(frozen=True)
class SeverityResult:
    severity_tier: str
    severity_score: int
    threat_type: str
    threat_confidence: float
    threat_class_probabilities: dict[str, float]
    threat_intel: dict
    scoring_breakdown: dict[str, int]
    latency_ms: float


class SeverityService:
    """Orchestrates severity evaluation for malicious prompts."""

    def __init__(
        self,
        threat_classifier: ThreatTypeClassifier | None = None,
        threat_intel: ThreatIntelService | None = None,
    ):
        self.threat_classifier = threat_classifier
        self.threat_intel = threat_intel
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(
        self,
        threat_classifier_dir: str | Path | None = None,
        threat_intel_cache_path: str | Path | None = None,
    ) -> None:
        """Load all severity sub-components.

        Args:
            threat_classifier_dir: Path to TF-IDF + XGBoost artifacts.
            threat_intel_cache_path: Path to threat_mapping.json.
        """
        # Load threat type classifier
        if threat_classifier_dir and Path(threat_classifier_dir).exists():
            try:
                self.threat_classifier = ThreatTypeClassifier.load(threat_classifier_dir)
                logger.info("Threat type classifier loaded")
            except Exception as e:
                logger.warning("Failed to load threat classifier: %s", e)
                self.threat_classifier = None
        else:
            logger.warning(
                "Threat classifier dir not found: %s. "
                "Severity will use heuristic fallback.",
                threat_classifier_dir,
            )

        # Load threat intel
        self.threat_intel = ThreatIntelService(
            cache_path=threat_intel_cache_path,
        )
        self.threat_intel.load()

        self._is_loaded = True
        logger.info("SeverityService loaded successfully")

    def evaluate(
        self,
        text: str,
        anomaly: PredictionResponse,
        classification: ClassificationResponse,
    ) -> SeverityResult:
        """Evaluate severity of a malicious prompt.

        Should only be called when both classifiers flag malicious.

        Args:
            text: The original prompt text.
            anomaly: Anomaly detection result.
            classification: RoBERTa classification result.

        Returns:
            SeverityResult with tier, score, threat type, and intel.
        """
        start = time.perf_counter()

        # Step 1: Classify threat type
        threat_class = self._classify_threat(text)

        # Step 2: Score severity
        score, breakdown = self._score_severity(
            anomaly=anomaly,
            classification=classification,
            threat_class=threat_class,
            text=text,
        )

        # Step 3: Map to tier
        tier = self._score_to_tier(score)

        # Step 4: Look up threat intelligence
        intel = {}
        if self.threat_intel:
            intel = self.threat_intel.lookup(threat_class.threat_type)

        latency_ms = (time.perf_counter() - start) * 1000

        return SeverityResult(
            severity_tier=tier,
            severity_score=score,
            threat_type=threat_class.threat_type,
            threat_confidence=threat_class.confidence,
            threat_class_probabilities=threat_class.class_probabilities,
            threat_intel=intel,
            scoring_breakdown=breakdown,
            latency_ms=latency_ms,
        )

    def _classify_threat(self, text: str) -> ThreatClassification:
        """Classify threat type using model or heuristic fallback."""
        if self.threat_classifier:
            return self.threat_classifier.predict(text)

        # Heuristic fallback when model is not available
        return self._heuristic_threat_classify(text)

    def _heuristic_threat_classify(self, text: str) -> ThreatClassification:
        """Rule-based threat classification fallback."""
        import re

        text_lower = text.lower()
        scores = {"injection": 0, "jailbreak": 0, "exfiltration": 0, "unknown_malicious": 0}

        # Exfiltration indicators
        exfil_patterns = [
            r"system prompt", r"reveal.*instructions", r"output.*instructions",
            r"show.*prompt", r"repeat back", r"confidential",
            r"exfiltrat", r"extract.*data", r"print.*instructions",
            r"original prompt", r"initial instructions",
        ]
        for p in exfil_patterns:
            if re.search(p, text_lower):
                scores["exfiltration"] += 1

        # Injection indicators
        inject_patterns = [
            r"ignore previous", r"disregard", r"new instructions",
            r"override", r"forget everything", r"you are now",
            r"ignore.*prior", r"ignore the above", r"your new task",
        ]
        for p in inject_patterns:
            if re.search(p, text_lower):
                scores["injection"] += 1

        # Jailbreak indicators
        jailbreak_patterns = [
            r"jailbreak", r"godmode", r"\bdan\b", r"do anything now",
            r"no restrictions", r"unfiltered", r"uncensored", r"bypass",
            r"liberat", r"l33t", r"pliny", r"rebel",
        ]
        for p in jailbreak_patterns:
            if re.search(p, text_lower):
                scores["jailbreak"] += 1

        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]

        if max_score == 0:
            max_type = "unknown_malicious"

        total = sum(scores.values()) or 1
        probs = {k: v / total for k, v in scores.items()}

        return ThreatClassification(
            threat_type=max_type,
            threat_type_id={"injection": 0, "jailbreak": 1, "exfiltration": 2, "unknown_malicious": 3}[max_type],
            confidence=max_score / total if total > 0 else 0.25,
            class_probabilities=probs,
        )

    def _score_severity(
        self,
        anomaly: PredictionResponse,
        classification: ClassificationResponse,
        threat_class: ThreatClassification,
        text: str,
    ) -> tuple[int, dict[str, int]]:
        """Compute severity score using weighted signals.

        Returns:
            Tuple of (total_score, breakdown_dict).
        """
        breakdown = {}

        # 1. Classifier confidence (0-3 points)
        conf = classification.confidence
        if conf >= 0.95:
            breakdown["classifier_confidence"] = 3
        elif conf >= 0.85:
            breakdown["classifier_confidence"] = 2
        elif conf >= 0.70:
            breakdown["classifier_confidence"] = 1
        else:
            breakdown["classifier_confidence"] = 0

        # 2. Anomaly margin (0-3 points)
        margin = anomaly.anomaly_score - anomaly.threshold
        if margin >= 0.50:
            breakdown["anomaly_margin"] = 3
        elif margin >= 0.25:
            breakdown["anomaly_margin"] = 2
        elif margin >= 0.10:
            breakdown["anomaly_margin"] = 1
        else:
            breakdown["anomaly_margin"] = 0

        # 3. Threat type weight (0-2 points)
        breakdown["threat_type_weight"] = THREAT_TYPE_WEIGHTS.get(
            threat_class.threat_type, 1
        )

        # 4. Text heuristics (0-2 points)
        text_score = 0
        import re
        # URL presence
        if re.search(r"https?://|www\.", text, re.IGNORECASE):
            text_score += 1
        # Base64 or encoding patterns
        if re.search(r"[A-Za-z0-9+/]{20,}={0,2}", text):
            text_score += 1
        breakdown["text_heuristics"] = text_score

        total = sum(breakdown.values())
        return total, breakdown

    def _score_to_tier(self, score: int) -> str:
        """Map numeric score to severity tier."""
        if score >= 8:
            return "critical"
        elif score >= 5:
            return "high"
        elif score >= 3:
            return "medium"
        else:
            return "low"
