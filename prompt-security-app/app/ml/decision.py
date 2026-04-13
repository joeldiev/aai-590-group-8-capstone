"""
Rule-based decision module for AGL prompt risk assessment.

Aggregates signals from the anomaly detector (autoencoder reconstruction error)
and the RoBERTa classifier into a single binary verdict (benign vs malicious)
using a cumulative scoring system.  The score thresholds and weights are tuned
to minimise false negatives — in a security context, missing a real threat is
far more costly than a false alarm.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.classification import ClassificationResponse
from app.schemas.prediction import PredictionResponse


# Label substrings used to interpret the classifier's predicted_label string.
MALICIOUS_LABEL_HINTS = ("malicious", "attack", "inject", "unsafe")
BENIGN_LABEL_HINTS = ("benign", "safe", "normal")


@dataclass(frozen=True)
class DecisionResult:
    final_label: str
    is_malicious: bool
    reasons: list[str]


def decide_prompt_risk(
    anomaly: PredictionResponse,
    classification: ClassificationResponse,
) -> DecisionResult:
    """Combine anomaly and classifier outputs into a final risk decision.

    A cumulative ``malicious_score`` is built from four signal groups:
      1. Anomaly flag (+2) and high margin (+1)
      2. Classifier malicious with confidence (+3) or without (+1)
      3. Classifier benign with confidence (-2)
      4. Cross-signal agreement (+1)

    A score >= 2 yields a "malicious" verdict.

    Args:
        anomaly: Result from the autoencoder anomaly detector.
        classification: Result from the RoBERTa classifier.

    Returns:
        DecisionResult with the final label, boolean flag, and reasoning trace.
    """
    reasons: list[str] = []
    malicious_score = 0

    normalized_label = classification.predicted_label.strip().lower()
    classifier_is_malicious = any(hint in normalized_label for hint in MALICIOUS_LABEL_HINTS)
    classifier_is_benign = any(hint in normalized_label for hint in BENIGN_LABEL_HINTS)
    classifier_confident = classification.confidence >= classification.min_confidence
    anomaly_margin = anomaly.anomaly_score - anomaly.threshold

    if anomaly.is_anomalous:
        malicious_score += 2
        reasons.append(
            f"anomaly_score_above_threshold ({anomaly.anomaly_score:.4f} >= {anomaly.threshold:.4f})"
        )
    else:
        reasons.append(
            f"anomaly_score_below_threshold ({anomaly.anomaly_score:.4f} < {anomaly.threshold:.4f})"
        )

    if anomaly_margin >= 0.25:
        malicious_score += 1
        reasons.append(f"anomaly_margin_high ({anomaly_margin:.4f})")

    if classifier_is_malicious and classifier_confident and not classification.is_uncertain:
        malicious_score += 3
        reasons.append(
            "classifier_predicted_malicious_with_confidence "
            f"({classification.predicted_label}, confidence={classification.confidence:.4f})"
        )
    elif classifier_is_malicious:
        malicious_score += 1
        reasons.append(
            "classifier_predicted_malicious_but_low_confidence "
            f"({classification.predicted_label}, confidence={classification.confidence:.4f})"
        )

    if classifier_is_benign and classifier_confident and not classification.is_uncertain:
        malicious_score -= 2
        reasons.append(
            "classifier_predicted_benign_with_confidence "
            f"({classification.predicted_label}, confidence={classification.confidence:.4f})"
        )
    elif classification.is_uncertain:
        reasons.append(
            "classifier_uncertain "
            f"(confidence={classification.confidence:.4f}, min_confidence={classification.min_confidence:.4f})"
        )

    if anomaly.is_anomalous and (classification.is_uncertain or classifier_is_malicious):
        malicious_score += 1
        reasons.append("cross_signal_agreement_on_risk")

    final_label = "malicious" if malicious_score >= 2 else "benign"
    return DecisionResult(
        final_label=final_label,
        is_malicious=(final_label == "malicious"),
        reasons=reasons,
    )
