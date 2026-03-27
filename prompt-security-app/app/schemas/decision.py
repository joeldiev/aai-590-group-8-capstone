from __future__ import annotations

from pydantic import BaseModel

from app.schemas.classification import ClassificationResponse
from app.schemas.prediction import PredictionResponse


class DecisionRequest(BaseModel):
    anomaly: PredictionResponse
    classification: ClassificationResponse


class DecisionResponse(BaseModel):
    anomaly: PredictionResponse
    classification: ClassificationResponse
    final_label: str
    is_malicious: bool
    reasons: list[str]
