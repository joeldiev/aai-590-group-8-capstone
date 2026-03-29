from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from app.schemas.classification import ClassificationResponse
from app.schemas.prediction import PredictionResponse
from app.schemas.severity import SeverityResponse


class DecisionRequest(BaseModel):
    anomaly: PredictionResponse
    classification: ClassificationResponse


class DecisionResponse(BaseModel):
    anomaly: PredictionResponse
    classification: ClassificationResponse
    severity: Optional[SeverityResponse] = None
    final_label: str
    is_malicious: bool
    reasons: list[str]
