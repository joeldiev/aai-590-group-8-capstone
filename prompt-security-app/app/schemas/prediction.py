from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Raw user prompt to score for anomaly detection.",
    )


class PredictionResponse(BaseModel):
    prompt: str
    prompt_normalized: str
    anomaly_score: float
    threshold: float
    is_anomalous: bool
    decision_label: str
    feature_count: int
    score_name: str
    model_type: str
