from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Raw user prompt to classify.",
    )


class ClassificationResponse(BaseModel):
    prompt: str
    prompt_normalized: str
    predicted_label: str
    predicted_class_id: int
    confidence: float
    class_probabilities: dict[str, float]
    decision_rule: str
    min_confidence: float
    is_uncertain: bool
    model_type: str
