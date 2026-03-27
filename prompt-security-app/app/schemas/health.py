from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    model_loaded: bool
    classifier_model_loaded: bool = False
    device: str
    sentence_transformer_model: str
    threshold: float | None = None
    classifier_checkpoint_dir: str | None = None
    classifier_error: str | None = None
    anomaly_error: str | None = None
