from fastapi import APIRouter, HTTPException, Request, status

from app.core.config import settings
from app.schemas.classification import ClassificationRequest, ClassificationResponse
from app.schemas.health import HealthResponse
from app.schemas.prediction import PredictionRequest, PredictionResponse

router = APIRouter(tags=["prompt-security"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    anomaly_service = getattr(request.app.state, "inference_service", None)
    anomaly_error = getattr(request.app.state, "inference_service_error", None)
    classifier_service = getattr(request.app.state, "classification_service", None)
    classifier_error = getattr(request.app.state, "classification_service_error", None)

    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        model_loaded=bool(anomaly_service and anomaly_service.is_loaded),
        classifier_model_loaded=bool(classifier_service and classifier_service.is_loaded),
        device=(anomaly_service.device if anomaly_service is not None else "unavailable"),
        sentence_transformer_model=(
            anomaly_service.feature_engineer.embedding_model_name
            if anomaly_service is not None and anomaly_service.feature_engineer is not None
            else settings.sentence_transformer_model
        ),
        threshold=(anomaly_service.threshold if anomaly_service is not None else None),
        classifier_checkpoint_dir=str(settings.classifier_model_dir),
        classifier_error=classifier_error,
        anomaly_error=anomaly_error,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
    service = getattr(request.app.state, "inference_service", None)
    service_error = getattr(request.app.state, "inference_service_error", None)

    if service is None or not service.is_loaded:
        detail = "Inference service is not ready."
        if service_error:
            detail = f"{detail} {service_error}"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )

    result = service.predict(payload.prompt)
    return PredictionResponse(**result)


@router.post("/classify", response_model=ClassificationResponse)
def classify(request: Request, payload: ClassificationRequest) -> ClassificationResponse:
    service = getattr(request.app.state, "classification_service", None)
    service_error = getattr(request.app.state, "classification_service_error", None)

    if service is None or not service.is_loaded:
        detail = "Classification service is not ready."
        if service_error:
            detail = f"{detail} {service_error}"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )

    result = service.predict(payload.prompt)
    return ClassificationResponse(**result)
