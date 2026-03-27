from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.classification import ClassificationRequest, ClassificationResponse
from app.schemas.health import HealthResponse
from app.schemas.prediction import PredictionRequest, PredictionResponse

router = APIRouter(tags=["anomaly-detection"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    anomaly_service = request.app.state.inference_service
    classifier_service = getattr(request.app.state, "classification_service", None)
    classifier_error = getattr(request.app.state, "classification_service_error", None)

    return HealthResponse(
        status="ok",
        app_name=anomaly_service.settings.app_name,
        version=anomaly_service.settings.app_version,
        model_loaded=anomaly_service.is_loaded,
        classifier_model_loaded=bool(classifier_service and classifier_service.is_loaded),
        device=anomaly_service.device,
        sentence_transformer_model=anomaly_service.feature_engineer.embedding_model_name,
        threshold=anomaly_service.threshold,
        classifier_checkpoint_dir=str(anomaly_service.settings.classifier_model_dir),
        classifier_error=classifier_error,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
    service = request.app.state.inference_service

    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service is not ready.",
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
