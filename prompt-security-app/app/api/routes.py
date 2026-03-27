from fastapi import APIRouter, HTTPException, Request, status

from app.core.config import settings
from app.ml.decision import decide_prompt_risk
from app.schemas.classification import ClassificationRequest, ClassificationResponse
from app.schemas.decision import DecisionRequest, DecisionResponse
from app.schemas.health import HealthResponse
from app.schemas.prediction import PredictionRequest, PredictionResponse

router = APIRouter(tags=["prompt-security"])


def _get_inference_service(request: Request):
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

    return service


def _get_classification_service(request: Request):
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

    return service


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
    service = _get_inference_service(request)
    result = service.predict(payload.prompt)
    return PredictionResponse(**result)


@router.post("/classify", response_model=ClassificationResponse)
def classify(request: Request, payload: ClassificationRequest) -> ClassificationResponse:
    service = _get_classification_service(request)
    result = service.predict(payload.prompt)
    return ClassificationResponse(**result)


@router.post("/decision", response_model=DecisionResponse)
def decision(payload: DecisionRequest) -> DecisionResponse:
    anomaly = payload.anomaly
    classification = payload.classification

    if anomaly.prompt != classification.prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Anomaly and classification results must reference the same prompt.",
        )

    decision_result = decide_prompt_risk(anomaly=anomaly, classification=classification)
    return DecisionResponse(
        anomaly=anomaly,
        classification=classification,
        final_label=decision_result.final_label,
        is_malicious=decision_result.is_malicious,
        reasons=decision_result.reasons,
    )


@router.post("/prompt", response_model=DecisionResponse)
def prompt(request: Request, payload: PredictionRequest) -> DecisionResponse:
    inference_service = _get_inference_service(request)
    classification_service = _get_classification_service(request)

    anomaly = PredictionResponse(**inference_service.predict(payload.prompt))
    classification = ClassificationResponse(**classification_service.predict(payload.prompt))
    decision_result = decide_prompt_risk(anomaly=anomaly, classification=classification)

    # Severity evaluation — only when both classifiers flag malicious
    severity_response = None
    if decision_result.is_malicious:
        severity_service = getattr(request.app.state, "severity_service", None)
        if severity_service and severity_service.is_loaded:
            from app.schemas.severity import SeverityResponse

            severity_result = severity_service.evaluate(
                text=payload.prompt,
                anomaly=anomaly,
                classification=classification,
            )
            severity_response = SeverityResponse(
                severity_tier=severity_result.severity_tier,
                severity_score=severity_result.severity_score,
                threat_type=severity_result.threat_type,
                threat_confidence=severity_result.threat_confidence,
                threat_class_probabilities=severity_result.threat_class_probabilities,
                threat_intel=severity_result.threat_intel,
                scoring_breakdown=severity_result.scoring_breakdown,
                latency_ms=severity_result.latency_ms,
            )

    return DecisionResponse(
        anomaly=anomaly,
        classification=classification,
        severity=severity_response,
        final_label=decision_result.final_label,
        is_malicious=decision_result.is_malicious,
        reasons=decision_result.reasons,
    )
