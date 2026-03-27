from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.ml.classification import ClassificationService
from app.ml.inference import InferenceService

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application...")
    inference_service = InferenceService(settings=settings)
    inference_service.load()
    app.state.inference_service = inference_service

    classification_service = ClassificationService(settings=settings)
    app.state.classification_service = None
    app.state.classification_service_error = None
    try:
        classification_service.load()
        app.state.classification_service = classification_service
    except Exception as exc:
        app.state.classification_service_error = str(exc)
        logger.exception("Classification service failed to initialize: %s", exc)

    logger.info("Application startup complete.")
    yield
    logger.info("Shutting down application...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "FastAPI service for prompt anomaly detection and prompt classification."
    ),
    lifespan=lifespan,
)

app.include_router(router, prefix=settings.api_prefix)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
