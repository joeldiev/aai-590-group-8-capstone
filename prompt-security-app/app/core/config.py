import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)


def _to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    api_prefix: str
    debug: bool

    host: str
    port: int

    model_dir: Path
    feature_artifact_dir: Path
    anomaly_model_dir: Path
    classifier_model_dir: Path

    feature_pipeline_bundle_path: Path
    feature_fill_values_path: Path
    feature_scaler_path: Path
    feature_selected_columns_path: Path
    feature_variance_selector_path: Path
    feature_correlation_drop_columns_path: Path
    feature_metadata_path: Path

    dae_state_dict_path: Path
    dae_metadata_path: Path
    threshold_path: Path
    anomaly_threshold_override: float | None
    classifier_inference_config_path: Path
    classifier_label_mapping_path: Path
    classifier_threshold_path: Path

    sentence_transformer_model: str
    embedding_batch_size: int
    max_prompt_length: int

    force_cpu: bool

    @classmethod
    def from_env(cls) -> "Settings":
        _load_env_file(Path(os.getenv("ENV_FILE", ".env")).resolve())

        project_root = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
        model_dir = (project_root / os.getenv("MODEL_DIR", "models")).resolve()

        feature_artifact_dir = (
            project_root / os.getenv("FEATURE_ARTIFACT_DIR", "models/feature_engineering")
        ).resolve()
        anomaly_model_dir = (
            project_root / os.getenv("ANOMALY_MODEL_DIR", "models/anomaly_detector")
        ).resolve()
        classifier_model_dir = (
            project_root / os.getenv("CLASSIFIER_MODEL_DIR", "models/classifier/best")
        ).resolve()

        return cls(
            app_name=os.getenv("APP_NAME", "dae-prompt-anomaly-api"),
            app_version=os.getenv("APP_VERSION", "1.0.0"),
            api_prefix=os.getenv("API_PREFIX", "/api/v1"),
            debug=_to_bool(os.getenv("DEBUG"), default=False),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            model_dir=model_dir,
            feature_artifact_dir=feature_artifact_dir,
            anomaly_model_dir=anomaly_model_dir,
            classifier_model_dir=classifier_model_dir,
            feature_pipeline_bundle_path=(
                project_root
                / os.getenv(
                    "FEATURE_PIPELINE_BUNDLE_PATH",
                    "models/feature_engineering/feature_pipeline.joblib",
                )
            ).resolve(),
            feature_fill_values_path=(
                project_root
                / os.getenv(
                    "FEATURE_FILL_VALUES_PATH",
                    "models/feature_engineering/fill_values.joblib",
                )
            ).resolve(),
            feature_scaler_path=(
                project_root
                / os.getenv(
                    "FEATURE_SCALER_PATH",
                    "models/feature_engineering/scaler.joblib",
                )
            ).resolve(),
            feature_selected_columns_path=(
                project_root
                / os.getenv(
                    "FEATURE_SELECTED_COLUMNS_PATH",
                    "models/feature_engineering/selected_features.joblib",
                )
            ).resolve(),
            feature_variance_selector_path=(
                project_root
                / os.getenv(
                    "FEATURE_VARIANCE_SELECTOR_PATH",
                    "models/feature_engineering/variance_selector.joblib",
                )
            ).resolve(),
            feature_correlation_drop_columns_path=(
                project_root
                / os.getenv(
                    "FEATURE_CORRELATION_DROP_COLUMNS_PATH",
                    "models/feature_engineering/correlation_drop_columns.joblib",
                )
            ).resolve(),
            feature_metadata_path=(
                project_root
                / os.getenv(
                    "FEATURE_METADATA_PATH",
                    "models/feature_engineering/feature_metadata.json",
                )
            ).resolve(),
            dae_state_dict_path=(
                project_root
                / os.getenv(
                    "DAE_STATE_DICT_PATH",
                    "models/anomaly_detector/dae_state_dict.pt",
                )
            ).resolve(),
            dae_metadata_path=(
                project_root
                / os.getenv(
                    "DAE_METADATA_PATH",
                    "models/anomaly_detector/dae_metadata.json",
                )
            ).resolve(),
            threshold_path=(
                project_root
                / os.getenv(
                    "THRESHOLD_PATH",
                    "models/anomaly_detector/threshold.json",
                )
            ).resolve(),
            anomaly_threshold_override=(
                float(os.getenv("ANOMALY_THRESHOLD_OVERRIDE"))
                if os.getenv("ANOMALY_THRESHOLD_OVERRIDE") is not None
                else None
            ),
            classifier_inference_config_path=(
                project_root
                / os.getenv(
                    "CLASSIFIER_INFERENCE_CONFIG_PATH",
                    "models/classifier/best/inference_config.json",
                )
            ).resolve(),
            classifier_label_mapping_path=(
                project_root
                / os.getenv(
                    "CLASSIFIER_LABEL_MAPPING_PATH",
                    "models/classifier/best/label_mapping.json",
                )
            ).resolve(),
            classifier_threshold_path=(
                project_root
                / os.getenv(
                    "CLASSIFIER_THRESHOLD_PATH",
                    "models/classifier/best/classification_threshold.json",
                )
            ).resolve(),
            sentence_transformer_model=os.getenv(
                "SENTENCE_TRANSFORMER_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            max_prompt_length=int(os.getenv("MAX_PROMPT_LENGTH", "20000")),
            force_cpu=_to_bool(os.getenv("FORCE_CPU"), default=False),
        )


settings = Settings.from_env()
