import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from app.core.logging import get_logger

logger = get_logger(__name__)


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_joblib_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    return joblib.load(path)


class ArtifactRegistry:
    """
    Supports either:
    1. one bundled feature_pipeline.joblib
    2. multiple separate files exported from notebooks

    Expected bundle keys if using a single joblib:
        - fill_values
        - scaler
        - selected_features
        - variance_selector
        - correlation_drop_columns
        - feature_names_after_engineering
        - embedding_model_name
        - embedding_dim

    Expected anomaly metadata JSON:
        {
          "input_dim": 391,
          "encoder_dims": [256, 128],
          "latent_dim": 64,
          "activation": "relu",
          "dropout": 0.1,
          "use_batch_norm": true
        }

    Expected threshold JSON:
        {
          "threshold": 0.0841,
          "score_name": "reconstruction_mse"
        }
    """

    def __init__(self, settings) -> None:
        self.settings = settings

        self.feature_bundle = None
        self.feature_metadata = None

        self.fill_values = None
        self.scaler = None
        self.selected_features = None
        self.variance_selector = None
        self.correlation_drop_columns = None

        self.dae_metadata = None
        self.threshold_metadata = None

    def load(self) -> None:
        logger.info("Loading artifacts...")

        self.feature_bundle = load_joblib_if_exists(self.settings.feature_pipeline_bundle_path)
        self.feature_metadata = load_json_if_exists(self.settings.feature_metadata_path)

        if self.feature_bundle is not None:
            logger.info("Loaded feature pipeline bundle from %s", self.settings.feature_pipeline_bundle_path)
            self.fill_values = self.feature_bundle.get("fill_values")
            self.scaler = self.feature_bundle.get("scaler")
            self.selected_features = self.feature_bundle.get("selected_features")
            self.variance_selector = self.feature_bundle.get("variance_selector")
            self.correlation_drop_columns = self.feature_bundle.get("correlation_drop_columns")
        else:
            self.fill_values = load_joblib_if_exists(self.settings.feature_fill_values_path)
            self.scaler = load_joblib_if_exists(self.settings.feature_scaler_path)
            self.selected_features = load_joblib_if_exists(self.settings.feature_selected_columns_path)
            self.variance_selector = load_joblib_if_exists(self.settings.feature_variance_selector_path)
            self.correlation_drop_columns = load_joblib_if_exists(
                self.settings.feature_correlation_drop_columns_path
            )

        self.dae_metadata = load_json_if_exists(self.settings.dae_metadata_path)
        self.threshold_metadata = load_json_if_exists(self.settings.threshold_path)

        if self.dae_metadata is None:
            raise FileNotFoundError(
                f"Missing autoencoder metadata file: {self.settings.dae_metadata_path}"
            )

        if not self.settings.dae_state_dict_path.exists():
            raise FileNotFoundError(
                f"Missing autoencoder state dict: {self.settings.dae_state_dict_path}"
            )

        if self.threshold_metadata is None or "threshold" not in self.threshold_metadata:
            raise FileNotFoundError(
                f"Missing threshold JSON or threshold key: {self.settings.threshold_path}"
            )

        logger.info("Artifacts loaded successfully.")

    @property
    def threshold(self) -> float:
        if self.settings.anomaly_threshold_override is not None:
            return float(self.settings.anomaly_threshold_override)
        return float(self.threshold_metadata["threshold"])

    @property
    def threshold_score_name(self) -> str | None:
        if not self.threshold_metadata:
            return None
        score_name = self.threshold_metadata.get("score_name")
        if score_name is None:
            return None
        return str(score_name)

    @property
    def threshold_score_formula(self) -> str | None:
        if not self.threshold_metadata:
            return None
        score_formula = self.threshold_metadata.get("score_formula")
        if score_formula is None:
            return None
        return str(score_formula)

    @property
    def is_threshold_compatible_with_reconstruction_mse(self) -> bool:
        score_name = (self.threshold_score_name or "").strip().lower()
        if score_name:
            return score_name == "reconstruction_mse"

        score_formula = (self.threshold_score_formula or "").strip().lower()
        if score_formula:
            if "robust_zscore" in score_formula:
                return False
            return "reconstruction_mse" in score_formula

        return True

    @property
    def embedding_model_name(self) -> str:
        if self.feature_bundle and self.feature_bundle.get("embedding_model_name"):
            return str(self.feature_bundle["embedding_model_name"])
        if self.feature_metadata and self.feature_metadata.get("embedding_model_name"):
            return str(self.feature_metadata["embedding_model_name"])
        return self.settings.sentence_transformer_model

    @property
    def embedding_dim(self) -> Optional[int]:
        if self.feature_bundle and self.feature_bundle.get("embedding_dim") is not None:
            return int(self.feature_bundle["embedding_dim"])
        if self.feature_metadata and self.feature_metadata.get("embedding_dim") is not None:
            return int(self.feature_metadata["embedding_dim"])
        return None
