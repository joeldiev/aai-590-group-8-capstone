from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Tuple

import torch
from torch import nn

from app.core.logging import get_logger
from app.ml.artifacts import ArtifactRegistry
from app.ml.feature_engineering import FeatureEngineer
from app.ml.model import DenoisingAutoencoder

logger = get_logger(__name__)


class CheckpointAutoencoder(nn.Module):
    def __init__(self, input_dim: int, encoder: nn.Sequential, decoder: nn.Sequential) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class InferenceService:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.artifacts = ArtifactRegistry(settings=settings)
        self.feature_engineer: FeatureEngineer | None = None
        self.model: nn.Module | None = None
        self.device = self._resolve_device()
        self.threshold = None
        self.is_loaded = False

    def _resolve_device(self) -> str:
        if self.settings.force_cpu:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        logger.info("Initializing inference service on device=%s", self.device)

        self.artifacts.load()
        if (
            self.settings.anomaly_threshold_override is None
            and not self.artifacts.is_threshold_compatible_with_reconstruction_mse
        ):
            logger.warning(
                "Threshold artifact appears incompatible with current anomaly scoring "
                "(API uses reconstruction_mse, threshold artifact appears to target a different score formula). "
                "Proceeding without override may produce poor anomaly decisions. "
                "Set ANOMALY_THRESHOLD_OVERRIDE to an MSE-calibrated value or export a compatible threshold JSON."
            )

        self.feature_engineer = FeatureEngineer(
            settings=self.settings,
            artifact_registry=self.artifacts,
        )
        self.feature_engineer.load()

        self.model = self._load_autoencoder()
        self.threshold = self.artifacts.threshold
        if self.settings.anomaly_threshold_override is not None:
            logger.warning(
                "Using ANOMALY_THRESHOLD_OVERRIDE=%s instead of threshold artifact value.",
                self.settings.anomaly_threshold_override,
            )
        self.is_loaded = True

        logger.info("Inference service loaded successfully.")

    def _load_autoencoder(self) -> nn.Module:
        try:
            state = torch.load(
                self.settings.dae_state_dict_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            state = torch.load(
                self.settings.dae_state_dict_path,
                map_location=self.device,
            )
        state = self._extract_state_dict(state)
        state = self._normalize_state_dict_keys(state)

        metadata = self.artifacts.dae_metadata or {}
        input_dim, encoder_dims, latent_dim = self._resolve_model_architecture(
            metadata=metadata,
            state_dict=state,
        )

        model = DenoisingAutoencoder(
            input_dim=input_dim,
            encoder_dims=encoder_dims,
            latent_dim=latent_dim,
            activation=metadata.get("activation", "relu"),
            dropout=float(metadata.get("dropout", 0.0)),
            use_batch_norm=bool(metadata.get("use_batch_norm", False)),
        )

        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            logger.warning(
                "Primary autoencoder loading failed; attempting legacy checkpoint layout fallback. Error: %s",
                exc,
            )
            model = self._build_legacy_checkpoint_model(state)
            model.load_state_dict(state, strict=True)

        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _extract_state_dict(state_obj: Any) -> Dict[str, torch.Tensor]:
        if isinstance(state_obj, dict):
            if "state_dict" in state_obj and isinstance(state_obj["state_dict"], dict):
                return state_obj["state_dict"]
            if "model_state_dict" in state_obj and isinstance(state_obj["model_state_dict"], dict):
                return state_obj["model_state_dict"]
            if state_obj and all(isinstance(k, str) for k in state_obj.keys()):
                return state_obj
        raise ValueError("Unsupported checkpoint format: expected a state_dict-like mapping.")

    @staticmethod
    def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Common when checkpoints are saved with DataParallel.
        if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
            return {key[len("module.") :]: value for key, value in state_dict.items()}
        return state_dict

    def _build_legacy_checkpoint_model(self, state_dict: Dict[str, torch.Tensor]) -> CheckpointAutoencoder:
        encoder = self._build_index_aligned_sequential("encoder", state_dict)
        decoder = self._build_index_aligned_sequential("decoder", state_dict)

        first_encoder_linear = next(
            (
                tensor
                for key, tensor in state_dict.items()
                if key.startswith("encoder.") and key.endswith(".weight") and getattr(tensor, "ndim", 0) == 2
            ),
            None,
        )
        if first_encoder_linear is None:
            raise ValueError("Could not infer input_dim from legacy checkpoint encoder weights.")

        input_dim = int(first_encoder_linear.shape[1])
        return CheckpointAutoencoder(input_dim=input_dim, encoder=encoder, decoder=decoder)

    def _build_index_aligned_sequential(
        self,
        prefix: str,
        state_dict: Dict[str, torch.Tensor],
    ) -> nn.Sequential:
        module_map: dict[int, nn.Module] = {}
        key_re = re.compile(
            rf"^{re.escape(prefix)}\.(\d+)\.(weight|bias|running_mean|running_var|num_batches_tracked)$"
        )

        for key, tensor in state_dict.items():
            match = key_re.match(key)
            if not match:
                continue

            idx = int(match.group(1))
            attr = match.group(2)

            if attr in {"running_mean", "running_var", "num_batches_tracked"}:
                if idx not in module_map:
                    num_features = int(tensor.shape[0]) if getattr(tensor, "ndim", 0) > 0 else 0
                    module_map[idx] = nn.BatchNorm1d(num_features=num_features)
                continue

            if attr == "weight":
                if getattr(tensor, "ndim", 0) == 2:
                    out_features, in_features = int(tensor.shape[0]), int(tensor.shape[1])
                    module_map[idx] = nn.Linear(in_features, out_features)
                elif getattr(tensor, "ndim", 0) == 1 and idx not in module_map:
                    module_map[idx] = nn.BatchNorm1d(num_features=int(tensor.shape[0]))

        if not module_map:
            raise ValueError(f"Could not build legacy `{prefix}` sequential from checkpoint keys.")

        max_idx = max(module_map.keys())
        param_indices = sorted(module_map.keys())
        modules: list[nn.Module] = []

        for idx in range(max_idx + 1):
            if idx in module_map:
                modules.append(module_map[idx])
                continue

            prev_param_idx = max((i for i in param_indices if i < idx), default=None)
            next_param_idx = min((i for i in param_indices if i > idx), default=None)
            if prev_param_idx is None or next_param_idx is None:
                modules.append(nn.Identity())
                continue

            gap_len = next_param_idx - prev_param_idx - 1
            position_in_gap = idx - prev_param_idx
            if position_in_gap == 1:
                modules.append(nn.LeakyReLU(negative_slope=0.01))
            elif gap_len >= 2:
                modules.append(nn.Dropout(p=0.0))
            else:
                modules.append(nn.Identity())

        return nn.Sequential(*modules)

    def _resolve_model_architecture(
        self,
        metadata: Dict[str, Any],
        state_dict: Dict[str, torch.Tensor],
    ) -> Tuple[int, Iterable[int], int]:
        inferred_input_dim, inferred_encoder_dims, inferred_latent_dim = self._infer_architecture_from_state_dict(
            state_dict
        )

        input_dim = self._coerce_optional_int(metadata.get("input_dim"))
        latent_dim = self._coerce_optional_int(metadata.get("latent_dim"))
        encoder_dims = self._coerce_optional_int_list(metadata.get("encoder_dims"))

        if input_dim is None:
            input_dim = inferred_input_dim
            logger.warning("`input_dim` missing in metadata; inferred input_dim=%s from checkpoint.", input_dim)
        if latent_dim is None:
            latent_dim = inferred_latent_dim
            logger.warning("`latent_dim` missing in metadata; inferred latent_dim=%s from checkpoint.", latent_dim)
        if encoder_dims is None:
            encoder_dims = inferred_encoder_dims
            logger.warning(
                "`encoder_dims` missing in metadata; inferred encoder_dims=%s from checkpoint.",
                encoder_dims,
            )

        return int(input_dim), list(encoder_dims), int(latent_dim)

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _coerce_optional_int_list(value: Any) -> list[int] | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        return None

    @staticmethod
    def _sort_key_for_layer(name: str) -> Tuple[Tuple[int, ...], str]:
        nums = tuple(int(n) for n in re.findall(r"\d+", name))
        return nums, name

    def _infer_architecture_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> Tuple[int, list[int], int]:
        encoder_linears: list[Tuple[str, torch.Tensor]] = []
        for name, tensor in state_dict.items():
            if not name.startswith("encoder."):
                continue
            if not name.endswith(".weight"):
                continue
            if getattr(tensor, "ndim", 0) != 2:
                continue
            encoder_linears.append((name, tensor))

        if not encoder_linears:
            raise ValueError(
                "Could not infer autoencoder architecture from checkpoint: no encoder linear weights found."
            )

        encoder_linears.sort(key=lambda item: self._sort_key_for_layer(item[0]))
        layer_shapes = [tuple(t.shape) for _, t in encoder_linears]

        input_dim = int(layer_shapes[0][1])
        latent_dim = int(layer_shapes[-1][0])
        encoder_dims = [int(shape[0]) for shape in layer_shapes[:-1]]

        return input_dim, encoder_dims, latent_dim

    def predict(self, prompt: str) -> Dict[str, Any]:
        anomaly_score, feature_count, metadata = self.score_prompt(prompt)
        is_anomalous = anomaly_score >= self.threshold

        return {
            "prompt": metadata["prompt_original"],
            "prompt_normalized": metadata["prompt_normalized"],
            "anomaly_score": anomaly_score,
            "threshold": float(self.threshold),
            "is_anomalous": bool(is_anomalous),
            "decision_label": "malicious_or_anomalous" if is_anomalous else "benign_like",
            "feature_count": feature_count,
            "score_name": "reconstruction_mse",
            "model_type": "denoising_deep_autoencoder",
        }

    def score_prompt(self, prompt: str) -> tuple[float, int, dict[str, str]]:
        if not self.is_loaded or self.model is None or self.feature_engineer is None:
            raise RuntimeError("Inference service is not loaded.")

        raw_feature_df, metadata = self.feature_engineer.build_feature_frame(prompt)
        model_input = self.feature_engineer.transform_for_model(raw_feature_df)

        input_dim = model_input.shape[1]
        expected_input_dim = self.model.input_dim

        if input_dim != expected_input_dim:
            raise ValueError(
                f"Feature dimension mismatch. Engineered features={input_dim}, "
                f"model expects={expected_input_dim}. "
                "Check exported feature pipeline artifacts and model metadata."
            )

        x = torch.tensor(model_input, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            reconstruction = self.model(x)
            per_feature_error = torch.square(x - reconstruction)
            reconstruction_mse = torch.mean(per_feature_error, dim=1).cpu().numpy()[0]

        anomaly_score = float(reconstruction_mse)
        return anomaly_score, int(model_input.shape[1]), metadata
