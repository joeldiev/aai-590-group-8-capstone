import json
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.logging import get_logger
from app.utils.text import normalize_prompt

logger = get_logger(__name__)


def _load_json_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class ClassificationService:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.device = self._resolve_device()

        self.tokenizer = None
        self.model = None
        self.inference_config: Dict[str, Any] = {}
        self.label_mapping: Dict[str, Any] = {}
        self.threshold_config: Dict[str, Any] = {}

        self.id2label: Dict[int, str] = {}
        self.min_confidence = 0.0
        self.is_loaded = False

    def _resolve_device(self) -> str:
        if self.settings.force_cpu:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        checkpoint_dir = self.settings.classifier_model_dir
        logger.info("Loading classifier from %s on device=%s", checkpoint_dir, self.device)

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Classifier checkpoint dir does not exist: {checkpoint_dir}")

        self.inference_config = _load_json_if_exists(self.settings.classifier_inference_config_path) or {}
        self.label_mapping = _load_json_if_exists(self.settings.classifier_label_mapping_path) or {}
        self.threshold_config = _load_json_if_exists(self.settings.classifier_threshold_path) or {}

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self._resolve_id2label()
        self.min_confidence = float(self.threshold_config.get("min_confidence", 0.0))
        self.is_loaded = True
        logger.info("Classifier loaded successfully.")

    def _resolve_id2label(self) -> Dict[int, str]:
        mapping = self.label_mapping.get("id2label")
        if isinstance(mapping, dict) and mapping:
            return {int(k): str(v) for k, v in mapping.items()}

        cfg_mapping = self.inference_config.get("id2label")
        if isinstance(cfg_mapping, dict) and cfg_mapping:
            return {int(k): str(v) for k, v in cfg_mapping.items()}

        if self.model is not None and getattr(self.model.config, "id2label", None):
            return {int(k): str(v) for k, v in self.model.config.id2label.items()}

        num_labels = int(getattr(self.model.config, "num_labels", 2)) if self.model is not None else 2
        return {i: f"class_{i}" for i in range(num_labels)}

    def predict(self, prompt: str) -> Dict[str, Any]:
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("Classification service is not loaded.")

        normalized = normalize_prompt(prompt, max_length=self.settings.max_prompt_length)
        max_seq_length = int(self.inference_config.get("max_seq_length", 128))

        encoded = self.tokenizer(
            normalized,
            truncation=True,
            padding=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

        predicted_idx = int(torch.argmax(logits, dim=-1).item())
        confidence = float(probs[predicted_idx])
        predicted_label = self.id2label.get(predicted_idx, str(predicted_idx))
        is_uncertain = confidence < self.min_confidence

        class_probabilities = {
            self.id2label.get(i, str(i)): float(p)
            for i, p in enumerate(probs)
        }

        return {
            "prompt": prompt,
            "prompt_normalized": normalized,
            "predicted_label": predicted_label,
            "predicted_class_id": predicted_idx,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "decision_rule": str(self.threshold_config.get("decision_rule", "argmax")),
            "min_confidence": self.min_confidence,
            "is_uncertain": is_uncertain,
            "model_type": str(self.inference_config.get("model_type", "roberta_sequence_classification")),
        }
