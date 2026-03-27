from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from app.core.logging import get_logger
from app.utils.text import normalize_prompt

logger = get_logger(__name__)


DEFAULT_TOKENIZER_MODEL = "distilbert-base-uncased"
DEFAULT_PCA_COMPONENTS = 32
DEFAULT_PHRASE_RULES = {
    "instruction_override_phrases": [],
    "roleplay_phrases": [],
    "payload_phrases": [],
    "social_engineering_phrases": [],
    "obfuscation_phrases": [],
}


class FeatureEngineer:
    def __init__(self, settings, artifact_registry) -> None:
        self.settings = settings
        self.artifacts = artifact_registry
        self.embedding_model_name = artifact_registry.embedding_model_name
        self.embedding_model: SentenceTransformer | None = None
        self.tokenizer: Any = None
        self.pca: Any = None
        self.phrase_rules: Dict[str, list[str]] = dict(DEFAULT_PHRASE_RULES)
        self.pca_components = DEFAULT_PCA_COMPONENTS

    def load(self) -> None:
        self._load_phrase_rules()
        self._load_pca()
        self._load_tokenizer()
        logger.info("Loading sentence-transformer model: %s", self.embedding_model_name)
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def build_feature_frame(self, prompt: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
        if self.embedding_model is None:
            raise RuntimeError("Embedding model has not been loaded.")

        normalized = normalize_prompt(prompt, max_length=self.settings.max_prompt_length)
        lexical_features = self._compute_schema_lexical_features(normalized)
        embedding_features = self._compute_schema_embedding_features(normalized)

        feature_row = {**lexical_features, **embedding_features}
        df = pd.DataFrame([feature_row])

        metadata = {
            "prompt_original": prompt,
            "prompt_normalized": normalized,
        }
        return df, metadata

    def transform_for_model(self, raw_feature_df: pd.DataFrame) -> np.ndarray:
        df = raw_feature_df.copy()

        if self.artifacts.selected_features is not None:
            selected = list(self.artifacts.selected_features)
            for column in selected:
                if column not in df.columns:
                    df[column] = 0.0
            df = df[selected]

        if self.artifacts.fill_values is not None:
            fill_values = self.artifacts.fill_values
            if isinstance(fill_values, pd.Series):
                for column in df.columns:
                    if column in fill_values.index:
                        df[column] = df[column].fillna(fill_values[column])
            elif isinstance(fill_values, dict):
                for column, value in fill_values.items():
                    if column in df.columns:
                        df[column] = df[column].fillna(value)

        if self.artifacts.correlation_drop_columns is not None:
            to_drop = [col for col in self.artifacts.correlation_drop_columns if col in df.columns]
            if to_drop:
                df = df.drop(columns=to_drop)

        if self.artifacts.variance_selector is not None:
            transformed = self.artifacts.variance_selector.transform(df)
            if hasattr(self.artifacts.variance_selector, "get_feature_names_out"):
                feature_names = self.artifacts.variance_selector.get_feature_names_out(df.columns)
                df = pd.DataFrame(transformed, columns=feature_names)
            else:
                df = pd.DataFrame(transformed)

        if self.artifacts.scaler is not None:
            scaled = self.artifacts.scaler.transform(df)
            return np.asarray(scaled, dtype=np.float32)

        return np.asarray(df, dtype=np.float32)

    def _compute_schema_embedding_features(self, prompt: str) -> Dict[str, float]:
        embedding = self.embedding_model.encode(
            [prompt],
            batch_size=self.settings.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        embedding = np.asarray(embedding, dtype=np.float32)
        embedding_norm = float(np.linalg.norm(embedding))

        pca_values = np.zeros(self.pca_components, dtype=np.float32)
        if self.pca is not None:
            transformed = self.pca.transform(embedding.reshape(1, -1))
            pca_values = np.asarray(transformed[0], dtype=np.float32)
            if len(pca_values) != self.pca_components:
                self.pca_components = len(pca_values)

        features: Dict[str, float] = {"embedding_norm": embedding_norm}
        for idx in range(self.pca_components):
            features[f"embed_pca_{idx + 1}"] = float(pca_values[idx] if idx < len(pca_values) else 0.0)
        return features

    def _compute_schema_lexical_features(self, prompt: str) -> Dict[str, float]:
        tokens = prompt.split()
        word_count = len(tokens)
        char_count = len(prompt)

        token_count = self._token_count(prompt)
        whitespace_count = sum(1 for c in prompt if c.isspace())
        punctuation_count = sum(1 for c in prompt if c in string.punctuation)
        non_ascii_count = sum(1 for c in prompt if ord(c) > 127)
        special_char_count = sum(1 for c in prompt if not c.isalnum() and not c.isspace())

        token_word_ratio = token_count / word_count if word_count else 0.0
        whitespace_ratio = whitespace_count / char_count if char_count else 0.0
        punctuation_ratio = punctuation_count / char_count if char_count else 0.0
        special_char_ratio = special_char_count / char_count if char_count else 0.0
        non_ascii_ratio = non_ascii_count / char_count if char_count else 0.0

        markdown_symbol_count = sum(prompt.count(ch) for ch in "#*`>_-[]()")
        has_code_block = 1.0 if "```" in prompt else 0.0
        repeated_punct_count = float(
            len(re.findall(r"([!?.,;:])\1+", prompt))
        )

        quote_count = prompt.count('"') + prompt.count("'")
        bracket_count = sum(prompt.count(ch) for ch in "()[]{}<>")
        colon_count = prompt.count(":")
        semicolon_count = prompt.count(";")

        instruction_override_score = self._phrase_score(prompt, self.phrase_rules["instruction_override_phrases"])
        roleplay_score = self._phrase_score(prompt, self.phrase_rules["roleplay_phrases"])
        payload_request_score = self._phrase_score(prompt, self.phrase_rules["payload_phrases"])
        social_engineering_score = self._phrase_score(prompt, self.phrase_rules["social_engineering_phrases"])
        obfuscation_score = self._phrase_score(prompt, self.phrase_rules["obfuscation_phrases"])

        return {
            "token_count": float(token_count),
            "token_word_ratio": float(token_word_ratio),
            "whitespace_count": float(whitespace_count),
            "whitespace_ratio": float(whitespace_ratio),
            "punctuation_count": float(punctuation_count),
            "punctuation_ratio": float(punctuation_ratio),
            "special_char_ratio": float(special_char_ratio),
            "non_ascii_count": float(non_ascii_count),
            "non_ascii_ratio": float(non_ascii_ratio),
            "markdown_symbol_count": float(markdown_symbol_count),
            "has_code_block": float(has_code_block),
            "repeated_punct_count": float(repeated_punct_count),
            "colon_count": float(colon_count),
            "semicolon_count": float(semicolon_count),
            "quote_count": float(quote_count),
            "bracket_count": float(bracket_count),
            "instruction_override_score": float(instruction_override_score),
            "roleplay_score": float(roleplay_score),
            "payload_request_score": float(payload_request_score),
            "social_engineering_score": float(social_engineering_score),
            "obfuscation_score": float(obfuscation_score),
            "has_instruction_override": float(1.0 if instruction_override_score > 0 else 0.0),
            "has_roleplay": float(1.0 if roleplay_score > 0 else 0.0),
            "has_payload_request": float(1.0 if payload_request_score > 0 else 0.0),
            "has_social_engineering": float(1.0 if social_engineering_score > 0 else 0.0),
            "has_obfuscation": float(1.0 if obfuscation_score > 0 else 0.0),
        }

    def _load_phrase_rules(self) -> None:
        rules_path = self.settings.feature_artifact_dir / "phrase_rules.json"
        if not rules_path.exists():
            logger.warning("phrase_rules.json not found at %s; using empty phrase rules.", rules_path)
            return
        try:
            with rules_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            merged = dict(DEFAULT_PHRASE_RULES)
            for key in merged:
                value = loaded.get(key, [])
                if isinstance(value, list):
                    merged[key] = [str(v).strip().lower() for v in value if str(v).strip()]
            self.phrase_rules = merged
        except Exception as exc:
            logger.warning("Failed to load phrase rules from %s: %s", rules_path, exc)

    def _load_pca(self) -> None:
        pca_path = self.settings.feature_artifact_dir / "pca.joblib"
        if not pca_path.exists():
            logger.warning("pca.joblib not found at %s; PCA features will be zeroed.", pca_path)
            return
        try:
            self.pca = joblib.load(pca_path)
            components = int(getattr(self.pca, "n_components_", DEFAULT_PCA_COMPONENTS))
            self.pca_components = components
            logger.info("Loaded PCA transform from %s with %s components.", pca_path, components)
        except Exception as exc:
            logger.warning("Failed to load PCA transform from %s: %s", pca_path, exc)
            self.pca = None

    def _load_tokenizer(self) -> None:
        metadata_path = self.settings.feature_metadata_path
        tokenizer_model = DEFAULT_TOKENIZER_MODEL
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                tokenizer_model = str(metadata.get("tokenizer_model", DEFAULT_TOKENIZER_MODEL))
            except Exception as exc:
                logger.warning("Failed to read feature metadata for tokenizer model: %s", exc)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
            logger.info("Loaded tokenizer model for token_count: %s", tokenizer_model)
        except Exception as exc:
            logger.warning(
                "Failed to load tokenizer model `%s`; falling back to whitespace token count. Error: %s",
                tokenizer_model,
                exc,
            )
            self.tokenizer = None

    def _token_count(self, prompt: str) -> int:
        if self.tokenizer is not None:
            try:
                encoded = self.tokenizer(prompt, add_special_tokens=True, truncation=False)
                return int(len(encoded["input_ids"]))
            except Exception:
                pass
        return len(prompt.split())

    @staticmethod
    def _phrase_score(text: str, phrases: list[str]) -> float:
        lowered = text.lower()
        return float(sum(1 for phrase in phrases if phrase and phrase in lowered))
