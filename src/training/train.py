"""
Unified training script for AGL.

Modes:
  --mode classifier  → Fine-tune RoBERTa with class-weighted loss
  --mode anomaly     → Extract [CLS] embeddings, fit Mahalanobis detector
  --mode both        → Run classifier then anomaly sequentially
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from src.config import (
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    NUM_EPOCHS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    SEED,
    MODELS_DIR,
    PROCESSED_DIR,
    PCA_COMPONENTS,
    OOD_RECALL_TARGET,
    NUM_LABELS,
)
from src.data.tokenize_dataset import load_tokenized_splits
from src.models.classifier import build_classifier, extract_cls_embeddings
from src.models.anomaly_detector import MahalanobisOODDetector
from src.training.callbacks import MetricsCallback, compute_metrics
from src.utils.reproducibility import set_seed


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_classifier(
    output_dir: str | Path | None = None,
    learning_rate: float = LEARNING_RATE,
    num_epochs: int = NUM_EPOCHS,
    max_length: int = MAX_SEQ_LENGTH,
    batch_size: int = TRAIN_BATCH_SIZE,
) -> Path:
    """Fine-tune RoBERTa classifier on the AGL dataset.

    Returns:
        Path to the best checkpoint directory.
    """
    set_seed(SEED)

    if output_dir is None:
        output_dir = MODELS_DIR / "classifier"
    output_dir = Path(output_dir)

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    splits = load_tokenized_splits(tokenizer, max_length)

    # Compute class weights (inverse frequency)
    train_labels = splits["train"]["labels"].numpy()
    class_counts = Counter(train_labels)
    total = sum(class_counts.values())
    n_classes = len(class_counts)
    class_weights = [
        total / (n_classes * class_counts.get(i, 1))
        for i in range(NUM_LABELS)
    ]
    print(f"Class weights: {class_weights}")

    # Build model
    model = build_classifier()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=learning_rate,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        logging_steps=50,
        report_to="none",  # Change to "wandb" if W&B is configured
        fp16=torch.cuda.is_available(),
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["val"],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
            MetricsCallback(),
        ],
    )

    # Train
    print(f"\nStarting training: {num_epochs} epochs, lr={learning_rate}")
    trainer.train()

    # Save best model + tokenizer
    best_path = output_dir / "best"
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))
    print(f"\nBest model saved → {best_path}")

    return best_path


def train_anomaly(
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Fit Mahalanobis OOD detector on [CLS] embeddings from the fine-tuned model.

    Args:
        checkpoint_path: Path to fine-tuned classifier. If None, uses default.
        output_dir: Where to save the detector. If None, uses default.

    Returns:
        Path to the saved detector directory.
    """
    set_seed(SEED)

    if checkpoint_path is None:
        checkpoint_path = MODELS_DIR / "classifier" / "best"
    if output_dir is None:
        output_dir = MODELS_DIR / "anomaly"

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)

    # Load model and data
    from src.models.classifier import load_trained_classifier
    model = load_trained_classifier(str(checkpoint_path))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    splits = load_tokenized_splits(tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract [CLS] embeddings from train and val sets
    print("Extracting [CLS] embeddings from training set...")
    train_embeddings = extract_cls_embeddings(model, splits["train"], device=device)
    train_labels = splits["train"]["labels"].numpy()

    print("Extracting [CLS] embeddings from validation set...")
    val_embeddings = extract_cls_embeddings(model, splits["val"], device=device)

    # Fit detector
    print(f"Fitting Mahalanobis detector (PCA → {PCA_COMPONENTS} dims)...")
    detector = MahalanobisOODDetector(n_components=PCA_COMPONENTS)
    detector.fit(train_embeddings.numpy(), train_labels)

    # Calibrate threshold on validation set
    detector.calibrate_threshold(
        val_embeddings.numpy(),
        recall_target=OOD_RECALL_TARGET,
    )

    # Save
    detector.save(output_dir)

    return output_dir


def train_both() -> dict[str, Path]:
    """Run classifier training followed by anomaly detector fitting."""
    classifier_path = train_classifier()
    anomaly_path = train_anomaly(checkpoint_path=classifier_path)
    return {"classifier": classifier_path, "anomaly": anomaly_path}
