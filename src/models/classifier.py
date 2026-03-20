"""
RoBERTa-based sequence classifier for AGL.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import MODEL_NAME, NUM_LABELS, ID2LABEL, LABEL2ID


def build_classifier(
    num_labels: int = NUM_LABELS,
    model_name: str = MODEL_NAME,
) -> AutoModelForSequenceClassification:
    """Load a pre-trained RoBERTa model with a classification head.

    Args:
        num_labels: Number of output classes.
        model_name: HuggingFace model identifier.

    Returns:
        AutoModelForSequenceClassification ready for fine-tuning.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model


def load_trained_classifier(
    checkpoint_path: str,
    num_labels: int = NUM_LABELS,
) -> AutoModelForSequenceClassification:
    """Load a fine-tuned classifier from a checkpoint directory."""
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        local_files_only=True,
    )
    return model


def get_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
    """Load the tokenizer for the classifier."""
    return AutoTokenizer.from_pretrained(model_name)


def extract_cls_embeddings(
    model: AutoModelForSequenceClassification,
    dataset,
    batch_size: int = 32,
    device: str = "cpu",
) -> torch.Tensor:
    """Extract [CLS] token embeddings from the fine-tuned model.

    Args:
        model: Fine-tuned RoBERTa classifier.
        dataset: HF Dataset with input_ids and attention_mask.
        batch_size: Batch size for inference.
        device: Device to use.

    Returns:
        Tensor of shape (N, hidden_dim) with [CLS] embeddings.
    """
    from torch.utils.data import DataLoader

    model = model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size)
    all_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # [CLS] is the first token
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)
