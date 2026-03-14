"""
Training callbacks and metric computation for HF Trainer.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import TrainerCallback


def compute_metrics(eval_pred) -> dict:
    """Compute per-class and macro metrics for Trainer eval loop."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
    }


class MetricsCallback(TrainerCallback):
    """Log metrics at the end of each epoch."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            epoch = state.epoch or 0
            print(f"\n[Epoch {epoch:.0f}] "
                  f"val_loss={metrics.get('eval_loss', 0):.4f}  "
                  f"macro_f1={metrics.get('eval_macro_f1', 0):.4f}  "
                  f"accuracy={metrics.get('eval_accuracy', 0):.4f}")
