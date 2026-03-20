"""
Evaluation metrics for AGL.

Per-class P/R/F1, macro-F1, confusion matrix, ROC-AUC, latency benchmarking.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.config import LABEL_NAMES, NUM_LABELS, RESULTS_DIR


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] = LABEL_NAMES,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Full evaluation suite.

    Args:
        y_true: Ground truth labels (int).
        y_pred: Predicted labels (int).
        label_names: Class names for display.
        y_prob: For binary: (N,) array of P(malicious).
                For multiclass: (N, C) array of probabilities.

    Returns:
        Dict with report, confusion matrix, and AUC scores.
    """
    present_labels = sorted(set(y_true) | set(y_pred))
    present_names = [label_names[i] for i in present_labels if i < len(label_names)]

    report = classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=present_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)

    result = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "label_names": present_names,
    }

    # ROC-AUC and PR-AUC
    if y_prob is not None:
        try:
            if NUM_LABELS == 2:
                # Binary: y_prob should be P(malicious) — shape (N,)
                probs = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                result["roc_auc"] = roc_auc_score(y_true, probs)
                result["pr_auc"] = average_precision_score(y_true, probs)
            else:
                # Multiclass: one-vs-rest
                result["roc_auc"] = roc_auc_score(
                    y_true, y_prob,
                    multi_class="ovr",
                    average="macro",
                    labels=present_labels,
                )
        except ValueError as e:
            print(f"[metrics] AUC computation failed: {e}")
            result["roc_auc"] = None

    return result


def benchmark_latency(
    pipeline,
    texts: list[str],
    n_runs: int = 3,
) -> dict:
    """Benchmark inference latency.

    Args:
        pipeline: AGLPipeline instance.
        texts: List of test prompts.
        n_runs: Number of timing runs for averaging.

    Returns:
        Dict with mean, p99, min latency in milliseconds.
    """
    latencies = []

    for _ in range(n_runs):
        for text in texts:
            start = time.perf_counter()
            pipeline.predict(text)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

    latencies = np.array(latencies)
    return {
        "mean_ms": float(latencies.mean()),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
        "std_ms": float(latencies.std()),
        "n_samples": len(texts),
        "n_runs": n_runs,
    }


def save_results(results: dict, name: str, output_dir: Path | None = None) -> Path:
    """Save evaluation results as JSON."""
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results → {path}")
    return path
