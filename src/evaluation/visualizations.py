"""
Visualization functions for AGL evaluation results.

Generates publication-quality figures for:
  - Confusion matrices
  - F1 comparison bar charts
  - Training/validation loss curves
  - ROC curves (one-vs-rest)
  - Latency histograms
  - OOD detection comparison
  - Dataset composition
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.config import LABEL_NAMES, FIGURES_DIR


plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] = LABEL_NAMES,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
    normalize: bool = True,
) -> plt.Figure:
    """Plot confusion matrix heatmap."""
    present_labels = sorted(set(y_true) | set(y_pred))
    present_names = [label_names[i] for i in present_labels if i < len(label_names)]

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    if normalize:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_display = cm_norm
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=present_names,
        yticklabels=present_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


def plot_f1_comparison(
    results: dict[str, float],
    title: str = "Macro-F1 Comparison",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart comparing macro-F1 across methods."""
    methods = list(results.keys())
    scores = list(results.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, scores, color=sns.color_palette("viridis", len(methods)))

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Macro-F1")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str] = LABEL_NAMES,
    title: str = "ROC Curves (One-vs-Rest)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot per-class ROC curves."""
    present_labels = sorted(set(y_true))
    present_names = [label_names[i] for i in present_labels if i < len(label_names)]
    n_classes = len(present_labels)

    y_bin = label_binarize(y_true, classes=present_labels)
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("husl", n_classes)

    for i, (label_idx, name) in enumerate(zip(present_labels, present_names)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, label_idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    title: str = "Training & Validation Loss",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, "o-", label="Train Loss", color="tab:blue")
    ax.plot(epochs, val_losses, "s-", label="Val Loss", color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


def plot_latency_histogram(
    latencies: np.ndarray,
    title: str = "Inference Latency Distribution",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Histogram of per-prompt inference latencies."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(latencies, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(latencies), color="red", linestyle="--", label=f"Mean: {np.mean(latencies):.1f}ms")
    ax.axvline(np.percentile(latencies, 99), color="orange", linestyle="--", label=f"P99: {np.percentile(latencies, 99):.1f}ms")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


def plot_dataset_composition(
    df,
    title: str = "Dataset Composition",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Stacked bar chart showing samples per source per class."""
    ct = pd.crosstab(df["source"], df["unified_label"])
    import pandas as pd  # noqa: F811

    fig, ax = plt.subplots(figsize=(10, 6))
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_xlabel("Source Dataset")
    ax.set_ylabel("Number of Samples")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    ax.legend(title="Class")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


def plot_ood_comparison(
    msp_scores: np.ndarray,
    maha_scores: np.ndarray,
    ood_labels: np.ndarray,
    title: str = "OOD Detection: MSP vs Mahalanobis",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Compare MSP and Mahalanobis OOD detection via ROC curves."""
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(8, 6))

    for scores, name, color in [
        (msp_scores, "MSP", "tab:blue"),
        (maha_scores, "Mahalanobis", "tab:red"),
    ]:
        fpr, tpr, _ = roc_curve(ood_labels, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUROC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig
