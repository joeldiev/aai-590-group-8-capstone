"""
Mahalanobis-based Out-of-Distribution (OOD) detector.

Operates on [CLS] embeddings from the fine-tuned RoBERTa model.
Pipeline: PCA dimensionality reduction → per-class Gaussian fit → Mahalanobis distance scoring.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


class MahalanobisOODDetector:
    """Mahalanobis distance-based anomaly/OOD detector.

    Fit on in-distribution [CLS] embeddings with known labels.
    Score new embeddings by distance to nearest class centroid.
    """

    def __init__(self, n_components: int = 100):
        self.n_components = n_components
        self.pca = None
        self.class_means = {}      # label → mean vector
        self.shared_precision = None  # inverse of shared covariance
        self.threshold = None
        self._fitted = False

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> "MahalanobisOODDetector":
        """Fit the detector on in-distribution embeddings.

        Args:
            embeddings: (N, D) array of [CLS] embeddings.
            labels: (N,) array of integer class labels.

        Returns:
            self
        """
        # PCA reduction
        self.pca = PCA(n_components=min(self.n_components, embeddings.shape[1]))
        reduced = self.pca.fit_transform(embeddings)

        # Per-class means
        unique_labels = np.unique(labels)
        for lab in unique_labels:
            mask = labels == lab
            self.class_means[int(lab)] = reduced[mask].mean(axis=0)

        # Shared covariance (pooled across classes)
        centered = np.zeros_like(reduced)
        for lab in unique_labels:
            mask = labels == lab
            centered[mask] = reduced[mask] - self.class_means[int(lab)]

        cov = np.cov(centered.T)

        # Regularize covariance to avoid singularity
        cov += np.eye(cov.shape[0]) * 1e-6

        try:
            self.shared_precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            print("[anomaly] WARNING: Covariance matrix singular — using pseudo-inverse")
            self.shared_precision = np.linalg.pinv(cov)

        self._fitted = True
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance to nearest class centroid.

        Args:
            embeddings: (N, D) array of [CLS] embeddings.

        Returns:
            (N,) array of Mahalanobis distances (lower = more in-distribution).
        """
        assert self._fitted, "Detector must be fitted before scoring"

        reduced = self.pca.transform(embeddings)

        # Distance to each class centroid
        distances = []
        for lab, mean in self.class_means.items():
            diff = reduced - mean
            maha = np.sum(diff @ self.shared_precision * diff, axis=1)
            distances.append(maha)

        # Min distance across classes (closest centroid)
        distances = np.stack(distances, axis=1)
        return distances.min(axis=1)

    def calibrate_threshold(
        self,
        val_embeddings: np.ndarray,
        recall_target: float = 0.95,
    ) -> float:
        """Set threshold so that `recall_target` fraction of in-distribution
        validation samples are below the threshold.

        Args:
            val_embeddings: (N, D) in-distribution validation embeddings.
            recall_target: Fraction of ID samples to keep (default: 0.95).

        Returns:
            Calibrated threshold value.
        """
        scores = self.score(val_embeddings)
        self.threshold = float(np.percentile(scores, recall_target * 100))
        print(f"[anomaly] Calibrated threshold: {self.threshold:.4f} "
              f"({recall_target*100:.0f}% ID recall)")
        return self.threshold

    def predict_ood(
        self,
        embeddings: np.ndarray,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Predict whether each sample is OOD.

        Args:
            embeddings: (N, D) array of [CLS] embeddings.
            threshold: If None, uses calibrated threshold.

        Returns:
            (N,) boolean array — True = OOD.
        """
        if threshold is None:
            threshold = self.threshold
        assert threshold is not None, "Must calibrate or provide threshold"

        scores = self.score(embeddings)
        return scores > threshold

    def save(self, path: str | Path) -> None:
        """Save detector to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "mahalanobis_detector.pkl", "wb") as f:
            pickle.dump(self, f)
        print(f"[anomaly] Saved detector → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "MahalanobisOODDetector":
        """Load detector from disk."""
        path = Path(path)
        with open(path / "mahalanobis_detector.pkl", "rb") as f:
            return pickle.load(f)
