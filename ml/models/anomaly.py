"""Isolation Forest based anomaly detector for FocusGuard."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Unsupervised anomaly detection using Isolation Forest."""

    def __init__(self, config) -> None:
        self.config = config
        self.model: Optional[IsolationForest] = None
        self.is_fitted: bool = False

    def train(self, features: np.ndarray) -> Dict[str, float]:
        """Fit the isolation forest and return training diagnostics."""
        logger.info("Training anomaly detection model...")
        self.model = IsolationForest(**self.config.ISOLATION_FOREST_PARAMS)
        self.model.fit(features)
        self.is_fitted = True

        predictions = self.model.predict(features)
        n_anomalies = int((predictions == -1).sum())
        stats = {
            "n_samples": float(len(features)),
            "n_features": float(features.shape[1]),
            "n_anomalies": float(n_anomalies),
            "n_normal": float(len(features) - n_anomalies),
            "anomaly_ratio": float(n_anomalies / len(features)) if len(features) else 0.0,
        }
        logger.info("Anomaly detector training summary: %s", stats)
        return stats

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return anomaly labels (-1, 1) and anomaly scores (higher = worse)."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Anomaly detector is not trained")

        labels = self.model.predict(features)
        scores = -self.model.score_samples(features)
        return labels, scores

    def save(self, target: Path) -> None:
        """Persist the fitted model to disk."""
        if self.model is None:
            raise RuntimeError("No anomaly detector to save")
        joblib.dump(self.model, str(target))
        logger.info("Anomaly detector saved to %s", target)

    def load(self, source: Path) -> None:
        """Load a previously fitted model from disk."""
        self.model = joblib.load(str(source))
        self.is_fitted = True
        logger.info("Anomaly detector loaded from %s", source)
