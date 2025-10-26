"""Isolation Forest based anomaly detector for FocusGuard."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Unsupervised anomaly detection using Isolation Forest."""

    def __init__(self, config) -> None:
        self.config = config
        self.model: Optional[IsolationForest] = None
        self.feature_scaler: Optional[StandardScaler] = None
        self.score_normalizer: Optional[MinMaxScaler] = None
        self.is_fitted: bool = False

    def train(self, features: np.ndarray, *, params_override: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Fit the isolation forest and return training diagnostics."""
        logger.info("Training anomaly detection model...")
        self.feature_scaler = StandardScaler()
        scaled_features = self.feature_scaler.fit_transform(features)

        params = dict(self.config.ISOLATION_FOREST_PARAMS)
        if params_override:
            params.update(params_override)

        self.model = IsolationForest(**params)
        self.model.fit(scaled_features)
        self.is_fitted = True

        raw_scores = -self.model.score_samples(scaled_features)
        self.score_normalizer = MinMaxScaler()
        self.score_normalizer.fit(raw_scores.reshape(-1, 1))

        predictions = self.model.predict(scaled_features)
        n_anomalies = int((predictions == -1).sum())
        stats = {
            "n_samples": float(len(features)),
            "n_features": float(features.shape[1]),
            "n_anomalies": float(n_anomalies),
            "n_normal": float(len(features) - n_anomalies),
            "anomaly_ratio": float(n_anomalies / len(features)) if len(features) else 0.0,
            "score_min": float(raw_scores.min()) if len(raw_scores) else 0.0,
            "score_max": float(raw_scores.max()) if len(raw_scores) else 0.0,
            "score_mean": float(raw_scores.mean()) if len(raw_scores) else 0.0,
            "score_std": float(raw_scores.std()) if len(raw_scores) else 0.0,
        }
        logger.info("Anomaly detector training summary: %s", stats)
        return stats

    def _transform_features(self, features: np.ndarray) -> np.ndarray:
        if self.feature_scaler is None:
            return features
        return self.feature_scaler.transform(features)

    def _transform_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        if self.score_normalizer is None:
            return raw_scores
        return self.score_normalizer.transform(raw_scores.reshape(-1, 1)).ravel()

    def predict(self, features: np.ndarray, *, return_raw: bool = False) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return anomaly labels and normalised scores (optionally raw scores)."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Anomaly detector is not trained")

        transformed = self._transform_features(features)
        labels = self.model.predict(transformed)
        raw_scores = -self.model.score_samples(transformed)
        scores = self._transform_scores(raw_scores)

        if return_raw:
            return labels, scores, raw_scores
        return labels, scores

    def save(self, target: Path) -> None:
        """Persist the fitted model to disk."""
        if self.model is None:
            raise RuntimeError("No anomaly detector to save")
        payload = {
            "model": self.model,
            "feature_scaler": self.feature_scaler,
            "score_normalizer": self.score_normalizer,
        }
        joblib.dump(payload, str(target))
        logger.info("Anomaly detector saved to %s", target)

    def load(self, source: Path) -> None:
        """Load a previously fitted model from disk."""
        loaded = joblib.load(str(source))

        if isinstance(loaded, dict):
            self.model = loaded.get("model")
            self.feature_scaler = loaded.get("feature_scaler")
            self.score_normalizer = loaded.get("score_normalizer")
        else:
            # Backwards compatibility with earlier artefacts containing only the model
            self.model = loaded
            self.feature_scaler = None
            self.score_normalizer = None

        self.is_fitted = self.model is not None
        logger.info("Anomaly detector loaded from %s", source)
