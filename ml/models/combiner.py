"""Meta-model for combining classifier probability and anomaly score."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score


@dataclass
class CombinerReport:
    """Metrics describing a combiner training run."""

    stats: Dict[str, float]


class EnsembleCombiner:
    """Learns how to blend classifier probabilities with anomaly scores."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.model: Optional[LogisticRegression] = None
        self.threshold: float = threshold

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def train(self, classifier_probs: np.ndarray, anomaly_scores: np.ndarray, labels: np.ndarray) -> CombinerReport:
        if classifier_probs.shape != anomaly_scores.shape:
            raise ValueError("Classifier probabilities and anomaly scores must share shape")

        features = np.column_stack([classifier_probs, anomaly_scores])
        model = LogisticRegression(max_iter=200, solver="lbfgs")
        model.fit(features, labels)
        self.model = model

        probabilities = self.model.predict_proba(features)[:, 1]
        preds = (probabilities >= 0.5).astype(int)
        auc = roc_auc_score(labels, probabilities)
        f1 = f1_score(labels, preds)

        stats = {
            "auc": float(auc),
            "f1": float(f1),
        }

        # Simple threshold tuning: keep 0.5 unless a better F1 exists scanning coarse grid
        candidate_thresholds = np.linspace(0.3, 0.7, num=9)
        best_threshold = 0.5
        best_f1 = f1
        for thr in candidate_thresholds:
            candidate_preds = (probabilities >= thr).astype(int)
            candidate_f1 = f1_score(labels, candidate_preds)
            if candidate_f1 > best_f1:
                best_f1 = candidate_f1
                best_threshold = float(thr)
        self.threshold = best_threshold
        stats["threshold"] = self.threshold
        stats["threshold_f1"] = float(best_f1)

        return CombinerReport(stats=stats)

    def predict_proba(self, classifier_probs: np.ndarray, anomaly_scores: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Combiner is not trained")
        features = np.column_stack([classifier_probs, anomaly_scores])
        return self.model.predict_proba(features)[:, 1]

    def predict(self, classifier_probs: np.ndarray, anomaly_scores: np.ndarray) -> Dict[str, np.ndarray]:
        probabilities = self.predict_proba(classifier_probs, anomaly_scores)
        labels = (probabilities >= self.threshold).astype(int)
        return {
            "combined_probability": probabilities,
            "combined_label": labels,
        }

    def save(self, target: Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Combiner is not trained")
        payload = {
            "model": self.model,
            "threshold": self.threshold,
        }
        joblib.dump(payload, str(target))

    def load(self, source: Path) -> None:
        payload = joblib.load(str(source))
        self.model = payload.get("model")
        self.threshold = float(payload.get("threshold", 0.5))
