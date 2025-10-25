"""FocusGuard ensemble that combines anomaly detection and classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..models import AnomalyDetector, ProcrastinationClassifier
from ..pipelines import AnomalyTrainingPipeline, ClassificationTrainingPipeline

logger = logging.getLogger(__name__)


class FocusGuardEnsemble:
    """Coordinates anomaly detection and supervised classification."""

    def __init__(
        self,
        config,
        anomaly_detector: Optional[AnomalyDetector] = None,
        classifier: Optional[ProcrastinationClassifier] = None,
    ) -> None:
        self.config = config
        self.anomaly_detector = anomaly_detector or AnomalyDetector(config)
        self.classifier = classifier or ProcrastinationClassifier(config)
        self.anomaly_pipeline = AnomalyTrainingPipeline(self.anomaly_detector, config)
        self.classification_pipeline = ClassificationTrainingPipeline(self.classifier, config)
        self.use_classifier = False

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def train_baseline(self, features: np.ndarray, *, artifact_path: Optional[Path] = None) -> Dict[str, float]:
        """Train the unsupervised detector; optionally persist artefacts."""
        if artifact_path is not None:
            report = self.anomaly_pipeline.fit(features, artifact_path)
            return report.stats
        return self.anomaly_detector.train(features)

    def train_classifier(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        """Train the supervised classifier; optionally persist artefacts."""
        if model_path and scaler_path:
            report = self.classification_pipeline.fit(features, labels, model_path, scaler_path)
            self.use_classifier = True
            return report.stats

        stats = self.classifier.train(features, labels)
        self.use_classifier = True
        return stats

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, base_path: Path) -> None:
        """Persist the trained ensemble components."""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        self.anomaly_detector.save(base_path / "anomaly_detector.joblib")
        if self.use_classifier and self.classifier.is_fitted:
            self.classifier.save(
                base_path / "classifier.joblib",
                base_path / "scaler.joblib",
            )

    def load(self, base_path: Path) -> None:
        """Restore ensemble components if the files are available."""
        base_path = Path(base_path)
        try:
            self.anomaly_detector.load(base_path / "anomaly_detector.joblib")
        except FileNotFoundError:
            logger.warning("Anomaly detector not found in %s", base_path)

        try:
            self.classifier.load(
                base_path / "classifier.joblib",
                base_path / "scaler.joblib",
            )
            self.use_classifier = True
        except FileNotFoundError:
            logger.info("Classifier artefacts not available; continuing with anomaly detector only")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Run the ensemble and return all intermediate signals."""
        results: Dict[str, np.ndarray] = {}

        labels, scores = self.anomaly_detector.predict(features)
        results["anomaly_prediction"] = labels
        results["anomaly_score"] = scores

        if self.use_classifier and self.classifier.is_fitted:
            clf_labels, clf_probs = self.classifier.predict(features)
            results["classifier_prediction"] = clf_labels
            results["classifier_probability"] = clf_probs

            # Normalise anomaly scores defensively before combining
            norm_scores = self._normalise_scores(scores)
            combined = 0.3 * norm_scores + 0.7 * clf_probs
            results["combined_score"] = combined
            results["is_procrastinating"] = (combined > 0.5).astype(int)
        else:
            results["is_procrastinating"] = (labels == -1).astype(int)

        return results

    @staticmethod
    def _normalise_scores(scores: np.ndarray) -> np.ndarray:
        """Scale anomaly scores to 0-1 without division by zero."""
        if len(scores) == 0:
            return scores
        minimum = float(np.min(scores))
        maximum = float(np.max(scores))
        if np.isclose(maximum, minimum):
            return np.clip(scores, 0.0, 1.0)
        return (scores - minimum) / (maximum - minimum)


# Backwards compatibility -------------------------------------------------
ModelEnsemble = FocusGuardEnsemble
