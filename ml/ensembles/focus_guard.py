"""FocusGuard ensemble that combines anomaly detection and classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..models import AnomalyDetector, ProcrastinationClassifier
from ..models.combiner import EnsembleCombiner
from ..pipelines import AnomalyTrainingPipeline, ClassificationTrainingPipeline

logger = logging.getLogger(__name__)


class FocusGuardEnsemble:
    """Coordinates anomaly detection and supervised classification."""

    def __init__(
        self,
        config,
        anomaly_detector: Optional[AnomalyDetector] = None,
        classifier: Optional[ProcrastinationClassifier] = None,
        combiner: Optional[EnsembleCombiner] = None,
    ) -> None:
        self.config = config
        self.anomaly_detector = anomaly_detector or AnomalyDetector(config)
        self.classifier = classifier or ProcrastinationClassifier(config)
        self.combiner = combiner or EnsembleCombiner()
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
        calibrator_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        """Train the supervised classifier; optionally persist artefacts."""
        if model_path and scaler_path:
            if calibrator_path is None:
                calibrator_path = model_path.with_name("classifier_calibrator.joblib")
            report = self.classification_pipeline.fit(
                features,
                labels,
                model_path,
                scaler_path,
                calibrator_path,
            )
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
                base_path / "classifier_calibrator.joblib",
            )

        combiner_path = base_path / "combiner.joblib"
        if self.combiner is not None and self.combiner.is_fitted:
            self.combiner.save(combiner_path)
        elif combiner_path.exists():
            combiner_path.unlink()

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
                base_path / "classifier_calibrator.joblib",
            )
            self.use_classifier = True
        except FileNotFoundError:
            logger.info("Classifier artefacts not available; continuing with anomaly detector only")

        combiner_path = base_path / "combiner.joblib"
        if combiner_path.exists():
            try:
                self.combiner.load(combiner_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to load combiner: %s", exc)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Run the ensemble and return all intermediate signals."""
        results: Dict[str, np.ndarray] = {}

        anomaly_labels, anomaly_scores, anomaly_raw = self.anomaly_detector.predict(features, return_raw=True)
        results["anomaly_prediction"] = anomaly_labels
        results["anomaly_score"] = anomaly_scores
        results["anomaly_score_raw"] = anomaly_raw

        if self.use_classifier and self.classifier.is_fitted:
            clf_labels, clf_probs = self.classifier.predict(features)
            results["classifier_prediction"] = clf_labels
            results["classifier_probability"] = clf_probs

            if self.combiner is not None and self.combiner.is_fitted:
                combiner_outputs = self.combiner.predict(clf_probs, anomaly_scores)
                results["combined_score"] = combiner_outputs["combined_probability"]
                results["is_procrastinating"] = combiner_outputs["combined_label"]
                results["combiner_threshold"] = np.full_like(
                    combiner_outputs["combined_probability"],
                    self.combiner.threshold,
                    dtype=float,
                )
            else:
                combined = 0.3 * anomaly_scores + 0.7 * clf_probs
                results["combined_score"] = combined
                results["is_procrastinating"] = (combined > 0.5).astype(int)
        else:
            results["is_procrastinating"] = (anomaly_labels == -1).astype(int)

        return results


# Backwards compatibility -------------------------------------------------
ModelEnsemble = FocusGuardEnsemble
