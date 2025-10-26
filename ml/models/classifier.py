"""Random forest classifier for procrastination detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


class ProcrastinationClassifier:
    """Supervised random forest classifier for FocusGuard."""

    def __init__(self, config) -> None:
        self.config = config
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False
        self.feature_importance: Optional[pd.Series] = None
        self.calibrator: Optional[CalibratedClassifierCV] = None

    def train(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Fit the classifier and report training diagnostics."""
        logger.info("Training procrastination classifier...")
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(features)

        self.model = RandomForestClassifier(**self.config.RANDOM_FOREST_PARAMS)
        self.model.fit(scaled, labels)
        self.is_fitted = True
        self.calibrator = None

        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.config.FEATURE_NAMES,
        ).sort_values(ascending=False)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, scaled, labels, cv=cv, scoring="f1")

        y_pred = self.model.predict(scaled)
        y_prob = self.model.predict_proba(scaled)[:, 1]

        try:
            auc = roc_auc_score(labels, y_prob)
        except ValueError:
            auc = 0.0

        stats: Dict[str, float] = {
            "n_samples": float(len(features)),
            "n_features": float(features.shape[1]),
            "n_distractions": float(labels.sum()),
            "n_normal": float((labels == 0).sum()),
            "cv_mean_f1": float(cv_scores.mean()),
            "cv_std_f1": float(cv_scores.std()),
            "auc_score": float(auc),
        }

        stats.update({f"feature::{k}": float(v) for k, v in self.feature_importance.head(5).items()})
        logger.info("Classifier training summary: %s", stats)
        return stats

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return class labels (0|1) and probabilities for the distraction class."""
        if not self.is_fitted or self.model is None or self.scaler is None:
            raise RuntimeError("Classifier is not trained")

        scaled = self.scaler.transform(features)
        predictor = self.calibrator if self.calibrator is not None else self.model
        labels = predictor.predict(scaled)
        probabilities = predictor.predict_proba(scaled)[:, 1]
        return labels, probabilities

    def calibrate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calibrate probability outputs using a validation split."""
        if not self.is_fitted or self.model is None or self.scaler is None:
            raise RuntimeError("Classifier must be trained before calibration")

        scaled = self.scaler.transform(features)
        calibrator = CalibratedClassifierCV(self.model, cv="prefit", method="sigmoid")
        calibrator.fit(scaled, labels)
        self.calibrator = calibrator

        calibrated_probs = calibrator.predict_proba(scaled)[:, 1]
        brier = brier_score_loss(labels, calibrated_probs)
        logger.info("Classifier calibration Brier score: %.4f", brier)
        return {"calibration_brier": float(brier)}

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """Expose feature importances for interpretability."""
        if self.feature_importance is None:
            return {}
        return {k: float(v) for k, v in self.feature_importance.head(top_n).items()}

    def save(self, model_path: Path, scaler_path: Path, calibrator_path: Optional[Path] = None) -> None:
        """Persist classifier and scaler."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("No classifier artefacts to save")
        joblib.dump(self.model, str(model_path))
        joblib.dump(self.scaler, str(scaler_path))
        if calibrator_path is not None:
            if self.calibrator is not None:
                joblib.dump(self.calibrator, str(calibrator_path))
            elif calibrator_path.exists():
                calibrator_path.unlink()
        logger.info("Classifier saved to %s and scaler to %s", model_path, scaler_path)

    def load(self, model_path: Path, scaler_path: Path, calibrator_path: Optional[Path] = None) -> None:
        """Load classifier and scaler from disk."""
        self.model = joblib.load(str(model_path))
        self.scaler = joblib.load(str(scaler_path))
        self.calibrator = None
        if calibrator_path is not None and calibrator_path.exists():
            self.calibrator = joblib.load(str(calibrator_path))
        self.is_fitted = True
        logger.info("Classifier loaded from %s", model_path)
