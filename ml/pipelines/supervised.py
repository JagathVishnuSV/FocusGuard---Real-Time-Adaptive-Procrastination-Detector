"""Pipelines for supervised procrastination classification training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from ..artifacts import ModelArtifact
from ..models.classifier import ProcrastinationClassifier


@dataclass
class ClassificationTrainingReport:
    """Summarises a classifier training run."""

    stats: Dict[str, float]
    artifact: ModelArtifact


class ClassificationTrainingPipeline:
    """Wraps classifier training while recording metadata."""

    def __init__(self, classifier: ProcrastinationClassifier, config) -> None:
        self.classifier = classifier
        self.config = config

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        model_path: Path,
        scaler_path: Path,
        calibrator_path: Path,
    ) -> ClassificationTrainingReport:
        """Train the classifier and persist the artefacts."""
        stats = self.classifier.train(features, labels)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.classifier.save(model_path, scaler_path, calibrator_path)

        artifact = ModelArtifact(
            name="procrastination_classifier",
            version=getattr(self.config, "APP_VERSION", "0"),
            path=model_path,
            features=list(getattr(self.config, "FEATURE_NAMES", [])),
            parameters=dict(self.config.RANDOM_FOREST_PARAMS),
            metrics=stats,
            metadata={
                "scaler_path": str(scaler_path),
                "calibrator_path": str(calibrator_path),
            },
        )
        return ClassificationTrainingReport(stats=stats, artifact=artifact)
