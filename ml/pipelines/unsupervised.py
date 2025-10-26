"""Pipelines for unsupervised anomaly detection training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from ..artifacts import ModelArtifact
from ..models.anomaly import AnomalyDetector


@dataclass
class AnomalyTrainingReport:
    """Summarises an anomaly detection training run."""

    stats: Dict[str, float]
    artifact: ModelArtifact


class AnomalyTrainingPipeline:
    """Wraps anomaly detector training with metadata persistence."""

    def __init__(self, detector: AnomalyDetector, config) -> None:
        self.detector = detector
        self.config = config

    def fit(self, features: np.ndarray, destination: Path, *, params_override: Dict[str, float] | None = None) -> AnomalyTrainingReport:
        """Train the detector and save the resulting artefact."""
        stats = self.detector.train(features, params_override=params_override)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.detector.save(destination)

        artifact = ModelArtifact(
            name="anomaly_detector",
            version=getattr(self.config, "APP_VERSION", "0"),
            path=destination,
            features=list(getattr(self.config, "FEATURE_NAMES", [])),
            parameters=dict(self.config.ISOLATION_FOREST_PARAMS),
            metrics=stats,
        )
        return AnomalyTrainingReport(stats=stats, artifact=artifact)
