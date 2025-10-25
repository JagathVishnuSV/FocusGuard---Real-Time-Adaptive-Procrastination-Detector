"""Backward-compatible shim for the legacy ``ml_model`` module.

The original project exposed ``AnomalyDetector``, ``ProcrastinationClassifier``
and ``ModelEnsemble`` from ``ml_model.py``. These implementations now live in
the structured ``ml`` package. This module simply re-exports the modern
classes so existing imports continue to function.
"""

from __future__ import annotations

from ml.ensembles.focus_guard import FocusGuardEnsemble
from ml.models.anomaly import AnomalyDetector
from ml.models.classifier import ProcrastinationClassifier

# Preserve the legacy name that other modules expect.
ModelEnsemble = FocusGuardEnsemble

__all__ = [
    "AnomalyDetector",
    "ProcrastinationClassifier",
    "ModelEnsemble",
    "FocusGuardEnsemble",
]