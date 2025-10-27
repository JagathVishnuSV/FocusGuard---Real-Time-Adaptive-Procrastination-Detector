"""Utilities for capturing per-user personalization feedback."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from config import (
    USER_DATA_DIR,
    USER_FEEDBACK_FILE,
    USER_FEATURE_SNAPSHOTS,
    USER_OVERRIDES_FILE,
    USER_PASSIVE_LABELS_FILE,
)

logger = logging.getLogger(__name__)

_overrides_cache: Optional[Dict[str, Any]] = None


@dataclass
class FeedbackRecord:
    """Lightweight container for user-labelled predictions."""

    timestamp: float
    session_id: Optional[str]
    user_label: str
    predicted_label: str
    confidence: Optional[float]
    combined_score: Optional[float]
    classifier_probability: Optional[float]
    anomaly_score: Optional[float]
    heuristic_triggered: Optional[bool]
    features: Optional[Dict[str, Any]] = None
    app_name: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Preserve millisecond precision while staying JSON serialisable.
        payload["timestamp"] = round(self.timestamp, 6)
        return payload


def _append_json_line(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def record_feedback(record: FeedbackRecord) -> None:
    """Persist a feedback record for later personalisation training."""
    _append_json_line(USER_FEEDBACK_FILE, record.to_dict())
    logger.info("Personalisation feedback captured: session=%s label=%s", record.session_id, record.user_label)


def record_feature_snapshot(
    session_id: Optional[str],
    feature_map: Dict[str, Any],
    *,
    source_timestamp: Optional[float] = None,
) -> None:
    """Persist the feature values associated with a prediction for replay training."""
    snapshot = {
        "timestamp": round(source_timestamp or time.time(), 6),
        "session_id": session_id,
        "features": feature_map,
    }
    _append_json_line(USER_FEATURE_SNAPSHOTS, snapshot)


def ensure_storage() -> None:
    """Create the user personalisation directory if it did not exist."""
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_overrides() -> Dict[str, Any]:
    global _overrides_cache
    if _overrides_cache is not None:
        return _overrides_cache

    ensure_storage()
    if USER_OVERRIDES_FILE.exists():
        try:
            with USER_OVERRIDES_FILE.open("r", encoding="utf-8") as handle:
                _overrides_cache = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load personalization overrides: %s", exc)
            _overrides_cache = {}
    else:
        _overrides_cache = {}

    if "apps" not in _overrides_cache:
        _overrides_cache["apps"] = {}
    if "domains" not in _overrides_cache:
        _overrides_cache["domains"] = {}

    return _overrides_cache


def get_override_category(app_name: Optional[str], url: Optional[str] = None) -> Optional[str]:
    """Return user override category ('productive'/'distraction') if configured."""
    overrides = _load_overrides()
    app_key = app_name.lower() if app_name else None
    if app_key and app_key in overrides.get("apps", {}):
        return overrides["apps"][app_key]

    if url:
        domain = url.lower().split("/")[0]
        for candidate in (domain, domain.lstrip("www.")):
            if candidate in overrides.get("domains", {}):
                return overrides["domains"][candidate]

    return None


def record_passive_label(
    session_id: Optional[str],
    label: str,
    *,
    confidence: float,
    reason: str,
    features: Optional[Dict[str, Any]] = None,
    app_name: Optional[str] = None,
) -> None:
    """Persist an automatically inferred label to assist future retraining."""
    payload = {
        "timestamp": round(time.time(), 6),
        "session_id": session_id,
        "label": label,
        "confidence": confidence,
        "reason": reason,
        "features": features or {},
    }
    if app_name:
        payload["app_name"] = app_name
    _append_json_line(USER_PASSIVE_LABELS_FILE, payload)


def upsert_override(*, app_name: Optional[str] = None, domain: Optional[str] = None, category: str) -> None:
    """Persist a user override and refresh cache."""
    if category not in {"productive", "distraction"}:
        raise ValueError("Override category must be 'productive' or 'distraction'")

    overrides = _load_overrides()
    if app_name:
        overrides.setdefault("apps", {})[app_name.lower()] = category
    if domain:
        normalized = domain.lower().lstrip("www.")
        overrides.setdefault("domains", {})[normalized] = category

    ensure_storage()
    try:
        with USER_OVERRIDES_FILE.open("w", encoding="utf-8") as handle:
            json.dump(overrides, handle, indent=2)
    except OSError as exc:
        logger.error("Failed to persist overrides: %s", exc)
        return

    global _overrides_cache
    _overrides_cache = overrides

