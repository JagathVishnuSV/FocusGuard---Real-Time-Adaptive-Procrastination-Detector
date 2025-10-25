"""Artifact metadata structures and helpers for FocusGuard ML assets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


@dataclass
class ModelArtifact:
    """Describes a persisted machine learning asset."""

    name: str
    version: str
    path: Path
    features: List[str] = field(default_factory=list)
    trained_at: datetime = field(default_factory=datetime.utcnow)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the artifact metadata for logging or persistence."""
        return {
            "name": self.name,
            "version": self.version,
            "path": str(self.path),
            "features": list(self.features),
            "trained_at": self.trained_at.strftime(ISO_FORMAT),
            "parameters": dict(self.parameters),
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelArtifact":
        """Rebuild an artifact description from serialized state."""
        return cls(
            name=payload.get("name", "unknown"),
            version=payload.get("version", "0"),
            path=Path(payload.get("path", "")),
            features=list(payload.get("features", [])),
            trained_at=datetime.strptime(payload["trained_at"], ISO_FORMAT)
            if payload.get("trained_at")
            else datetime.utcnow(),
            parameters=dict(payload.get("parameters", {})),
            metrics=dict(payload.get("metrics", {})),
            metadata=dict(payload.get("metadata", {})),
        )


def write_metadata(artifacts: Iterable[ModelArtifact], destination: Path) -> None:
    """Persist a collection of artifacts to disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump([artifact.to_dict() for artifact in artifacts], handle, indent=2)


def read_metadata(source: Path) -> List[ModelArtifact]:
    """Load artifact metadata from disk if the file exists."""
    if not source.exists():
        return []
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [ModelArtifact.from_dict(item) for item in payload]
