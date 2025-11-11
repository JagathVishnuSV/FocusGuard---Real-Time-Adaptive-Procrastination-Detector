"""Lightweight Cognitive Twin (Ghost) predictor.

This module provides a small, flexible 'ghost' that simulates the user's
likely next action (next app or domain) and a simple heuristic probability
that the user will be distracted in the near future. The implementation is
intended to be easy to replace with a learned model later.

API:
  GhostTwin(config)
    .update(events) -> updates internal history with recent ActivityEvent objects
    .predict(events, feature_map, horizon_seconds=60) -> dict with predictions

Notes:
- Only uses a compact transition-frequency model + simple heuristics so no
  heavy dependencies are required. This keeps the change minimal and allows
  iterative replacement with ML models (sentence-transformers, TF.js, etc.).
"""

from __future__ import annotations

from collections import deque, defaultdict
from typing import Deque, Dict, Iterable, Optional
from urllib.parse import urlparse

try:
    # Import type only if available in runtime; avoids hard dependency for core repo
    from activity_stream import ActivityEvent
except Exception:  # pragma: no cover - defensive
    ActivityEvent = object


class GhostTwin:
    """A small cognitive twin that predicts a likely next app and distraction probability.

    Implementation:
      - Maintains a recent history of app_names and simple transition counts
      - Heuristic distraction probability computed from feature_map
    """

    def __init__(self, config):
        self.config = config
        self.history: Deque[str] = deque(maxlen=getattr(config, "GHOST_TWIN_HISTORY", 128))
        # transition_counts[prev_app][next_app] = count
        self.transition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._history_updates = 0

    def _normalize_app(self, app_name: Optional[str], window_title: Optional[str] = "") -> str:
        if not app_name:
            return "unknown"
        name = str(app_name).lower()
        # For browser windows, try to extract a domain from the window_title
        if any(browser in name for browser in ["chrome", "firefox", "msedge", "safari", "opera"]):
            text = (window_title or "").lower()
            # crude parse of a domain-like token in the title
            for token in text.split():
                if "." in token and len(token) > 3:
                    try:
                        parsed = urlparse(token if token.startswith("http") else f"https://{token}")
                        host = parsed.netloc or token
                        return host.lower()
                    except Exception:
                        continue
        return name

    def update(self, events: Iterable[object]):
        """Update internal history and transitions using a sequence of ActivityEvent objects."""
        for e in events:
            try:
                app = self._normalize_app(getattr(e, "app_name", None), getattr(e, "window_title", ""))
            except Exception:
                app = "unknown"

            if not self.history or self.history[-1] != app:
                # register transition
                if self.history:
                    prev_app = self.history[-1]
                    self.transition_counts[prev_app][app] += 1
                self.history.append(app)
                self._history_updates += 1

    def predict(self, events: Optional[Iterable[object]] = None, feature_map: Optional[Dict[str, float]] = None, horizon_seconds: Optional[int] = None) -> Dict[str, object]:
        """Return a small prediction dict describing likely next app and distraction prob.

        Args:
            events: optional recent events to incorporate before prediction
            feature_map: feature vector (dict) produced by FeatureExtractor for heuristics
            horizon_seconds: how far ahead to predict (unused in heuristics but part of API)

        Returns:
            {
              "predicted_next": str,  # predicted next app or domain token
              "prob_distracted": float,  # [0-1]
              "support": int,  # number of transitions supporting the predicted_next
            }
        """
        if events:
            try:
                self.update(events)
            except Exception:
                pass

        last_app = self.history[-1] if self.history else "unknown"

        # transition-based next prediction
        next_candidate = "unknown"
        support = 0
        if last_app in self.transition_counts:
            counts = self.transition_counts[last_app]
            if counts:
                next_candidate, support = max(counts.items(), key=lambda kv: kv[1])

        # fallback: most common in history
        if (not next_candidate or next_candidate == "unknown") and self.history:
            freq = defaultdict(int)
            for a in self.history:
                freq[a] += 1
            next_candidate = max(freq.items(), key=lambda kv: kv[1])[0]

        # Heuristic distraction probability
        prob = 0.0
        if feature_map:
            try:
                distraction_ratio = float(feature_map.get("distraction_app_ratio", 0.0) or 0.0)
                productive_ratio = float(feature_map.get("productive_app_ratio", 0.0) or 0.0)
                keystrokes = float(feature_map.get("keystrokes_per_sec", 0.0) or 0.0)
                switch_freq = float(feature_map.get("app_switch_frequency", 0.0) or 0.0)

                # base contribution from site categorization
                prob = 0.5 * distraction_ratio

                # low typing activity increases distraction probability
                prob += 0.3 * (1.0 - min(keystrokes / 1.0, 1.0))

                # high app switching increases chance of imminent distraction
                prob += 0.2 * min(switch_freq * 10.0, 1.0)

                # reduce if clearly productive
                if productive_ratio >= 0.7 and distraction_ratio <= 0.2:
                    prob = max(0.0, prob - 0.35)

            except Exception:
                prob = 0.0

        # clamp
        prob = max(0.0, min(1.0, prob))

        return {
            "predicted_next": next_candidate,
            "prob_distracted": round(float(prob), 3),
            "support": int(support),
            "last_app": last_app,
            "history_size": len(self.history),
            "transitions_observed": self._history_updates,
            "horizon_seconds": horizon_seconds or getattr(self.config, "GHOST_PREDICT_HORIZON_SECONDS", 60),
        }

    # small utility to allow later ML-driven training
    def record_transition(self, prev_app: str, next_app: str):
        """Record a single transition pair (useful for online updates)."""
        self.transition_counts[prev_app][next_app] += 1

    def reset(self):
        """Clear cached history and transition counts."""
        self.history.clear()
        self.transition_counts.clear()
        self._history_updates = 0
