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

import re
from collections import deque, defaultdict
from typing import Deque, Dict, Iterable, Optional
from urllib.parse import urlparse

try:
    # Import type only if available in runtime; avoids hard dependency for core repo
    from activity_stream import ActivityEvent
except Exception:  # pragma: no cover - defensive
    ActivityEvent = object


TITLE_SUFFIX_RE = re.compile(
    r"\s+[-–—]\s+(Google Chrome|Mozilla Firefox|Microsoft Edge|Brave|Opera|Safari|Visual Studio Code)$",
    flags=re.IGNORECASE,
)

PLATFORM_SUFFIX_RE = re.compile(
    r"\s+[-–—]\s+(YouTube|Netflix|Twitch|Spotify|ChatGPT|Gmail|Notion|Figma)$",
    flags=re.IGNORECASE,
)

SITE_ALIAS_MAP = {
    "youtube": "youtube.com",
    "netflix": "netflix.com",
    "twitch": "twitch.tv",
    "spotify": "spotify.com",
    "chatgpt": "chatgpt.com",
    "openai": "openai.com",
    "gmail": "mail.google.com",
    "drive": "drive.google.com",
    "calendar": "calendar.google.com",
    "notion": "notion.so",
    "figma": "figma.com",
}

MEDIA_DOMAINS = {
    "youtube.com",
    "twitch.tv",
    "netflix.com",
    "spotify.com",
    "discord.com",
}

MAX_DETAIL_LENGTH = 42


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
        self._browser_tokens = ("chrome", "firefox", "msedge", "safari", "opera", "edge", "brave")
        self._token_display: Dict[str, str] = {}
        self._last_prediction_version = -1

    @staticmethod
    def _clean_title(title: Optional[str]) -> str:
        if not title:
            return ""
        cleaned = TITLE_SUFFIX_RE.sub("", title)
        cleaned = PLATFORM_SUFFIX_RE.sub("", cleaned)
        return cleaned.strip()

    @staticmethod
    def _extract_host(value: Optional[str]) -> str:
        if not value:
            return ""
        candidate = value.strip()
        if not candidate:
            return ""
        if "://" not in candidate:
            candidate = f"https://{candidate}"
        try:
            parsed = urlparse(candidate)
            host = parsed.netloc or parsed.path
        except Exception:
            host = value
        host = host.lower().strip()
        if host.startswith("www."):
            host = host[4:]
        return host.rstrip("/")

    @staticmethod
    def _guess_host_from_title(title: str) -> str:
        lowered = title.lower()
        for alias, host in SITE_ALIAS_MAP.items():
            if alias in lowered:
                return host
        return ""

    def _register_token_display(self, token: str, *, detail: str, host: str, fallback: str) -> None:
        if not token:
            return
        host_display = host or fallback or "unknown"
        trimmed_detail = (detail or "").strip()
        if len(trimmed_detail) > MAX_DETAIL_LENGTH:
            trimmed_detail = trimmed_detail[: MAX_DETAIL_LENGTH - 1].rstrip() + "…"

        if host_display in MEDIA_DOMAINS:
            label = host_display
        elif trimmed_detail and trimmed_detail.lower() != host_display.lower():
            label = f"{trimmed_detail} · {host_display}"
        else:
            label = host_display or trimmed_detail or fallback or "Unknown"

        self._token_display[token] = label

    def _prettify_token(self, token: Optional[str]) -> str:
        if not token:
            return "Unknown"
        cached = self._token_display.get(token)
        if cached:
            return cached
        if "::" in token:
            host, detail = token.split("::", 1)
            host = host.strip()
            detail = detail.strip()
            if host and detail:
                return f"{detail} · {host}"
            return detail or host or "Unknown"
        return token or "Unknown"

    def _normalize_app(self, app_name: Optional[str], window_title: Optional[str] = "", url: Optional[str] = None) -> str:
        fallback = (app_name or "unknown").strip() or "unknown"
        app_token = fallback.lower()
        cleaned_title = self._clean_title(window_title)
        host = self._extract_host(url) if url else ""

        if not host and any(browser in app_token for browser in self._browser_tokens):
            host = self._guess_host_from_title(cleaned_title) or app_token
        elif not host and cleaned_title:
            host = self._guess_host_from_title(cleaned_title)

        detail = cleaned_title or fallback
        normalized_host = (host.lower().strip() if host else app_token) or app_token
        normalized_detail = detail.lower().strip()

        if normalized_host in MEDIA_DOMAINS:
            normalized_detail = ""
        elif normalized_detail == normalized_host:
            normalized_detail = ""

        token = f"{normalized_host}::{normalized_detail}" if normalized_detail else normalized_host
        self._register_token_display(token, detail=detail, host=host or normalized_host, fallback=fallback)
        return token

    def update(self, events: Iterable[object]):
        """Update internal history and transitions using a sequence of ActivityEvent objects."""
        for e in events:
            try:
                app = self._normalize_app(
                    getattr(e, "app_name", None),
                    getattr(e, "window_title", ""),
                    getattr(e, "url", None),
                )
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

        snapshot_stale = self._history_updates == self._last_prediction_version
        self._last_prediction_version = self._history_updates

        return {
            "predicted_next": self._prettify_token(next_candidate),
            "prob_distracted": round(float(prob), 3),
            "support": int(support),
            "last_app": self._prettify_token(last_app),
            "history_size": len(self.history),
            "transitions_observed": self._history_updates,
            "horizon_seconds": horizon_seconds or getattr(self.config, "GHOST_PREDICT_HORIZON_SECONDS", 60),
            "is_stale": snapshot_stale,
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
