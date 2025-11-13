"""Gemini client helpers for FocusGuard."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    value: str
    expires_at: float


class GeminiClient:
    """Thin wrapper around Google Gemini generateContent endpoint.

    The client intentionally keeps a tiny surface area so core logic can call a
    handful of high-level helper methods without knowing transport details.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str],
        model: str,
        base_url: str,
        timeout: float,
        enabled: bool,
        cache_ttl: int = 600,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._enabled = bool(enabled and api_key)
        self._cache_ttl = cache_ttl

        self._cache: Dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @classmethod
    def from_config(cls, config_module: Any) -> "GeminiClient":
        return cls(
            api_key=getattr(config_module, "GEMINI_API_KEY", None),
            model=getattr(config_module, "GEMINI_MODEL_NAME", "gemini-1.5-flash"),
            base_url=getattr(
                config_module, "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/models"
            ),
            timeout=float(getattr(config_module, "GEMINI_TIMEOUT_SECONDS", 8.0)),
            enabled=bool(getattr(config_module, "ENABLE_GEMINI", False) and getattr(config_module, "GEMINI_API_KEY", None)),
            cache_ttl=int(getattr(config_module, "GEMINI_CACHE_TTL_SECONDS", 600)),
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def summarise_context(
        self,
        *,
        app_name: Optional[str],
        window_title: Optional[str],
        url: Optional[str],
        context_label: Optional[str],
        context_confidence: Optional[float],
    ) -> Optional[str]:
        """Return a short friendly description of the current context."""
        if not self.is_enabled:
            return None

        facts = {
            "app_name": app_name or "unknown",
            "window_title": window_title or "",
            "url": url or "",
            "context_label": context_label or "",
            "context_confidence": context_confidence,
        }
        prompt = (
            "You are FocusGuard, a productivity companion."
            " Describe the user's current context in one short, encouraging sentence"
            " (max 25 words). Highlight intent when possible (e.g. 'researching', 'video browsing')."
            " Avoid guessing brands if uncertain. Data:\n"
            f"{json.dumps(facts, ensure_ascii=False)}"
        )
        return self._normalise_text(
            self._invoke(prompt, temperature=0.25, max_output_tokens=96, cache_ns="context")
        )

    def generate_focus_insight(
        self,
        *,
        stats_today: Dict[str, Any],
        weekly_trend: Any,
        hourly_pattern: Any,
        top_distractions: Dict[str, Any],
    ) -> Optional[str]:
        """Produce a high-level summary insight for the insights panel."""
        if not self.is_enabled:
            return None

        payload = {
            "today": stats_today,
            "weekly_trend": weekly_trend,
            "hourly_pattern": hourly_pattern,
            "top_distractions": top_distractions,
        }
        prompt = (
            "You are FocusGuard, a friendly productivity coach."
            " Summarise the user's current focus health in two sentences (<= 35 words each)."
            " Use an encouraging tone and reference concrete numbers when meaningful."
            " Do not repeat the input JSON.\n"
            f"Data: {json.dumps(payload, ensure_ascii=False)}"
        )
        return self._normalise_text(
            self._invoke(prompt, temperature=0.4, max_output_tokens=180, cache_ns="insight")
        )

    def explain_prediction(
        self,
        *,
        prediction: Dict[str, Any],
        cognitive_twin: Optional[Dict[str, Any]],
    ) -> Dict[str, Optional[str]]:
        """Return short human-readable explanation for the latest prediction."""
        if not self.is_enabled:
            return {}

        payload = {
            "prediction": prediction,
            "cognitive_twin": cognitive_twin or {},
        }
        prompt = (
            "You are FocusGuard's cognitive explainer."
            " In plain language, summarise why the latest focus prediction looks the way it does."
            " Provide two labelled lines exactly in this format:"
            " Summary: <25-word explanation of the prediction>\n"
            " Ghost: <25-word commentary on what the cognitive twin expects next>."
            " If you lack details for a line, output an empty string after the colon."
            " Data to reason about:\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        text = self._normalise_text(
            self._invoke(prompt, temperature=0.3, max_output_tokens=220, cache_ns="prediction")
        )
        if not text:
            return {}

        summary: Optional[str] = None
        ghost: Optional[str] = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            lowered = line.lower()
            if lowered.startswith("summary:") and summary is None:
                summary = line.split(":", 1)[1].strip() or None
            elif lowered.startswith("ghost:") and ghost is None:
                ghost = line.split(":", 1)[1].strip() or None

        if summary is None and ghost is None:
            # Fall back to treating the whole response as a summary.
            summary = text

        return {"summary": summary, "ghost_narrative": ghost}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _invoke(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        cache_ns: str,
    ) -> Optional[str]:
        if not self.is_enabled:
            return None

        cache_key = self._make_cache_key(cache_ns, prompt, temperature, max_output_tokens)
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        url = f"{self._base_url}/{self._model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        headers = {"Content-Type": "application/json"}
        params = {"key": self._api_key}

        try:
            response = requests.post(url, json=payload, headers=headers, params=params, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:  # pragma: no cover - network failure guard
            logger.warning("Gemini request failed: %s", exc)
            return None
        except ValueError as exc:
            logger.warning("Gemini returned non-JSON response: %s", exc)
            return None

        text = self._extract_text(data)
        if not text:
            return None

        with self._lock:
            self._cache[cache_key] = _CacheEntry(value=text, expires_at=time.time() + self._cache_ttl)
        return text

    def _extract_text(self, response: Dict[str, Any]) -> Optional[str]:
        candidates = response.get("candidates") or []
        for candidate in candidates:
            content = candidate.get("content") or {}
            parts = content.get("parts") or candidate.get("output") or []
            if isinstance(parts, list):
                for part in parts:
                    text = part.get("text") if isinstance(part, dict) else None
                    if text:
                        return text.strip()
            elif isinstance(parts, dict):
                text = parts.get("text")
                if text:
                    return text.strip()
        return None

    def _normalise_text(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        cleaned = text.strip()
        return cleaned or None

    def _cache_get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            if entry.expires_at < time.time():
                self._cache.pop(key, None)
                return None
            return entry.value

    def _make_cache_key(self, namespace: str, prompt: str, temperature: float, max_output_tokens: int) -> str:
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}:{temperature}:{max_output_tokens}"


__all__ = ["GeminiClient"]
