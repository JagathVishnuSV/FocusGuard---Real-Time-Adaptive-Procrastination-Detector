"""Gemini client helpers for FocusGuard."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

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
        min_request_interval: float = 2.0,
        cooldown_seconds: float = 30.0,
        max_requests_per_minute: int = 6,
        max_output_tokens_per_minute: int = 1800,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._enabled = bool(enabled and api_key)
        self._cache_ttl = cache_ttl
        self._min_request_interval = max(0.0, min_request_interval)
        self._cooldown_seconds = max(0.0, cooldown_seconds)
        self._max_requests_per_minute = max(0, max_requests_per_minute)
        self._max_tokens_per_minute = max(0, max_output_tokens_per_minute)

        self._cache: Dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self._last_request_at: float = 0.0
        self._cooldown_until: float = 0.0
        self._cooldown_backoff: float = 1.0
        self._request_log: Deque[float] = deque()
        self._token_log: Deque[Tuple[float, int]] = deque()

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
            min_request_interval=float(getattr(config_module, "GEMINI_MIN_REQUEST_INTERVAL_SECONDS", 3.0)),
            cooldown_seconds=float(getattr(config_module, "GEMINI_COOLDOWN_SECONDS", 45.0)),
            max_requests_per_minute=int(getattr(config_module, "GEMINI_MAX_REQUESTS_PER_MINUTE", 6)),
            max_output_tokens_per_minute=int(getattr(config_module, "GEMINI_MAX_OUTPUT_TOKENS_PER_MINUTE", 1800)),
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

    def generate_enrichment_bundle(
        self,
        *,
        context: Dict[str, Any],
        stats_today: Dict[str, Any],
        prediction: Dict[str, Any],
        ghost_snapshot: Optional[Dict[str, Any]],
        top_distractions: Dict[str, Any],
    ) -> Dict[str, Optional[str]]:
        """Produce all enrichment strings in a single Gemini call."""
        if not self.is_enabled:
            return {}

        payload = {
            "context": context,
            "stats_today": stats_today,
            "prediction": prediction,
            "ghost_snapshot": ghost_snapshot or {},
            "top_distractions": top_distractions,
        }
        prompt = (
            "You are FocusGuard's AI summariser. Using the JSON data, respond ONLY with a JSON object "
            "containing these string fields (empty string when unknown): context_summary, focus_insight, "
            "prediction_summary, ghost_narrative. Keep each value under 30 words, use an encouraging tone, "
            "and avoid repeating the raw JSON.\n"
            f"DATA: {json.dumps(payload, ensure_ascii=False)}"
        )
        text = self._invoke(prompt, temperature=0.35, max_output_tokens=320, cache_ns="bundle")
        if not text:
            return {}

        parsed = self._parse_json_response(text)
        if not isinstance(parsed, dict):
            return {}

        result: Dict[str, Optional[str]] = {}
        for key in ("context_summary", "focus_insight", "prediction_summary", "ghost_narrative"):
            value = parsed.get(key)
            if value is None:
                result[key] = None
                continue
            cleaned = str(value).strip()
            result[key] = cleaned or None
        return result

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

        now = time.time()
        if now < self._cooldown_until:
            logger.debug(
                "Gemini call skipped; still cooling down for %.1fs",
                self._cooldown_until - now,
            )
            return None

        if self._min_request_interval > 0:
            since_last = now - self._last_request_at
            if self._last_request_at > 0 and since_last < self._min_request_interval:
                wait_time = self._min_request_interval - since_last
                logger.debug("Gemini call waiting %.2fs to respect min interval", wait_time)
                time.sleep(wait_time)
                now = time.time()
                if now < self._cooldown_until:
                    logger.debug(
                        "Gemini call skipped; still cooling down for %.1fs",
                        self._cooldown_until - now,
                    )
                    return None

        if self._should_rate_limit(max_output_tokens):
            logger.debug("Gemini call skipped; local rate limiter active")
            return None

        self._last_request_at = now

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
            if response.status_code == 429:
                self._record_request(time.time(), max_output_tokens)
                self._enter_cooldown("rate limit reached")
                logger.warning("Gemini request hit upstream rate limit (429)")
                return None
            response.raise_for_status()
            data = response.json()
        except requests.HTTPError as exc:  # pragma: no cover - network failure guard
            if getattr(exc.response, "status_code", None) == 429:
                self._record_request(time.time(), max_output_tokens)
                self._enter_cooldown("rate limit reached")
            logger.warning("Gemini request failed: %s", exc)
            return None
        except requests.RequestException as exc:  # pragma: no cover - network failure guard
            logger.warning("Gemini request failed: %s", exc)
            return None
        except ValueError as exc:
            logger.warning("Gemini returned non-JSON response: %s", exc)
            return None

        self._record_request(now, max_output_tokens)
        self._cooldown_backoff = 1.0

        text = self._extract_text(data)
        if not text:
            return None

        with self._lock:
            self._cache[cache_key] = _CacheEntry(value=text, expires_at=time.time() + self._cache_ttl)
        return text

    def _enter_cooldown(self, reason: str) -> None:
        if self._cooldown_seconds <= 0:
            return
        duration = self._cooldown_seconds * self._cooldown_backoff
        self._cooldown_until = time.time() + duration
        self._cooldown_backoff = min(self._cooldown_backoff * 2, 8.0)
        logger.warning(
            "Gemini %s; backing off for %.0f seconds",
            reason,
            duration,
        )

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

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Best-effort JSON parser for model output."""
        text = (text or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return None

    def _cache_get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            if entry.expires_at < time.time():
                self._cache.pop(key, None)
                return None
            return entry.value

    def _should_rate_limit(self, planned_tokens: int) -> bool:
        """Return True if local per-minute limits are exceeded."""
        if self._max_requests_per_minute <= 0 and self._max_tokens_per_minute <= 0:
            return False

        now = time.time()
        cutoff = now - 60

        while self._request_log and self._request_log[0] < cutoff:
            self._request_log.popleft()
        while self._token_log and self._token_log[0][0] < cutoff:
            self._token_log.popleft()

        requests_in_window = len(self._request_log)
        tokens_in_window = sum(tokens for _, tokens in self._token_log)

        request_limited = (
            self._max_requests_per_minute > 0 and requests_in_window >= self._max_requests_per_minute
        )
        token_limited = (
            self._max_tokens_per_minute > 0 and (tokens_in_window + max(planned_tokens, 0)) > self._max_tokens_per_minute
        )

        if request_limited or token_limited:
            reason = "request budget" if request_limited else "token budget"
            self._enter_cooldown(f"local {reason} exhausted")
            return True
        return False

    def _record_request(self, timestamp: float, tokens: int) -> None:
        if self._max_requests_per_minute > 0:
            self._request_log.append(timestamp)
        if self._max_tokens_per_minute > 0:
            self._token_log.append((timestamp, max(0, tokens)))

    def _make_cache_key(self, namespace: str, prompt: str, temperature: float, max_output_tokens: int) -> str:
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}:{temperature}:{max_output_tokens}"


__all__ = ["GeminiClient"]
