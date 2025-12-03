# ======================= PATCHED FEATURE_EXTRACTOR.PY =========================

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import math
from urllib.parse import urlparse
import re

from activity_stream import ActivityEvent, EventType
from config import FEATURE_NAMES, WEBSITE_CATEGORIES
from personalization import get_override_category

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.website_categories = WEBSITE_CATEGORIES

        # Track last context summary
        self._last_context_summary = {
            "dominant_context": "unknown",
            "context_confidence": 0.0,
            "context_counts": {},
        }

        # App sets
        self.productive_apps = set(config.SIMULATION_PARAMS.get("productive_apps", []))
        self.distraction_apps = set(config.SIMULATION_PARAMS.get("distraction_apps", []))

        self.browser_apps = {
            "chrome.exe", "firefox.exe", "msedge.exe", "iexplore.exe",
            "opera.exe", "brave.exe", "safari.exe"
        }

        # Development host patterns (new)
        self.development_hosts = [
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            ".local",
            "localtest.me",
            "dev.",
            "staging.",
            "test.",
            "preview.",
            "ngrok-free.app"
        ]

        # Context keyword heuristics
        self.context_keywords = {
            "development": [
                "code", "ide", "compiler", "terminal", "repository", "pull request",
                "stack overflow", "documentation", "npm", "pip", "build", "deploy",
                "pytest", "webpack", "intellij", "studio", "webstorm", "android studio",
                "xcode", "localhost", "127.0.0.1", "chatgpt", "openai", "devtools",
                "ci/cd", "playwright", "localhost:", "vercel", "preview"
            ],
            "research": ["tutorial", "docs", "analysis", "reference", "guide"],
            "communication": ["slack", "teams", "zoom", "email", "calendar"],
            "project": ["notion", "trello", "asana", "jira", "clickup", "monday"],
            "entertainment": ["youtube", "netflix", "spotify", "discord", "social"],
            "design": ["figma", "photoshop", "illustrator", "blender"],
            "finance": ["invoice", "quickbooks", "tally", "ledger", "stripe", "bank"],
            "writing": ["draft", "copy", "editor", "blog", "markdown"],
            "ops": ["grafana", "prometheus", "datadog", "kibana", "pagerduty"],
        }

        self.context_negative_keywords = {
            "entertainment": ["documentation", "developer", "api", "jira"],
            "development": ["playlist", "trailer", "season"],
            "communication": ["gaming", "spotify"],
        }

        self.context_process_hints = {
            "development": ["\\idea", "\\studio", "python.exe", "node.exe", "code.exe"],
            "design": ["photoshop.exe", "xd.exe", "figma.exe", "illustrator.exe"],
            "finance": ["excel.exe", "quickbooks.exe", "powerbi.exe"],
            "communication": ["teams.exe", "slack.exe", "zoom.exe"],
            "entertainment": ["steam.exe", "vlc.exe", "spotify.exe"],
            "ops": ["grafana", "kibana", "datadog"],
        }

    # ------------------ URL & Domain Helpers ------------------

    def _extract_host(self, url: str) -> str:
        if not url:
            return ""

        try:
            if "://" not in url:
                url = "https://" + url
            host = urlparse(url).netloc.lower()
            return host
        except Exception:
            return ""

    # ------------------ Categorization ------------------

    def _categorize_app(self, app_name: str, window_title="", url="", process_path=None, window_class=None):
        if not app_name:
            return "neutral"

        app_lower = app_name.lower()
        host = self._extract_host(url or "") if url else ""

        # User-defined overrides
        override = get_override_category(app_lower, url)
        if override in ("productive", "distraction"):
            return override

        # Cached context inference
        ctx_label, ctx_conf = self.infer_context(
            app_name, window_title, url,
            process_path=process_path,
            window_class=window_class
        )

        # Confident category assignment
        if ctx_label == "entertainment" and ctx_conf >= 0.4:
            return "distraction"

        if ctx_label in {"development", "research", "project", "design", "writing", "finance", "ops"} and ctx_conf >= 0.4:
            return "productive"

        if ctx_label == "communication" and ctx_conf >= 0.6:
            return "productive"

        # Exact exe match fallback
        if app_lower in (p.lower() for p in self.productive_apps):
            return "productive"

        if app_lower in (d.lower() for d in self.distraction_apps):
            return "distraction"

        # Browser heuristic
        if any(b in app_lower for b in self.browser_apps):
            combined = f"{window_title} {url}".lower()

            # Config domains
            for domain, category in self.website_categories.items():
                if domain in combined:
                    if category in ("development", "learning", "productivity"):
                        return "productive"
                    if category == "distraction":
                        return "distraction"

            # Dev domains
            if any(x in combined for x in self.development_hosts) or (host and any(x in host for x in self.development_hosts)):
                return "productive"

            # Keywords
            if any(k in combined for k in ["github", "stackoverflow", "tutorial", "docs", "chatgpt", "openai"]):
                return "productive"

            if any(k in combined for k in ["youtube", "netflix", "meme", "video"]):
                return "distraction"

        return "neutral"

    # ------------------ Context Inference ------------------

    def infer_context(self, app_name, window_title="", url="", *, process_path=None, window_class=None):
        if not (app_name or window_title or url):
            return "unknown", 0.0

        confidence = 0.0
        label = "unknown"
        app_lower = (app_name or "").lower()

        host = self._extract_host(url)
        aggregated = " ".join([
            app_name or "",
            window_title or "",
            url or "",
            host,
            (process_path or "").lower(),
            (window_class or "").lower()
        ]).lower()

        # Helper
        def assign(cat, boost):
            nonlocal label, confidence
            if boost > confidence:
                label = cat
                confidence = boost

        # Direct binary signals
        if app_lower.endswith(("code.exe", "pycharm64.exe", "webstorm64.exe")):
            assign("development", 0.85)

        if any(x in app_lower for x in ["slack", "teams", "zoom", "outlook"]):
            assign("communication", 0.85)

        if any(x in app_lower for x in ["spotify", "discord", "vlc", "steam"]):
            assign("entertainment", 0.85)

        # Process hints
        if process_path:
            lower_path = process_path.lower()
            for cat, hints in self.context_process_hints.items():
                if any(h in lower_path for h in hints):
                    assign(cat, max(confidence, 0.8))

        # Keyword matches
        for cat, kw_list in self.context_keywords.items():
            if any(k in aggregated for k in kw_list):
                assign(cat, max(confidence, 0.6))

        # Dev hosts
        if host:
            if any(h in host for h in self.development_hosts):
                assign("development", 0.9)

        # Config categories
        for domain, category in self.website_categories.items():
            if domain in aggregated:
                if category in {"development", "learning", "productivity"}:
                    assign("development", 0.75)
                elif category == "distraction":
                    assign("entertainment", 0.8)
                elif category == "shopping":
                    assign("project", 0.5)

        # Negative suppression
        if label in self.context_negative_keywords:
            if any(t in aggregated for t in self.context_negative_keywords[label]):
                confidence -= 0.3
                if confidence < 0.4:
                    return "unknown", confidence

        if confidence < 0.35:
            return "unknown", round(confidence, 3)

        return label, round(confidence, 3)

    # ------------------ Basic Metrics ------------------

    def _calculate_entropy(self, data):
        if not data:
            return 0.0
        counter = Counter(data)
        total = len(data)
        ent = 0
        for c in counter.values():
            p = c / total
            ent -= p * math.log2(p)
        return ent

    def _calculate_burst_score(self, timestamps, threshold=2.0):
        if len(timestamps) < 2:
            return 0.0
        intervals = np.diff(sorted(timestamps))
        if len(intervals) == 0:
            return 0.0
        m = np.mean(intervals)
        if m <= 0:
            return 0.0
        score = np.std(intervals) / m
        return min(1.0, score / threshold)

    def _calculate_idle_ratio(self, events, window_secs):
        total_idle = 0.0
        for e in events:
            if e.event_type == EventType.IDLE and e.detail:
                match = re.search(r"idle_for_(\d+(\.\d+)?)s", e.detail)
                if match:
                    total_idle += float(match.group(1))
        return min(1.0, total_idle / window_secs)

    # ------------------ Focus Duration Fix ------------------

    def _compute_focus_duration(self, events):
        events = sorted(events, key=lambda e: e.timestamp)
        focus_spans = defaultdict(list)

        prev_app = None
        start_time = None

        for e in events:
            if not e.app_name:
                continue
            if e.app_name != prev_app:
                if prev_app and start_time:
                    focus_spans[prev_app].append(e.timestamp - start_time)
                start_time = e.timestamp
                prev_app = e.app_name

        if prev_app and start_time:
            focus_spans[prev_app].append(events[-1].timestamp - start_time)

        durations = [sum(v) for v in focus_spans.values()]
        return np.mean(durations) if durations else 0.0

    # ------------------ Feature Extraction ------------------

    def extract_features(self, events, window_size_seconds=30.0):
        if not events:
            return np.zeros(16)

        # Window filter
        latest = max(e.timestamp for e in events)
        start = latest - window_size_seconds
        window = [e for e in events if e.timestamp >= start]

        if not window:
            return np.zeros(16)

        keystrokes = [e.timestamp for e in window if e.event_type == EventType.KEYSTROKE]
        clicks = [e.timestamp for e in window if e.event_type == EventType.CLICK]
        app_switches = len([e for e in window if e.event_type == EventType.APP_SWITCH])

        apps_used = [e.app_name for e in window if e.app_name]
        app_entropy = self._calculate_entropy(apps_used)

        idle_ratio = self._calculate_idle_ratio(window, window_size_seconds)

        # Categorization (cached)
        context_counter = Counter()
        ctx_conf_budget = defaultdict(float)
        categories = []

        infer_cache = {}

        for e in window:
            if not e.app_name:
                continue

            key = (e.app_name, e.window_title, e.url, e.process_path, e.window_class)
            if key not in infer_cache:
                infer_cache[key] = self.infer_context(
                    e.app_name, e.window_title or "", e.url or "",
                    process_path=e.process_path, window_class=e.window_class
                )

            ctx_label, ctx_conf = infer_cache[key]
            ctx_counter = ctx_label
            context_counter[ctx_label] += 1
            ctx_conf_budget[ctx_label] += ctx_conf

            categories.append(self._categorize_app(
                e.app_name, e.window_title or "", e.url or "",
                e.process_path, e.window_class
            ))

        total = len(categories)
        prod_ratio = categories.count("productive") / total if total else 0.0
        dist_ratio = categories.count("distraction") / total if total else 0.0

        features = [
            len(keystrokes) / window_size_seconds,
            len(clicks) / window_size_seconds,
            app_switches,
            app_entropy,
            idle_ratio,
            prod_ratio,
            dist_ratio,
            self._calculate_burst_score(keystrokes),
            self._calculate_burst_score(clicks),
            app_switches / window_size_seconds,
            np.var(np.diff(sorted(keystrokes))) if len(keystrokes) > 1 else 0.0,
            np.var(np.diff(sorted(clicks))) if len(clicks) > 1 else 0.0,
            len(keystrokes) / (len(keystrokes) + len(clicks)) if (len(keystrokes) + len(clicks)) > 0 else 0.0,
            len([e for e in window if e.event_type == EventType.IDLE]),
            self._compute_focus_duration(window),
            app_switches / len(window)
        ]

        vect = np.nan_to_num(np.array(features))

        # Context summary
        if context_counter:
            dom, hits = context_counter.most_common(1)[0]
            avg_conf = ctx_conf_budget[dom] / max(hits, 1)
            self._last_context_summary = {
                "dominant_context": dom,
                "context_confidence": round(min(avg_conf + (hits / sum(context_counter.values())) * 0.2, 1.0), 3),
                "context_counts": dict(context_counter),
            }

        return vect

    def extract_features_batch(self, batch_events, window_size_seconds=30.0):
        return np.array([self.extract_features(ev, window_size_seconds) for ev in batch_events])

    def get_feature_names(self):
        return FEATURE_NAMES.copy()

    def get_last_context_summary(self):
        return self._last_context_summary.copy()
