# ======================= PATCHED ACTIVITY_STREAM.PY =========================
"""
FocusGuard - Real-Time Activity Stream Module (patched)
- better URL extraction using regex/domain heuristics
- thread-safe event queue operations
- more robust idle detection (tracks last_idle_event_time)
- emits WINDOW_FOCUS and URL_CHANGE events where appropriate
- safer APP_SWITCH detail formatting
- graceful behavior when windows-specific modules are missing
"""

import time
import threading
import logging
import ctypes
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Iterator, Dict, Any
from enum import Enum
from collections import deque, defaultdict
import psutil
import re

try:
    import winsound  # Windows sound playback
except Exception:
    winsound = None

# Windows-specific imports
try:
    import win32gui
    import win32process
    import win32con
    import win32api
    from pynput import mouse, keyboard
    import wmi
    WINDOWS_AVAILABLE = True
except Exception:
    WINDOWS_AVAILABLE = False
    # logger may not be configured by consumer yet; create temp logger
    logging.getLogger(__name__).warning(
        "Windows-specific modules not available. Real-time monitoring limited."
    )

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of user activity events"""
    KEYSTROKE = "keystroke"
    CLICK = "click"
    APP_SWITCH = "app_switch"
    IDLE = "idle"
    WINDOW_FOCUS = "window_focus"
    URL_CHANGE = "url_change"
    IDLE_ALERT = "idle_alert"


@dataclass
class ActivityEvent:
    """Represents a single user activity event"""
    timestamp: float
    event_type: EventType
    app_name: Optional[str] = None
    window_title: Optional[str] = None
    url: Optional[str] = None
    detail: Optional[str] = None
    process_path: Optional[str] = None
    window_class: Optional[str] = None

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "app_name": self.app_name,
            "window_title": self.window_title,
            "url": self.url,
            "detail": self.detail,
            "process_path": self.process_path,
            "window_class": self.window_class,
        }


class RealTimeActivityMonitor:
    """Real-time Windows activity monitoring using system APIs"""

    def __init__(self, config=None):
        self.config = config
        self.events_queue = deque(maxlen=10000)
        self._queue_lock = threading.Lock()
        self.running = False
        self.threads: List[threading.Thread] = []

        # Activity tracking
        self.keystroke_count = 0
        self.click_count = 0
        self.last_app = None
        self.last_window_title = None
        self.last_url = None
        self.last_activity_time = time.time()
        self.idle_threshold = 10.0  # seconds
        self.last_idle_event_time = 0.0
        self.idle_alert_enabled = bool(getattr(self.config, "ENABLE_IDLE_ALERTS", False))
        alert_threshold = float(getattr(self.config, "IDLE_ALERT_THRESHOLD_SECONDS", 300.0))
        self.idle_alert_threshold = max(alert_threshold, self.idle_threshold)
        self.idle_alert_cooldown = float(getattr(self.config, "IDLE_ALERT_COOLDOWN_SECONDS", 180.0))
        self.idle_alert_popup = bool(getattr(self.config, "ENABLE_IDLE_POPUP", False))
        self._last_idle_alert_time = 0.0

        # Browser URL tracking (simplified)
        self.current_url = None
        self.browser_apps = {'chrome.exe', 'firefox.exe', 'msedge.exe', 'safari.exe', 'brave.exe', 'opera.exe'}

        # Input monitoring placeholders
        self.mouse_listener = None
        self.keyboard_listener = None

        # WMI connection (optional)
        if WINDOWS_AVAILABLE:
            try:
                self.wmi_connection = wmi.WMI()
            except Exception as e:
                logger.warning(f"WMI initialization failed: {e}")
                self.wmi_connection = None
        else:
            self.wmi_connection = None

    # ------------------ Utilities ------------------

    def _safe_append_event(self, event: ActivityEvent):
        """Thread-safe append to event queue."""
        with self._queue_lock:
            self.events_queue.append(event)

    def _trigger_idle_alert(self, idle_duration: float, window_info: Optional[Dict[str, Any]] = None):
        """Play a sound / popup warning and log an idle alert event."""
        if not self.idle_alert_enabled:
            return

        now = time.time()
        if now - getattr(self, "_last_idle_alert_time", 0.0) < self.idle_alert_cooldown:
            return

        self._last_idle_alert_time = now
        window_meta = window_info or {}
        app_name = window_meta.get("app_name")
        window_title = window_meta.get("window_title")
        message = f"You've been idle for {int(idle_duration)} seconds. Time to refocus."

        if winsound:
            try:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except Exception as exc:
                logger.debug("Idle alert sound failed: %s", exc)

        if self.idle_alert_popup and WINDOWS_AVAILABLE:
            try:
                ctypes.windll.user32.MessageBoxW(0, message, "FocusGuard Idle Alert", 0x00001030)
            except Exception as exc:
                logger.debug("Idle alert popup failed: %s", exc)

        alert_event = ActivityEvent(
            timestamp=now,
            event_type=EventType.IDLE_ALERT,
            app_name=app_name,
            window_title=window_title,
            url=self.last_url,
            detail=message,
            process_path=window_meta.get("process_path"),
            window_class=window_meta.get("window_class"),
        )
        self._safe_append_event(alert_event)

    def _get_active_window_info(self) -> Dict[str, Any]:
        """Get information about the currently active window"""
        if not WINDOWS_AVAILABLE:
            return {"app_name": "Unknown", "window_title": "Unknown", "process_id": 0, "process_path": None, "window_class": None}

        try:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd == 0:
                return {"app_name": "Unknown", "window_title": "Unknown", "process_id": 0, "process_path": None, "window_class": None}

            window_title = win32gui.GetWindowText(hwnd) or ""
            _, process_id = win32process.GetWindowThreadProcessId(hwnd)

            try:
                process = psutil.Process(process_id)
                app_name = process.name()
                try:
                    process_path = process.exe()
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    process_path = None
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                app_name = "Unknown"
                process_path = None

            try:
                window_class = win32gui.GetClassName(hwnd)
            except Exception:
                window_class = None

            return {
                "app_name": app_name,
                "window_title": window_title,
                "process_id": process_id,
                "hwnd": hwnd,
                "process_path": process_path,
                "window_class": window_class,
            }
        except Exception as e:
            logger.error(f"Error getting active window info: {e}")
            return {"app_name": "Unknown", "window_title": "Unknown", "process_id": 0, "process_path": None, "window_class": None}

    # ------------------ URL Extraction ------------------

    _domain_re = re.compile(
        r"(?P<domain>(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?|localhost(?::\d+)?|127\.0\.0\.1(?::\d+)?|\[::1\](?::\d+)?|(?:dev|staging|test)\.[a-z0-9.-]+)",
        flags=re.IGNORECASE
    )

    def _extract_url_from_title(self, title: str, app_name: str) -> Optional[str]:
        """
        Try to extract a URL or domain from a browser title using heuristics and regex.
        This is still approximate; for production, use browser-level APIs.
        """
        if not title:
            return None

        app_norm = (app_name or "").lower().replace('.exe', '')
        is_browser = any(b.replace('.exe', '') in app_norm for b in self.browser_apps)
        if not is_browser:
            return None

        # Common patterns where title contains domain or site name
        # Examples:
        #  - "My Page - example.com - Google Chrome"
        #  - "example.com - YouTube"
        #  - "filename - Stack Overflow"
        # We'll try to find domain-like tokens.
        # First, remove typical suffixes added by browsers.
        cleaned = re.sub(r"\s+[-—–]\s+(Google Chrome|Mozilla Firefox|Microsoft Edge|Chrome|Firefox|Edge)$", "", title, flags=re.IGNORECASE).strip()

        # If the cleaned title looks like a URL or contains a domain, return that segment.
        m = self._domain_re.search(cleaned)
        if m:
            domain = m.group("domain")
            return domain.lower()

        # Local/dev hosts sometimes appear without dots; explicitly check tokenised title parts.
        dev_aliases = ["localhost", "127.0.0.1", "dev", "staging", "test", "intranet", "internal"]
        for token in re.split(r"[\s\|]+", cleaned):
            stripped = token.strip(" -—–·")
            if not stripped:
                continue
            lower = stripped.lower()
            if lower.startswith(tuple(dev_aliases)):
                return lower

        # As a last resort, if the title contains known site words, return the site token.
        known_sites = [
            "youtube", "github", "stackoverflow", "reddit", "twitter", "linkedin",
            "google", "notion", "chatgpt", "openai", "figma", "slack"
        ]
        low = cleaned.lower()
        for site in known_sites:
            if site in low:
                return site + ".com"

        return None

    # ------------------ Input Handlers ------------------

    def _on_key_press(self, key):
        current_time = time.time()
        self.last_activity_time = current_time
        self.keystroke_count += 1

        window_info = self._get_active_window_info()

        detail = None
        try:
            # pynput Key/char handling may differ; coerce to string
            detail = key.char if hasattr(key, "char") and key.char is not None else str(key)
        except Exception:
            detail = str(key)

        event = ActivityEvent(
            timestamp=current_time,
            event_type=EventType.KEYSTROKE,
            app_name=window_info.get("app_name"),
            window_title=window_info.get("window_title"),
            detail=detail,
            process_path=window_info.get("process_path"),
            window_class=window_info.get("window_class"),
        )

        self._safe_append_event(event)

    def _on_mouse_click(self, x, y, button, pressed):
        if not pressed:
            return

        current_time = time.time()
        self.last_activity_time = current_time
        self.click_count += 1

        window_info = self._get_active_window_info()

        event = ActivityEvent(
            timestamp=current_time,
            event_type=EventType.CLICK,
            app_name=window_info.get("app_name"),
            window_title=window_info.get("window_title"),
            detail=f"{getattr(button, 'name', 'button')}_click_at_{int(x)}_{int(y)}",
            process_path=window_info.get("process_path"),
            window_class=window_info.get("window_class"),
        )

        self._safe_append_event(event)

    # ------------------ Window/Idle Monitor ------------------

    def _monitor_window_changes(self):
        """Monitor for window/app changes, idle periods, and generate APP_SWITCH / IDLE / WINDOW_FOCUS / URL_CHANGE events."""
        poll_interval = 1.0
        while self.running:
            try:
                current_time = time.time()
                window_info = self._get_active_window_info()

                current_app = window_info.get("app_name")
                current_title = window_info.get("window_title") or ""
                current_url = self._extract_url_from_title(current_title, current_app)

                # APP_SWITCH or WINDOW_FOCUS detection
                switched = False
                if (current_app != self.last_app) or (current_title != self.last_window_title):
                    # Build safe detail
                    prev_app_safe = self.last_app if self.last_app is not None else "none"
                    new_app_safe = current_app if current_app is not None else "unknown"

                    # Create app switch event
                    switch_event = ActivityEvent(
                        timestamp=current_time,
                        event_type=EventType.APP_SWITCH,
                        app_name=current_app,
                        window_title=current_title,
                        url=current_url,
                        detail=f"switched_from_{prev_app_safe}_to_{new_app_safe}",
                        process_path=window_info.get("process_path"),
                        window_class=window_info.get("window_class"),
                    )
                    self._safe_append_event(switch_event)

                    # Also emit WINDOW_FOCUS (explicit)
                    focus_event = ActivityEvent(
                        timestamp=current_time,
                        event_type=EventType.WINDOW_FOCUS,
                        app_name=current_app,
                        window_title=current_title,
                        url=current_url,
                        detail=f"focus_{new_app_safe}",
                        process_path=window_info.get("process_path"),
                        window_class=window_info.get("window_class"),
                    )
                    self._safe_append_event(focus_event)

                    switched = True
                    self.last_app = current_app
                    self.last_window_title = current_title

                # URL change detection (within same browser or when browser swapped)
                if current_url != self.last_url:
                    url_change_event = ActivityEvent(
                        timestamp=current_time,
                        event_type=EventType.URL_CHANGE,
                        app_name=current_app,
                        window_title=current_title,
                        url=current_url,
                        detail=f"url_change_from_{self.last_url or 'none'}_to_{current_url or 'none'}",
                        process_path=window_info.get("process_path"),
                        window_class=window_info.get("window_class"),
                    )
                    self._safe_append_event(url_change_event)
                    self.last_url = current_url

                # Idle detection: generate idle event if we've been idle beyond threshold
                idle_time = current_time - self.last_activity_time
                if idle_time > self.idle_threshold:
                    # Avoid emitting idle repeatedly by checking last_idle_event_time
                    if current_time - self.last_idle_event_time > self.idle_threshold:
                        idle_event = ActivityEvent(
                            timestamp=current_time,
                            event_type=EventType.IDLE,
                            app_name=current_app,
                            window_title=current_title,
                            url=current_url,
                            detail=f"idle_for_{idle_time:.1f}s",
                            process_path=window_info.get("process_path"),
                            window_class=window_info.get("window_class"),
                        )
                        self._safe_append_event(idle_event)
                        self.last_idle_event_time = current_time

                        if self.idle_alert_enabled and idle_time >= self.idle_alert_threshold:
                            self._trigger_idle_alert(idle_time, window_info)

                # Sleep little to avoid busy loop
                time.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Error in window monitoring loop: {e}", exc_info=True)
                # avoid spinning on exceptions
                time.sleep(1.0)

    # ------------------ Control Methods ------------------

    def start_monitoring(self):
        """Start real-time activity monitoring"""
        if self.running:
            return

        logger.info("Starting real-time activity monitoring...")
        self.running = True

        if not WINDOWS_AVAILABLE:
            logger.error("Windows-specific monitoring not available. start_monitoring aborted.")
            return

        try:
            # Start input listeners
            self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
            self.mouse_listener = mouse.Listener(on_click=self._on_mouse_click)

            self.keyboard_listener.start()
            self.mouse_listener.start()

            # Start window monitoring thread
            monitor_thread = threading.Thread(target=self._monitor_window_changes, daemon=True, name="WindowMonitor")
            monitor_thread.start()
            self.threads.append(monitor_thread)

            logger.info("Activity monitoring started successfully")

        except Exception as e:
            logger.error(f"Failed to start activity monitoring: {e}", exc_info=True)
            self.stop_monitoring()

    def stop_monitoring(self):
        """Stop activity monitoring"""
        logger.info("Stopping activity monitoring...")
        self.running = False

        # stop listeners safely
        try:
            if getattr(self, "keyboard_listener", None):
                self.keyboard_listener.stop()
        except Exception as e:
            logger.debug(f"Error stopping keyboard listener: {e}")

        try:
            if getattr(self, "mouse_listener", None):
                self.mouse_listener.stop()
        except Exception as e:
            logger.debug(f"Error stopping mouse listener: {e}")

        # join monitor threads
        for thread in self.threads:
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass

        self.threads.clear()
        logger.info("Activity monitoring stopped")

    # ------------------ Event Accessors ------------------

    def get_events(self, since_timestamp: Optional[float] = None) -> List[ActivityEvent]:
        """Get events since a specific timestamp"""
        with self._queue_lock:
            if since_timestamp is None:
                return list(self.events_queue)
            # events_queue is ordered from oldest to newest
            return [event for event in list(self.events_queue) if event.timestamp >= since_timestamp]

    def clear_events(self):
        """Clear the events queue"""
        with self._queue_lock:
            self.events_queue.clear()
            self.keystroke_count = 0
            self.click_count = 0

    # ------------------ Streaming Interfaces ------------------

    def stream(self, duration_seconds: float, interval_seconds: float = 1.0) -> Iterator[List[ActivityEvent]]:
        """
        Stream activity events in real-time

        Yields lists of ActivityEvent objects captured in each interval.
        """
        if not self.running and WINDOWS_AVAILABLE:
            self.start_monitoring()

        start_time = time.time()
        last_check = start_time

        while time.time() - start_time < duration_seconds:
            now = time.time()
            events = self.get_events(since_timestamp=last_check)
            last_check = now
            if events:
                yield events
            time.sleep(interval_seconds)

    def get_current_state(self) -> dict:
        """Get current state of the activity monitor"""
        window_info = self._get_active_window_info()
        current_time = time.time()
        idle_time = current_time - self.last_activity_time
        return {
            "current_app": window_info.get("app_name"),
            "window_title": window_info.get("window_title"),
            "current_url": self._extract_url_from_title(window_info.get("window_title") or "", window_info.get("app_name") or ""),
            "is_idle": idle_time > self.idle_threshold,
            "idle_time": idle_time,
            "keystroke_count": self.keystroke_count,
            "click_count": self.click_count,
        }


# Compatibility alias for existing code paths
ActivityStreamSimulator = RealTimeActivityMonitor


class RealTimeActivityStream:
    """Real-time activity stream interface wrapper."""

    def __init__(self, config=None):
        self.config = config
        self.monitor = RealTimeActivityMonitor(config)
        self.events_buffer: List[ActivityEvent] = []

    def get_next_batch(self, batch_size: int = 100) -> List[ActivityEvent]:
        """Get next batch of events (most recent)"""
        events = self.monitor.get_events()
        if len(events) > batch_size:
            return events[-batch_size:]
        return events

    def stream_continuously(self, interval: float = 1.0) -> Iterator[ActivityEvent]:
        """Stream events one at a time; yields indefinitely until monitor stopped externally."""
        if not self.monitor.running and WINDOWS_AVAILABLE:
            self.monitor.start_monitoring()

        last_timestamp = time.time()
        try:
            while True:
                current_timestamp = time.time()
                events = self.monitor.get_events(since_timestamp=last_timestamp)
                for event in events:
                    yield event
                last_timestamp = current_timestamp
                time.sleep(interval)
        except GeneratorExit:
            # consumer closed the generator
            return

    def reset(self):
        """Reset the activity stream (clears events)."""
        self.monitor.clear_events()

    def stop(self):
        """Stop monitoring."""
        self.monitor.stop_monitoring()
