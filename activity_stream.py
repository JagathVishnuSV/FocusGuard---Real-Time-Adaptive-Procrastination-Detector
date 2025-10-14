"""
FocusGuard - Real-Time Activity Stream Module
Monitors actual Windows user activity in real-time
"""

import time
import threading
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Iterator, Dict, Any
from enum import Enum
from collections import deque, defaultdict
import psutil

# Windows-specific imports
try:
    import win32gui
    import win32process
    import win32con
    import win32api
    from pynput import mouse, keyboard
    import wmi
    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False
    logging.warning("Windows-specific modules not available. Some features may be limited.")

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of user activity events"""
    KEYSTROKE = "keystroke"
    CLICK = "click"
    APP_SWITCH = "app_switch"
    IDLE = "idle"
    WINDOW_FOCUS = "window_focus"
    URL_CHANGE = "url_change"


@dataclass
class ActivityEvent:
    """Represents a single user activity event"""
    timestamp: float
    event_type: EventType
    app_name: Optional[str] = None
    window_title: Optional[str] = None
    url: Optional[str] = None
    detail: Optional[str] = None
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "app_name": self.app_name,
            "window_title": self.window_title,
            "url": self.url,
            "detail": self.detail,
        }


class RealTimeActivityMonitor:
    """Real-time Windows activity monitoring using system APIs"""
    
    def __init__(self, config):
        self.config = config
        self.events_queue = deque(maxlen=10000)
        self.running = False
        self.threads = []
        
        # Activity tracking
        self.keystroke_count = 0
        self.click_count = 0
        self.last_app = None
        self.last_window_title = None
        self.last_activity_time = time.time()
        self.idle_threshold = 10.0  # seconds
        
        # Browser URL tracking (simplified)
        self.current_url = None
        self.browser_apps = {'chrome.exe', 'firefox.exe', 'msedge.exe', 'safari.exe'}
        
        # Input monitoring
        self.mouse_listener = None
        self.keyboard_listener = None
        
        # Initialize WMI for process monitoring
        if WINDOWS_AVAILABLE:
            try:
                self.wmi_connection = wmi.WMI()
            except Exception as e:
                logger.warning(f"WMI initialization failed: {e}")
                self.wmi_connection = None
        else:
            self.wmi_connection = None
    
    def _get_active_window_info(self) -> Dict[str, Any]:
        """Get information about the currently active window"""
        if not WINDOWS_AVAILABLE:
            return {"app_name": "Unknown", "window_title": "Unknown", "process_id": 0}
        
        try:
            # Get the active window handle
            hwnd = win32gui.GetForegroundWindow()
            if hwnd == 0:
                return {"app_name": "Unknown", "window_title": "Unknown", "process_id": 0}
            
            # Get window title
            window_title = win32gui.GetWindowText(hwnd)
            
            # Get process ID and name
            _, process_id = win32process.GetWindowThreadProcessId(hwnd)
            
            try:
                process = psutil.Process(process_id)
                app_name = process.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                app_name = "Unknown"
            
            return {
                "app_name": app_name,
                "window_title": window_title,
                "process_id": process_id,
                "hwnd": hwnd
            }
        except Exception as e:
            logger.error(f"Error getting active window info: {e}")
            return {"app_name": "Unknown", "window_title": "Unknown", "process_id": 0}
    
    def _extract_url_from_title(self, title: str, app_name: str) -> Optional[str]:
        """Extract URL from browser window title (basic implementation)"""
        if not app_name.lower().replace('.exe', '') in [app.replace('.exe', '') for app in self.browser_apps]:
            return None
        
        # Basic URL extraction from common browser title patterns
        # This is simplified - in production, you might use browser automation APIs
        common_patterns = [
            " - Google Chrome",
            " - Mozilla Firefox", 
            " - Microsoft Edge",
            " — Mozilla Firefox",
            " — Google Chrome"
        ]
        
        url_title = title
        for pattern in common_patterns:
            url_title = url_title.replace(pattern, "")
        
        # Check if it looks like a URL or domain
        if any(domain in url_title.lower() for domain in 
               ['youtube.com', 'github.com', 'stackoverflow.com', 'reddit.com', 
                'twitter.com', 'facebook.com', 'linkedin.com', 'google.com']):
            return url_title
        
        return None
    
    def _on_key_press(self, key):
        """Handle keyboard events"""
        current_time = time.time()
        self.last_activity_time = current_time
        self.keystroke_count += 1
        
        window_info = self._get_active_window_info()
        
        event = ActivityEvent(
            timestamp=current_time,
            event_type=EventType.KEYSTROKE,
            app_name=window_info["app_name"],
            window_title=window_info["window_title"],
            detail=str(key) if hasattr(key, 'char') else str(key)
        )
        
        self.events_queue.append(event)
    
    def _on_mouse_click(self, x, y, button, pressed):
        """Handle mouse click events"""
        if pressed:  # Only count on press, not release
            current_time = time.time()
            self.last_activity_time = current_time
            self.click_count += 1
            
            window_info = self._get_active_window_info()
            
            event = ActivityEvent(
                timestamp=current_time,
                event_type=EventType.CLICK,
                app_name=window_info["app_name"],
                window_title=window_info["window_title"],
                detail=f"{button.name}_click_at_{x}_{y}"
            )
            
            self.events_queue.append(event)
    
    def _monitor_window_changes(self):
        """Monitor for window/app changes"""
        last_check = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                window_info = self._get_active_window_info()
                
                # Check for app switch
                if (window_info["app_name"] != self.last_app or 
                    window_info["window_title"] != self.last_window_title):
                    
                    event = ActivityEvent(
                        timestamp=current_time,
                        event_type=EventType.APP_SWITCH,
                        app_name=window_info["app_name"],
                        window_title=window_info["window_title"],
                        url=self._extract_url_from_title(
                            window_info["window_title"], 
                            window_info["app_name"]
                        ),
                        detail=f"switched_from_{self.last_app}_to_{window_info['app_name']}"
                    )
                    
                    self.events_queue.append(event)
                    self.last_app = window_info["app_name"]
                    self.last_window_title = window_info["window_title"]
                
                # Check for idle periods
                idle_time = current_time - self.last_activity_time
                if idle_time > self.idle_threshold and (current_time - last_check) > self.idle_threshold:
                    event = ActivityEvent(
                        timestamp=current_time,
                        event_type=EventType.IDLE,
                        app_name=window_info["app_name"],
                        detail=f"idle_for_{idle_time:.1f}s"
                    )
                    self.events_queue.append(event)
                    last_check = current_time
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in window monitoring: {e}")
                time.sleep(1.0)
    
    def start_monitoring(self):
        """Start real-time activity monitoring"""
        if self.running:
            return
        
        logger.info("Starting real-time activity monitoring...")
        self.running = True
        
        if not WINDOWS_AVAILABLE:
            logger.error("Windows-specific monitoring not available")
            return
        
        try:
            # Start input listeners
            self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
            self.mouse_listener = mouse.Listener(on_click=self._on_mouse_click)
            
            self.keyboard_listener.start()
            self.mouse_listener.start()
            
            # Start window monitoring thread
            monitor_thread = threading.Thread(target=self._monitor_window_changes, daemon=True)
            monitor_thread.start()
            self.threads.append(monitor_thread)
            
            logger.info("Activity monitoring started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start activity monitoring: {e}")
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop activity monitoring"""
        logger.info("Stopping activity monitoring...")
        self.running = False
        
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        if self.mouse_listener:
            self.mouse_listener.stop()
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=1.0)
        
        logger.info("Activity monitoring stopped")
    
    def get_events(self, since_timestamp: Optional[float] = None) -> List[ActivityEvent]:
        """Get events since a specific timestamp"""
        if since_timestamp is None:
            return list(self.events_queue)
        
        return [event for event in self.events_queue if event.timestamp >= since_timestamp]
    
    def clear_events(self):
        """Clear the events queue"""
        self.events_queue.clear()
        self.keystroke_count = 0
        self.click_count = 0
    
    def stream(self, duration_seconds: float, interval_seconds: float = 1.0) -> Iterator[List[ActivityEvent]]:
        """
        Stream activity events in real-time
        
        Args:
            duration_seconds: Total duration to stream
            interval_seconds: Interval between batches
            
        Yields:
            List of ActivityEvent objects from real system monitoring
        """
        if not self.running:
            self.start_monitoring()
        
        start_time = time.time()
        last_check = start_time
        
        while time.time() - start_time < duration_seconds:
            current_time = time.time()
            
            # Get events since last check
            events = self.get_events(since_timestamp=last_check)
            last_check = current_time
            
            if events:
                yield events
            
            time.sleep(interval_seconds)
    
    def get_current_state(self) -> dict:
        """Get current state of the activity monitor"""
        window_info = self._get_active_window_info()
        current_time = time.time()
        idle_time = current_time - self.last_activity_time
        
        return {
            "current_app": window_info["app_name"],
            "window_title": window_info["window_title"],
            "current_url": self._extract_url_from_title(
                window_info["window_title"], 
                window_info["app_name"]
            ),
            "is_idle": idle_time > self.idle_threshold,
            "idle_time": idle_time,
            "keystroke_count": self.keystroke_count,
            "click_count": self.click_count,
        }


# Compatibility aliases for existing code - maps old simulation to real monitoring
ActivityStreamSimulator = RealTimeActivityMonitor

class RealTimeActivityStream:
    """Real-time activity stream interface"""
    
    def __init__(self, config):
        self.config = config
        self.monitor = RealTimeActivityMonitor(config)
        self.events_buffer: List[ActivityEvent] = []
        
    def get_next_batch(self, batch_size: int = 100) -> List[ActivityEvent]:
        """Get next batch of events"""
        events = self.monitor.get_events()
        if len(events) > batch_size:
            return events[-batch_size:]
        return events
    
    def stream_continuously(self, interval: float = 1.0) -> Iterator[ActivityEvent]:
        """Stream events one at a time"""
        if not self.monitor.running:
            self.monitor.start_monitoring()
        
        last_timestamp = time.time()
        
        while True:
            current_timestamp = time.time()
            events = self.monitor.get_events(since_timestamp=last_timestamp)
            
            for event in events:
                yield event
            
            last_timestamp = current_timestamp
            time.sleep(interval)
    
    def reset(self):
        """Reset the activity stream"""
        self.monitor.clear_events()
        
    def stop(self):
        """Stop monitoring"""
        self.monitor.stop_monitoring()