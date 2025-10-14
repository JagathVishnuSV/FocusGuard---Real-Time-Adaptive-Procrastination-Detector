"""
FocusGuard - Feature Extraction Module
Extracts 16 machine learning features from real-time activity events
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
from datetime import datetime
import math

from activity_stream import ActivityEvent, EventType
from config import FEATURE_NAMES, WEBSITE_CATEGORIES

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts behavioral features from activity events for ML models"""
    
    def __init__(self, config):
        self.config = config
        self.website_categories = WEBSITE_CATEGORIES
        
        # Define productive and distraction apps from config
        self.productive_apps = set(config.SIMULATION_PARAMS.get("productive_apps", [
            "vscode.exe", "pycharm64.exe", "sublime_text.exe", "cmd.exe", "powershell.exe",
            "notepad++.exe", "outlook.exe", "winword.exe", "excel.exe", "teams.exe"
        ]))
        
        self.distraction_apps = set(config.SIMULATION_PARAMS.get("distraction_apps", [
            "youtube.com", "twitter.com", "reddit.com", "netflix.com", "steam.exe",
            "spotify.exe", "discord.exe", "whatsapp.exe", "telegram.exe"
        ]))
        
        self.browser_apps = {
            "chrome.exe", "firefox.exe", "msedge.exe", "iexplore.exe", 
            "opera.exe", "brave.exe", "safari.exe"
        }
    
    def _categorize_app(self, app_name: str, window_title: str = "", url: str = "") -> str:
        """Categorize an app as productive, distraction, or neutral"""
        if not app_name:
            return "neutral"
        
        app_lower = app_name.lower()
        
        # Check direct app categorization
        if any(prod_app.lower() in app_lower for prod_app in self.productive_apps):
            return "productive"
        
        if any(dist_app.lower() in app_lower for dist_app in self.distraction_apps):
            return "distraction"
        
        # For browsers, check URL or window title
        if any(browser.lower() in app_lower for browser in self.browser_apps):
            combined_text = f"{window_title} {url}".lower()
            
            # Check website categories
            for domain, category in self.website_categories.items():
                if domain in combined_text:
                    if category in ["development", "learning", "productivity"]:
                        return "productive"
                    elif category == "distraction":
                        return "distraction"
            
            # Check for productive keywords
            productive_keywords = [
                "github", "stackoverflow", "documentation", "docs", "tutorial",
                "learning", "course", "education", "work", "project", "code"
            ]
            if any(keyword in combined_text for keyword in productive_keywords):
                return "productive"
            
            # Check for distraction keywords
            distraction_keywords = [
                "youtube", "netflix", "social", "game", "entertainment", 
                "funny", "meme", "video", "stream", "music", "chat"
            ]
            if any(keyword in combined_text for keyword in distraction_keywords):
                return "distraction"
        
        return "neutral"
    
    def _calculate_entropy(self, data: List[Any]) -> float:
        """Calculate Shannon entropy of a list"""
        if not data:
            return 0.0
        
        counter = Counter(data)
        total = len(data)
        entropy = 0.0
        
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _calculate_burst_score(self, timestamps: List[float], threshold: float = 2.0) -> float:
        """Calculate burstiness score for events"""
        if len(timestamps) < 2:
            return 0.0
        
        # Calculate inter-event intervals
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if not intervals:
            return 0.0
        
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval == 0:
            return 0.0
        
        # Burstiness: ratio of std to mean
        burstiness = std_interval / mean_interval if mean_interval > 0 else 0.0
        
        # Normalize to 0-1 range
        return min(1.0, burstiness / threshold)
    
    def _calculate_idle_time_ratio(self, events: List[ActivityEvent], window_size_seconds: float) -> float:
        """Calculate ratio of idle time in the window"""
        idle_events = [e for e in events if e.event_type == EventType.IDLE]
        if not idle_events:
            return 0.0
        
        total_idle_time = 0.0
        for event in idle_events:
            if event.detail and "idle_for_" in event.detail:
                try:
                    idle_duration = float(event.detail.split("idle_for_")[1].split("s")[0])
                    total_idle_time += idle_duration
                except (ValueError, IndexError):
                    pass
        
        return min(1.0, total_idle_time / window_size_seconds)
    
    def extract_features(self, events: List[ActivityEvent], window_size_seconds: float = 30.0) -> np.ndarray:
        """
        Extract 16-dimensional feature vector from activity events
        
        Args:
            events: List of ActivityEvent objects
            window_size_seconds: Time window for analysis
            
        Returns:
            16-dimensional numpy array of features
        """
        if not events:
            return np.zeros(16)
        
        # Filter events within time window
        latest_time = max(event.timestamp for event in events)
        window_start = latest_time - window_size_seconds
        window_events = [e for e in events if e.timestamp >= window_start]
        
        if not window_events:
            return np.zeros(16)
        
        # Separate events by type
        keystroke_events = [e for e in window_events if e.event_type == EventType.KEYSTROKE]
        click_events = [e for e in window_events if e.event_type == EventType.CLICK]
        app_switch_events = [e for e in window_events if e.event_type == EventType.APP_SWITCH]
        idle_events = [e for e in window_events if e.event_type == EventType.IDLE]
        
        # Feature 1: Keystrokes per second
        keystrokes_per_sec = len(keystroke_events) / window_size_seconds
        
        # Feature 2: Clicks per second
        clicks_per_sec = len(click_events) / window_size_seconds
        
        # Feature 3: App switches
        app_switches = len(app_switch_events)
        
        # Feature 4: App entropy (diversity of apps used)
        apps_used = [e.app_name for e in window_events if e.app_name]
        app_entropy = self._calculate_entropy(apps_used)
        
        # Feature 5: Idle time ratio
        idle_time_ratio = self._calculate_idle_time_ratio(window_events, window_size_seconds)
        
        # Feature 6 & 7: Productive vs distraction app ratios
        app_categories = []
        for event in window_events:
            if event.app_name:
                category = self._categorize_app(
                    event.app_name, 
                    event.window_title or "", 
                    event.url or ""
                )
                app_categories.append(category)
        
        total_categorized = len(app_categories)
        productive_count = app_categories.count("productive")
        distraction_count = app_categories.count("distraction")
        
        productive_app_ratio = productive_count / total_categorized if total_categorized > 0 else 0.0
        distraction_app_ratio = distraction_count / total_categorized if total_categorized > 0 else 0.0
        
        # Feature 8: Keystroke burst score
        keystroke_timestamps = [e.timestamp for e in keystroke_events]
        keystroke_burst_score = self._calculate_burst_score(keystroke_timestamps)
        
        # Feature 9: Click burst score
        click_timestamps = [e.timestamp for e in click_events]
        click_burst_score = self._calculate_burst_score(click_timestamps)
        
        # Feature 10: App switch frequency
        app_switch_frequency = app_switches / window_size_seconds
        
        # Feature 11: Keystroke variance
        if len(keystroke_timestamps) > 1:
            keystroke_intervals = [keystroke_timestamps[i+1] - keystroke_timestamps[i] 
                                 for i in range(len(keystroke_timestamps)-1)]
            keystroke_variance = np.var(keystroke_intervals) if keystroke_intervals else 0.0
        else:
            keystroke_variance = 0.0
        
        # Feature 12: Click variance
        if len(click_timestamps) > 1:
            click_intervals = [click_timestamps[i+1] - click_timestamps[i] 
                             for i in range(len(click_timestamps)-1)]
            click_variance = np.var(click_intervals) if click_intervals else 0.0
        else:
            click_variance = 0.0
        
        # Feature 13: Keystroke to click ratio
        total_inputs = len(keystroke_events) + len(click_events)
        keystroke_click_ratio = len(keystroke_events) / total_inputs if total_inputs > 0 else 0.0
        
        # Feature 14: Idle transitions (number of idle periods)
        idle_transitions = len(idle_events)
        
        # Feature 15: App focus duration (average time per app)
        app_times = defaultdict(list)
        for event in window_events:
            if event.app_name:
                app_times[event.app_name].append(event.timestamp)
        
        if app_times:
            focus_durations = []
            for app, timestamps in app_times.items():
                if len(timestamps) > 1:
                    duration = max(timestamps) - min(timestamps)
                    focus_durations.append(duration)
            
            app_focus_duration = np.mean(focus_durations) if focus_durations else 0.0
        else:
            app_focus_duration = 0.0
        
        # Feature 16: Context switch cost (normalized by events)
        context_switch_cost = app_switches / len(window_events) if window_events else 0.0
        
        # Assemble feature vector
        features = np.array([
            keystrokes_per_sec,
            clicks_per_sec,
            app_switches,
            app_entropy,
            idle_time_ratio,
            productive_app_ratio,
            distraction_app_ratio,
            keystroke_burst_score,
            click_burst_score,
            app_switch_frequency,
            keystroke_variance,
            click_variance,
            keystroke_click_ratio,
            idle_transitions,
            app_focus_duration,
            context_switch_cost
        ])
        
        # Handle any NaN or infinite values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Log feature extraction
        logger.debug(f"Extracted features from {len(window_events)} events: {dict(zip(FEATURE_NAMES, features))}")
        
        return features
    
    def extract_features_batch(self, events_list: List[List[ActivityEvent]], 
                             window_size_seconds: float = 30.0) -> np.ndarray:
        """
        Extract features from multiple event batches
        
        Args:
            events_list: List of event lists
            window_size_seconds: Time window for analysis
            
        Returns:
            2D numpy array (n_samples, n_features)
        """
        feature_vectors = []
        
        for events in events_list:
            features = self.extract_features(events, window_size_seconds)
            feature_vectors.append(features)
        
        return np.array(feature_vectors)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return FEATURE_NAMES.copy()
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of each feature"""
        return {
            "keystrokes_per_sec": "Rate of keyboard input (typing intensity)",
            "clicks_per_sec": "Rate of mouse clicks", 
            "app_switches": "Number of application switches",
            "app_entropy": "Diversity of applications used (Shannon entropy)",
            "idle_time_ratio": "Proportion of time spent idle",
            "productive_app_ratio": "Proportion of time in productive applications",
            "distraction_app_ratio": "Proportion of time in distracting applications",
            "keystroke_burst_score": "Burstiness of typing patterns",
            "click_burst_score": "Burstiness of clicking patterns",
            "app_switch_frequency": "Rate of application switching",
            "keystroke_variance": "Variability in typing timing",
            "click_variance": "Variability in clicking timing",
            "keystroke_click_ratio": "Ratio of keystrokes to total inputs",
            "idle_transitions": "Number of idle periods",
            "app_focus_duration": "Average time spent per application",
            "context_switch_cost": "Relative cost of context switching"
        }
