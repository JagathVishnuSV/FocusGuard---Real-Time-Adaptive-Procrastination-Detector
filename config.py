"""
FocusGuard - Configuration Management
Central configuration file for all constants and settings
"""

import os
from pathlib import Path
from typing import Dict, List

# ===== PROJECT PATHS =====
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
USER_DATA_DIR = DATA_DIR / "personalization"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, USER_DATA_DIR]:
    directory.mkdir(exist_ok=True)

# ===== FILE PATHS =====
RAW_DATA_FILE = DATA_DIR / "raw_uncalibrated.csv"
LABELED_DATA_FILE = DATA_DIR / "labeled_feedback.csv"
MODEL_FILE = MODELS_DIR / "anomaly_detector.joblib"
RANDOM_FOREST_MODEL_FILE = MODELS_DIR / "classifier.joblib"
SCALER_FILE = MODELS_DIR / "scaler.joblib"
CLASSIFIER_CALIBRATOR_FILE = MODELS_DIR / "classifier_calibrator.joblib"
ENSEMBLE_COMBINER_FILE = MODELS_DIR / "combiner.joblib"
LOG_FILE = LOGS_DIR / "focusguard.log"
ANALYTICS_DB = DATA_DIR / "analytics.json"
SESSION_LOG = DATA_DIR / "session_log.jsonl"
MODEL_REGISTRY_FILE = MODELS_DIR / "artifacts.json"
USER_FEEDBACK_FILE = USER_DATA_DIR / "feedback.jsonl"
USER_FEATURE_SNAPSHOTS = USER_DATA_DIR / "feature_snapshots.jsonl"
USER_OVERRIDES_FILE = USER_DATA_DIR / "overrides.json"
USER_PASSIVE_LABELS_FILE = USER_DATA_DIR / "passive_labels.jsonl"

# ===== PHASE 1: CALIBRATION SETTINGS =====
CALIBRATION_DURATION_SECONDS = 300  # 5 minutes (90 for quick testing)
CALIBRATION_WINDOW_SIZE = 10  # seconds for feature extraction during calibration

# ===== PHASE 2: DETECTION SETTINGS =====
DETECTION_INTERVAL_SECONDS = 5  # Check every 5 seconds
DETECTION_WINDOW_SIZE = 30  # Use last 30 seconds of activity for feature extraction
ANOMALY_CONFIDENCE_THRESHOLD = 0.6  # Confidence threshold for alerts (0-1)
PASSIVE_LABEL_MIN_INTERVAL_SECONDS = 20  # Minimum spacing between auto labels to avoid flooding

# ===== MODEL PARAMETERS =====
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 300,
    "contamination": 0.05,  # Expect ~5% anomalies by default
    "random_state": 42,
    "n_jobs": -1,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",
}

# ===== REAL-TIME MONITORING PARAMETERS =====
SIMULATION_PARAMS = {
    # These are now used for categorization rather than simulation
    "productive_apps": [
        "vscode.exe", "pycharm64.exe", "sublime_text.exe", "cmd.exe", "powershell.exe",
        "notepad++.exe", "outlook.exe", "winword.exe", "excel.exe", "teams.exe",
        "slack.exe", "code.exe", "devenv.exe", "vim.exe", "emacs.exe"
    ],
    "distraction_apps": [
        "steam.exe", "spotify.exe", "discord.exe", "whatsapp.exe", "telegram.exe",
        "netflix.exe", "vlc.exe", "mediaplayer.exe", "games.exe"
    ],
}

# ===== FEATURE ENGINEERING =====
FEATURE_NAMES = [
    "keystrokes_per_sec",
    "clicks_per_sec",
    "app_switches",
    "app_entropy",
    "idle_time_ratio",
    "productive_app_ratio",
    "distraction_app_ratio",
    "keystroke_burst_score",
    "click_burst_score",
    "app_switch_frequency",
    "keystroke_variance",
    "click_variance",
    "keystroke_click_ratio",
    "idle_transitions",
    "app_focus_duration",
    "context_switch_cost",
]

# ===== USER INTERFACE =====
ENABLE_GUI = True  # Set to True to enable web dashboard
WEB_SERVER_PORT = 8000
WEB_SERVER_HOST = "127.0.0.1"

# ===== LOGGING =====
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ===== FEEDBACK COLLECTION =====
FEEDBACK_PROMPT_TIMEOUT = 10  # seconds to wait for user response
FEEDBACK_AUTO_SKIP = False  # Skip feedback prompt if True

# ===== ANALYTICS & REPORTING =====
REPORT_INTERVAL_SECONDS = 300  # Generate report every 5 minutes
MIN_SAMPLES_FOR_TRAINING = 100  # Minimum labeled samples before retraining
RETRAINING_INTERVAL_MINUTES = 30  # Retrain model every 30 minutes
MIN_PERSONAL_FEEDBACK_FOR_RETRAIN = 12

# ===== WEBSITE CATEGORIZATION =====
WEBSITE_CATEGORIES: Dict[str, str] = {
    # Productive Development
    "github.com": "development",
    "stackoverflow.com": "development",
    "pypi.org": "development",
    "docs.python.org": "development",
    "npmjs.com": "development",
    "npm.com": "development",
    "crates.io": "development",
    "maven.apache.org": "development",
    
    # Learning & Documentation
    "udemy.com": "learning",
    "coursera.org": "learning",
    "edx.org": "learning",
    "linkedin.com/learning": "learning",
    "w3schools.com": "learning",
    "mdn.mozilla.org": "learning",
    "developer.mozilla.org": "learning",
    "getbootstrap.com": "learning",
    
    # Productivity
    "notion.so": "productivity",
    "trello.com": "productivity",
    "asana.com": "productivity",
    "monday.com": "productivity",
    "slack.com": "productivity",
    "zoom.us": "productivity",
    "meet.google.com": "productivity",
    "calendar.google.com": "productivity",
    
    # Social Media & Entertainment (Distractions)
    "youtube.com": "distraction",
    "netflix.com": "distraction",
    "reddit.com": "distraction",
    "twitter.com": "distraction",
    "facebook.com": "distraction",
    "instagram.com": "distraction",
    "tiktok.com": "distraction",
    "twitch.tv": "distraction",
    "spotify.com": "distraction",
    "9gag.com": "distraction",
    "hacker-news.firebaseapp.com": "distraction",
    
    # Shopping (Context-dependent)
    "amazon.com": "shopping",
    "ebay.com": "shopping",
    "alibaba.com": "shopping",
}

# ===== ALERTS & NOTIFICATIONS =====
ALERT_STYLES = {
    "warning": "🔶",
    "danger": "🔴",
    "success": "✅",
    "info": "ℹ️",
}

# ===== VERSION INFO =====
APP_NAME = "FocusGuard"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Real-Time Adaptive Procrastination Detector"