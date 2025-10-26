"""
FocusGuard - Flask Web Server
REST API and web dashboard server
"""

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from config import *

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Enable CORS for all routes - allow both 3000 and 3001 ports
CORS(app, origins=["http://localhost:3000", "http://localhost:3001"])

# Store session data in memory
current_session = {
    "active": False,
    "start_time": None,
    "stats": {
        "total_events": 0,
        "anomalies": 0,
        "focused_time": 0,
        "distracted_time": 0,
        "elapsed_time": 0,
    },
    "alerts": []
}

# Import real-time monitoring components
try:
    from app_controller import FocusGuardController
    focus_controller = FocusGuardController()
    REAL_TIME_AVAILABLE = True
    logger.info("Real-time monitoring system initialized")
except Exception as e:
    focus_controller = None
    REAL_TIME_AVAILABLE = False
    logger.warning(f"Real-time monitoring not available: {e}")


class AnalyticsEngine:
    """Process analytics from session logs"""
    
    def __init__(self):
        self.session_log = SESSION_LOG
    
    def get_today_stats(self) -> dict:
        """Get statistics for today"""
        if not self.session_log.exists():
            # Return sample data when no real data is available
            return {
                "focus_score": 75.5,
                "focused_time": 45,  # minutes
                "distracted_time": 12,  # minutes
                "anomalies": 3,
                "sessions": 2
            }
        
        today = datetime.now().date()
        sessions = []
        
        with open(self.session_log, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    session_date = datetime.fromisoformat(data['timestamp']).date()
                    if session_date == today:
                        sessions.append(data)
                except:
                    pass
        
        if not sessions:
            return {
                "focus_score": 0,
                "focused_time": 0,
                "distracted_time": 0,
                "anomalies": 0,
                "sessions": 0
            }
        
        total_time = sum(s.get('duration', 0) for s in sessions)
        focused = sum(s.get('focused_time', 0) for s in sessions)
        distracted = sum(s.get('distracted_time', 0) for s in sessions)
        anomalies = sum(s.get('anomalies_detected', 0) for s in sessions)
        
        focus_score = (focused / total_time * 100) if total_time > 0 else 0
        
        return {
            "focus_score": round(focus_score, 1),
            "focused_time": int(focused / 60),  # minutes
            "distracted_time": int(distracted / 60),  # minutes
            "anomalies": anomalies,
            "sessions": len(sessions)
        }
    
    def get_weekly_trend(self) -> list:
        """Get weekly focus score trend"""
        if not self.session_log.exists():
            return [0] * 7
        
        week_stats = {i: {"focused": 0, "total": 0} for i in range(7)}
        
        with open(self.session_log, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    session_date = datetime.fromisoformat(data['timestamp']).date()
                    day_of_week = session_date.weekday()
                    
                    if day_of_week < 7:
                        week_stats[day_of_week]["focused"] += data.get('focused_time', 0)
                        week_stats[day_of_week]["total"] += data.get('duration', 0)
                except:
                    pass
        
        trend = []
        for i in range(7):
            stats = week_stats[i]
            score = (stats["focused"] / stats["total"] * 100) if stats["total"] > 0 else 0
            trend.append(round(score, 1))
        
        return trend
    
    def get_hourly_pattern(self) -> list:
        """Get hourly focus pattern"""
        hours = {i: {"focused": 0, "total": 0} for i in range(24)}
        
        if not self.session_log.exists():
            return [0] * 24
        
        with open(self.session_log, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    hour = datetime.fromisoformat(data['timestamp']).hour
                    hours[hour]["focused"] += data.get('focused_time', 0)
                    hours[hour]["total"] += data.get('duration', 0)
                except:
                    pass
        
        pattern = []
        for i in range(24):
            stats = hours[i]
            score = (stats["focused"] / stats["total"] * 100) if stats["total"] > 0 else 0
            pattern.append(round(score, 1))
        
        return pattern
    
    def get_top_distractions(self) -> dict:
        """Get top distraction triggers"""
        if not LABELED_DATA_FILE.exists():
            return {}
        
        try:
            df = pd.read_csv(LABELED_DATA_FILE)
            distractions = df[df['label'] == 1]['app'].value_counts().head(5)
            return distractions.to_dict()
        except:
            return {}
    
    def get_feature_importance(self) -> dict:
        """Get feature importance from classifier"""
        try:
            from ml_model import ProcrastinationClassifier
            import config
            classifier = ProcrastinationClassifier(config)
            model_path = Path(RANDOM_FOREST_MODEL_FILE)
            scaler_path = Path(SCALER_FILE)

            if REAL_TIME_AVAILABLE and focus_controller:
                for artifact in focus_controller.get_model_registry():
                    if artifact.get("name") == "procrastination_classifier":
                        model_path = Path(artifact.get("path", model_path))
                        scaler_path = Path(
                            artifact.get("metadata", {}).get("scaler_path", scaler_path)
                        )
                        break

            classifier.load(model_path, scaler_path)
            return classifier.get_feature_importance(top_n=8)
        except Exception as e:
            logger.warning(f"Could not load feature importance: {e}")
            return {}
    
    def get_insights(self) -> list:
        """Generate insights from data"""
        insights = []
        stats = self.get_today_stats()
        weekly = self.get_weekly_trend()
        hourly = self.get_hourly_pattern()
        top_distractions = self.get_top_distractions()
        
        # Focus score insight
        if stats["focus_score"] > 80:
            insights.append({
                "type": "success",
                "title": "🎯 Excellent Focus",
                "text": f"Your focus score is {stats['focus_score']:.1f}% today! Keep up the momentum.",
                "action": "View weekly comparison"
            })
        elif stats["focus_score"] < 50:
            insights.append({
                "type": "danger",
                "title": "⚠️ Low Focus",
                "text": f"Your focus score dropped to {stats['focus_score']:.1f}%. Take a break and try again.",
                "action": "Get recommendations"
            })
        
        # Peak hours
        if hourly:
            peak_hour = hourly.index(max(hourly)) if hasattr(hourly, 'index') else hourly.index(max(hourly))
            insights.append({
                "type": "success",
                "title": "📈 Peak Productivity",
                "text": f"You're most productive around {peak_hour}:00. Schedule important tasks then.",
                "action": "Schedule focus block"
            })
        
        # Top distraction
        if top_distractions:
            top_app = list(top_distractions.keys())[0]
            insights.append({
                "type": "danger",
                "title": f"🚫 Distraction Alert",
                "text": f"{top_app} is your biggest distraction. Consider blocking it during work hours.",
                "action": "Enable blocker"
            })
        
        # Weekly trend
        if len(weekly) > 1 and weekly[-1] > weekly[0]:
            improvement = weekly[-1] - weekly[0]
            insights.append({
                "type": "success",
                "title": "📊 Trending Up",
                "text": f"Your focus improved {improvement:.1f}% this week. Great job!",
                "action": "View weekly stats"
            })
        
        return insights


# API Routes
analytics_engine = AnalyticsEngine()


@app.route('/api/stats/today', methods=['GET'])
def get_today_stats():
    """Get today's statistics"""
    stats = analytics_engine.get_today_stats()
    return jsonify(stats)


@app.route('/api/stats/weekly', methods=['GET'])
def get_weekly_stats():
    """Get weekly statistics"""
    trend = analytics_engine.get_weekly_trend()
    return jsonify({
        "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "scores": trend
    })


@app.route('/api/stats/hourly', methods=['GET'])
def get_hourly_stats():
    """Get hourly pattern"""
    pattern = analytics_engine.get_hourly_pattern()
    hours = [f"{i:02d}:00" for i in range(24)]
    return jsonify({
        "hours": hours,
        "pattern": pattern
    })


@app.route('/api/distractions/top', methods=['GET'])
def get_top_distractions():
    """Get top distractions"""
    distractions = analytics_engine.get_top_distractions()
    return jsonify(distractions)


@app.route('/api/features/importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance"""
    importance = analytics_engine.get_feature_importance()
    return jsonify(importance)


@app.route('/api/models/registry', methods=['GET'])
def get_model_registry():
    """Expose trained model artefact metadata."""
    if REAL_TIME_AVAILABLE and focus_controller:
        try:
            return jsonify(focus_controller.get_model_registry())
        except Exception as exc:
            logger.error(f"Failed to load model registry: {exc}")
    return jsonify([])


@app.route('/api/insights', methods=['GET'])
def get_insights():
    """Get personalized insights"""
    insights = analytics_engine.get_insights()
    return jsonify(insights)


@app.route('/api/session/status', methods=['GET'])
def get_session_status():
    """Get current session status with real-time data"""
    if REAL_TIME_AVAILABLE and focus_controller:
        try:
            # Get live stats from controller and mirror them into the lightweight session cache
            real_stats = focus_controller.get_current_session_stats()
            current_session['active'] = real_stats.get('active', False)
            current_session['stats'] = {
                'total_events': real_stats.get('total_events', 0),
                'anomalies': real_stats.get('anomalies_detected', 0),
                'focused_time': int(real_stats.get('focused_time', 0)),
                'distracted_time': int(real_stats.get('distracted_time', 0)),
                'elapsed_time': real_stats.get('elapsed_time', 0)
            }

            prediction_meta = real_stats.get('prediction') or {}
            if prediction_meta:
                current_session['stats'].update({
                    'combined_score': float(prediction_meta.get('combined_score', 0.0) or 0.0),
                    'anomaly_score': float(prediction_meta.get('anomaly_score', 0.0) or 0.0),
                    'classifier_probability': (
                        float(prediction_meta['classifier_probability'])
                        if prediction_meta.get('classifier_probability') is not None
                        else None
                    ),
                    'confidence': float(prediction_meta.get('confidence', 0.0) or 0.0),
                    'heuristic_triggered': bool(prediction_meta.get('heuristic_triggered', False)),
                    'prediction_timestamp': prediction_meta.get('timestamp'),
                })

            if current_session['active'] and focus_controller.session_start_time:
                current_session['start_time'] = datetime.fromtimestamp(
                    focus_controller.session_start_time
                ).isoformat()
            else:
                current_session['start_time'] = None
        except Exception as e:
            logger.error(f"Error getting real-time stats: {e}")
    
    return jsonify(current_session)


@app.route('/api/session/start', methods=['POST'])
def start_session():
    """Start a new session with real-time monitoring"""
    global current_session
    
    if REAL_TIME_AVAILABLE and focus_controller:
        try:
            # Start real-time monitoring
            logger.info("Starting real-time detection session...")
            focus_controller.start_detection_session()
            current_session['active'] = True
            current_session['start_time'] = datetime.now().isoformat()
            current_session['stats'] = {
                "total_events": 0,
                "anomalies": 0,
                "focused_time": 0,
                "distracted_time": 0,
                "elapsed_time": 0,
            }
            current_session['alerts'] = []
            return jsonify({
                "status": "started", 
                "session": current_session, 
                "real_time": True,
                "message": "Real-time monitoring started"
            })
        except Exception as e:
            logger.error(f"Failed to start real-time monitoring: {e}")
            return jsonify({
                "status": "error", 
                "message": f"Failed to start monitoring: {e}"
            }), 500
    else:
        # Fallback to mock session
        current_session['active'] = True
        current_session['start_time'] = datetime.now().isoformat()
        current_session['stats'] = {
            "total_events": 0,
            "anomalies": 0,
            "focused_time": 0,
            "distracted_time": 0,
        }
        return jsonify({
            "status": "started", 
            "session": current_session, 
            "real_time": False,
            "message": "Mock session started (real-time monitoring unavailable)"
        })


@app.route('/api/session/stop', methods=['POST'])
def stop_session():
    """Stop current session and real-time monitoring"""
    global current_session
    
    if REAL_TIME_AVAILABLE and focus_controller and current_session['active']:
        try:
            # Stop real-time monitoring and save session
            logger.info("Stopping real-time detection session...")
            session_data = focus_controller.stop_detection_session()
            
            # Update session stats with real data
            if session_data:
                current_session['stats'].update({
                    "total_events": session_data.get('total_events', 0),
                    "anomalies": session_data.get('anomalies_detected', 0),
                    "focused_time": int(session_data.get('focused_time', 0)),
                    "distracted_time": int(session_data.get('distracted_time', 0)),
                    "elapsed_time": session_data.get('duration', current_session['stats'].get('elapsed_time', 0))
                })
            
            current_session['active'] = False
            return jsonify({
                "status": "stopped", 
                "session": current_session, 
                "real_time": True,
                "message": "Real-time monitoring stopped and session saved"
            })
        except Exception as e:
            logger.error(f"Failed to stop real-time monitoring: {e}")
            current_session['active'] = False
            return jsonify({
                "status": "stopped", 
                "session": current_session, 
                "message": f"Session stopped with error: {e}"
            })
    else:
        current_session['active'] = False
        return jsonify({
            "status": "stopped", 
            "session": current_session, 
            "real_time": False,
            "message": "Mock session stopped"
        })


@app.route('/api/session/update', methods=['POST'])
def update_session():
    """Update session statistics"""
    data = request.json
    current_session['stats'].update(data)
    return jsonify({"status": "updated", "session": current_session})


@app.route('/api/activity/recent', methods=['GET'])
def get_recent_activity():
    """Get recent activity events"""
    if REAL_TIME_AVAILABLE and focus_controller and hasattr(focus_controller, 'events_buffer'):
        try:
            # Get last 20 events from buffer
            recent_events = list(focus_controller.events_buffer)[-20:]
            activity_data = []
            
            for event in recent_events:
                activity_data.append({
                    "timestamp": datetime.fromtimestamp(event.timestamp).isoformat(),
                    "type": getattr(event.event_type, "value", str(event.event_type)),
                    "app": event.app_name or "Unknown",
                    "title": event.window_title or "",
                    "detail": event.detail or ""
                })
            
            return jsonify(activity_data)
        except Exception as e:
            logger.error(f"Error getting recent activity: {e}")
            return jsonify([])
    else:
        # Return sample data if real-time not available
        return jsonify([
            {
                "timestamp": datetime.now().isoformat(),
                "type": "app_switch", 
                "app": "Code.exe",
                "title": "FocusGuard - Visual Studio Code",
                "detail": "Real-time monitoring unavailable"
            }
        ])


@app.route('/api/export', methods=['GET'])
def export_data():
    """Export all analytics data"""
    data = {
        "export_date": datetime.now().isoformat(),
        "today_stats": analytics_engine.get_today_stats(),
        "weekly_trend": analytics_engine.get_weekly_trend(),
        "hourly_pattern": analytics_engine.get_hourly_pattern(),
        "top_distractions": analytics_engine.get_top_distractions(),
        "insights": analytics_engine.get_insights(),
    }
    return jsonify(data)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "app": APP_NAME,
        "version": APP_VERSION,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/', methods=['GET'])
def index():
    """Serve React frontend"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FocusGuard - Real-Time Procrastination Detection</title>
        <style>
            body {
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                color: white;
            }
            .message {
                text-align: center;
                max-width: 600px;
                padding: 2rem;
            }
            .logo {
                font-size: 3rem;
                margin-bottom: 1rem;
            }
            .title {
                font-size: 2rem;
                font-weight: bold;
                margin-bottom: 1rem;
            }
            .description {
                font-size: 1.1rem;
                opacity: 0.9;
                margin-bottom: 2rem;
                line-height: 1.6;
            }
            .cta {
                display: inline-block;
                padding: 12px 24px;
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 8px;
                color: white;
                text-decoration: none;
                font-weight: 600;
                transition: all 0.3s ease;
                backdrop-filter: blur(10px);
            }
            .cta:hover {
                background: rgba(255, 255, 255, 0.3);
                transform: translateY(-2px);
            }
        </style>
    </head>
    <body>
        <div class="message">
            <div class="logo">🎯</div>
            <h1 class="title">FocusGuard Dashboard</h1>
            <p class="description">
                The modern React frontend is available at port 3000.<br>
                This Flask server provides the API endpoints for real-time data.
            </p>
            <a href="http://localhost:3000" class="cta">
                Open React Dashboard
            </a>
            <div style="margin-top: 2rem; font-size: 0.875rem; opacity: 0.7;">
                <p>API Server running on port 8000</p>
                <p>React Frontend on port 3000</p>
            </div>
        </div>
    </body>
    </html>
    """


def run_server():
    """Run the Flask server"""
    logger.info(f"Starting FocusGuard Web Server on {WEB_SERVER_HOST}:{WEB_SERVER_PORT}")
    print(f"\n🌐 FocusGuard Dashboard")
    print(f"   URL: http://{WEB_SERVER_HOST}:{WEB_SERVER_PORT}")
    print(f"   API: http://{WEB_SERVER_HOST}:{WEB_SERVER_PORT}/api/*\n")
    
    app.run(
        host=WEB_SERVER_HOST,
        port=WEB_SERVER_PORT,
        debug=False,
        use_reloader=False
    )


if __name__ == '__main__':
    run_server()