"""
FocusGuard - Flask Web Server
REST API and web dashboard server
"""

from flask import Flask, render_template_string, jsonify, request
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

# Store session data in memory
current_session = {
    "active": False,
    "start_time": None,
    "stats": {
        "total_events": 0,
        "anomalies": 0,
        "focused_time": 0,
        "distracted_time": 0,
    },
    "alerts": []
}


class AnalyticsEngine:
    """Process analytics from session logs"""
    
    def __init__(self):
        self.session_log = SESSION_LOG
    
    def get_today_stats(self) -> dict:
        """Get statistics for today"""
        if not self.session_log.exists():
            return {
                "focus_score": 0,
                "focused_time": 0,
                "distracted_time": 0,
                "anomalies": 0,
                "sessions": 0
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
            classifier.load(
                RANDOM_FOREST_MODEL_FILE,
                SCALER_FILE
            )
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


@app.route('/api/insights', methods=['GET'])
def get_insights():
    """Get personalized insights"""
    insights = analytics_engine.get_insights()
    return jsonify(insights)


@app.route('/api/session/status', methods=['GET'])
def get_session_status():
    """Get current session status"""
    return jsonify(current_session)


@app.route('/api/session/start', methods=['POST'])
def start_session():
    """Start a new session"""
    current_session['active'] = True
    current_session['start_time'] = datetime.now().isoformat()
    current_session['stats'] = {
        "total_events": 0,
        "anomalies": 0,
        "focused_time": 0,
        "distracted_time": 0,
    }
    return jsonify({"status": "started", "session": current_session})


@app.route('/api/session/stop', methods=['POST'])
def stop_session():
    """Stop current session"""
    current_session['active'] = False
    return jsonify({"status": "stopped", "session": current_session})


@app.route('/api/session/update', methods=['POST'])
def update_session():
    """Update session statistics"""
    data = request.json
    current_session['stats'].update(data)
    return jsonify({"status": "updated", "session": current_session})


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
    """Serve dashboard"""
    # Read dashboard HTML from file or embed it
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FocusGuard Dashboard</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #1f2937; min-height: 100vh; padding: 20px; }
            .container { max-width: 1400px; margin: 0 auto; }
            header { background: white; padding: 30px; border-radius: 12px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); 
                     display: flex; justify-content: space-between; align-items: center; }
            h1 { font-size: 28px; margin-bottom: 5px; color: #6366f1; }
            .status-badge { padding: 8px 16px; border-radius: 20px; font-size: 13px; font-weight: 600; 
                           background: #10b981; color: white; }
            button { padding: 10px 20px; border: none; border-radius: 6px; background: #6366f1; 
                    color: white; cursor: pointer; font-weight: 600; }
            button:hover { background: #4f46e5; }
            .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .card { background: white; padding: 24px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
            .card-title { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #9ca3af; margin-bottom: 15px; }
            .card-value { font-size: 32px; font-weight: 700; color: #6366f1; }
            .chart-card { background: white; padding: 24px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin-bottom: 20px; }
            .chart-container { position: relative; height: 400px; }
            .insight-card { background: white; padding: 24px; border-radius: 12px; border-left: 4px solid #6366f1; margin-bottom: 20px; }
            .insight-title { font-size: 14px; font-weight: 700; margin-bottom: 10px; }
            .insight-text { font-size: 13px; color: #6b7280; line-height: 1.6; }
            @media (max-width: 768px) { header { flex-direction: column; gap: 20px; } }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div>
                    <h1>🎯 FocusGuard</h1>
                    <p>Real-Time Procrastination Detection</p>
                </div>
                <div>
                    <span class="status-badge" id="status">Ready</span>
                </div>
            </header>
            
            <div class="dashboard-grid">
                <div class="card">
                    <div class="card-title">Focus Score</div>
                    <div class="card-value" id="focusScore">--</div>
                </div>
                <div class="card">
                    <div class="card-title">Focused Time</div>
                    <div class="card-value" id="focusedTime">--</div>
                </div>
                <div class="card">
                    <div class="card-title">Distracted Time</div>
                    <div class="card-value" id="distractedTime">--</div>
                </div>
                <div class="card">
                    <div class="card-title">Anomalies</div>
                    <div class="card-value" id="anomalies">--</div>
                </div>
            </div>
            
            <div class="chart-card">
                <h2 style="margin-bottom: 20px;">Weekly Trend</h2>
                <div class="chart-container">
                    <canvas id="weeklyChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <h2 style="margin-bottom: 20px;">Hourly Pattern</h2>
                <div class="chart-container">
                    <canvas id="hourlyChart"></canvas>
                </div>
            </div>
            
            <div id="insightsContainer"></div>
        </div>
        
        <script>
            async function loadStats() {
                try {
                    const today = await fetch('/api/stats/today').then(r => r.json());
                    const weekly = await fetch('/api/stats/weekly').then(r => r.json());
                    const hourly = await fetch('/api/stats/hourly').then(r => r.json());
                    const insights = await fetch('/api/insights').then(r => r.json());
                    
                    document.getElementById('focusScore').textContent = today.focus_score + '%';
                    document.getElementById('focusedTime').textContent = today.focused_time + 'm';
                    document.getElementById('distractedTime').textContent = today.distracted_time + 'm';
                    document.getElementById('anomalies').textContent = today.anomalies;
                    
                    // Weekly chart
                    new Chart(document.getElementById('weeklyChart'), {
                        type: 'line',
                        data: {
                            labels: weekly.days,
                            datasets: [{
                                label: 'Focus Score',
                                data: weekly.scores,
                                borderColor: '#6366f1',
                                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                                tension: 0.4,
                                fill: true
                            }]
                        },
                        options: { responsive: true, maintainAspectRatio: false }
                    });
                    
                    // Hourly chart
                    new Chart(document.getElementById('hourlyChart'), {
                        type: 'bar',
                        data: {
                            labels: hourly.hours,
                            datasets: [{
                                label: 'Focus %',
                                data: hourly.pattern,
                                backgroundColor: '#6366f1'
                            }]
                        },
                        options: { responsive: true, maintainAspectRatio: false }
                    });
                    
                    // Insights
                    const container = document.getElementById('insightsContainer');
                    insights.forEach(insight => {
                        const card = document.createElement('div');
                        card.className = 'insight-card';
                        card.innerHTML = `
                            <div class="insight-title">${insight.title}</div>
                            <div class="insight-text">${insight.text}</div>
                        `;
                        container.appendChild(card);
                    });
                } catch (error) {
                    console.error('Error loading stats:', error);
                }
            }
            
            loadStats();
            setInterval(loadStats, 30000); // Refresh every 30 seconds
        </script>
    </body>
    </html>
    """
    return render_template_string(dashboard_html)


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