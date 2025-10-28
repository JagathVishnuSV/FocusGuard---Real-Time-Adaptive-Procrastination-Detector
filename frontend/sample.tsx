import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, Radar } from 'recharts';
import { AlertCircle, CheckCircle, TrendingUp, Activity, Zap, Target, Brain, Clock, Award, Menu, X, Home, TrendingDown, Calendar, Lightbulb, ChevronRight, Trophy, Flame, Timer, BarChart3 } from 'lucide-react';

// API Base URL
const API_BASE = 'http://127.0.0.1:8000';

// Types
interface SessionStatus {
  focus_score: number;
  focused_minutes: number;
  distracted_minutes: number;
  session_count: number;
  current_state: 'focused' | 'distracted' | 'idle';
}

interface ActivityEvent {
  timestamp: string;
  label: 'Focused' | 'Distracted';
  distraction_score: number;
  context?: string;
}

interface Insight {
  severity: 'alert' | 'opportunity' | 'success';
  message: string;
  cta?: string;
}

interface DistractionSource {
  name: string;
  count: number;
}

// Custom Hooks
const useFetch = <T,>(url: string, interval: number = 5000) => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const json = await response.json();
      setData(json);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch');
    } finally {
      setLoading(false);
    }
  }, [url]);

  useEffect(() => {
    fetchData();
    const timer = setInterval(fetchData, interval);
    return () => clearInterval(timer);
  }, [fetchData, interval]);

  return { data, loading, error, refetch: fetchData };
};

// Shared Components
const ProgressRing = ({ score, size = 120 }: { score: number; size?: number }) => {
  const radius = (size - 20) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="transform -rotate-90">
        <circle cx={size / 2} cy={size / 2} r={radius} stroke="currentColor" strokeWidth="8" fill="none" className="text-gray-200" />
        <circle cx={size / 2} cy={size / 2} r={radius} stroke="currentColor" strokeWidth="8" fill="none" strokeDasharray={circumference} strokeDashoffset={offset} className="text-blue-600 transition-all duration-1000 ease-out" strokeLinecap="round" />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-bold text-gray-900">{score}</span>
        <span className="text-xs text-gray-500">Score</span>
      </div>
    </div>
  );
};

const StatCard = ({ icon: Icon, label, value, color, subtitle }: any) => (
  <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100 hover:shadow-lg transition-all hover:scale-105">
    <div className="flex items-center gap-3">
      <div className={`p-3 rounded-lg ${color}`}>
        <Icon className="w-5 h-5 text-white" />
      </div>
      <div>
        <p className="text-sm text-gray-600">{label}</p>
        <p className="text-2xl font-bold text-gray-900">{value}</p>
        {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
      </div>
    </div>
  </div>
);

const Toast = ({ message, onClose }: { message: string; onClose: () => void }) => (
  <div className="fixed bottom-4 right-4 bg-gray-900 text-white px-4 py-3 rounded-lg shadow-lg animate-in fade-in slide-in-from-bottom-4 z-50">
    <div className="flex items-center gap-2">
      <CheckCircle className="w-5 h-5" />
      <span>{message}</span>
      <button onClick={onClose} className="ml-2 text-gray-400 hover:text-white">×</button>
    </div>
  </div>
);

// Dashboard Page
const DashboardPage = ({ session, activities, insights, distractions, onFeedback }: any) => {
  const [currentStreak, setCurrentStreak] = useState(0);

  useEffect(() => {
    let streak = 0;
    activities?.slice().reverse().forEach((activity: ActivityEvent) => {
      if (activity.label === 'Focused') streak++;
      else streak = 0;
    });
    setCurrentStreak(streak);
  }, [activities]);

  const activityData = activities?.slice(-20).map((a: ActivityEvent, i: number) => ({
    time: i,
    score: a.distraction_score
  })) || [];

  const contextData = activities?.reduce((acc: any, a: ActivityEvent) => {
    const ctx = a.context || 'Unknown';
    acc[ctx] = (acc[ctx] || 0) + 1;
    return acc;
  }, {} as Record<string, number>) || {};

  const pieData = Object.entries(contextData).map(([name, value]) => ({ name, value }));
  const COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'];

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <div className="lg:col-span-1 flex justify-center items-center bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <ProgressRing score={session?.focus_score || 0} />
        </div>
        <StatCard icon={Target} label="Focused" value={session?.focused_minutes || 0} subtitle="minutes today" color="bg-green-500" />
        <StatCard icon={AlertCircle} label="Distracted" value={session?.distracted_minutes || 0} subtitle="minutes today" color="bg-red-500" />
        <StatCard icon={Award} label="Sessions" value={session?.session_count || 0} subtitle="completed" color="bg-blue-500" />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-600" />
            Distraction Trends (ML Predictions)
          </h2>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={activityData}>
              <XAxis dataKey="time" hide />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Line type="monotone" dataKey="score" stroke="#3b82f6" strokeWidth={3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl p-6 text-white shadow-lg">
          <div className="flex items-center gap-2 mb-4">
            <Flame className="w-6 h-6" />
            <h3 className="text-lg font-semibold">Focus Streak</h3>
          </div>
          <div className="text-center">
            <p className="text-5xl font-bold mb-2">{currentStreak}</p>
            <p className="text-sm opacity-90">Consecutive Focused Activities</p>
          </div>
          {currentStreak >= 5 && (
            <div className="mt-4 p-3 bg-white/20 rounded-lg backdrop-blur-sm animate-in fade-in">
              <p className="text-sm font-medium text-center">🔥 On fire! Keep it up!</p>
            </div>
          )}
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-green-600" />
            AI Insights
          </h2>
          <div className="space-y-3">
            {insights?.length ? insights.map((insight: Insight, idx: number) => {
              const config = {
                alert: { icon: AlertCircle, color: 'bg-red-50 border-red-200', iconColor: 'text-red-600' },
                opportunity: { icon: TrendingUp, color: 'bg-yellow-50 border-yellow-200', iconColor: 'text-yellow-600' },
                success: { icon: CheckCircle, color: 'bg-green-50 border-green-200', iconColor: 'text-green-600' }
              };
              const { icon: Icon, color, iconColor } = config[insight.severity];
              return (
                <div key={idx} className={`${color} border rounded-lg p-4 transition-all hover:shadow-md`}>
                  <div className="flex items-start gap-3">
                    <Icon className={`w-5 h-5 ${iconColor} flex-shrink-0 mt-0.5`} />
                    <p className="text-sm text-gray-800">{insight.message}</p>
                  </div>
                </div>
              );
            }) : <p className="text-gray-500 text-sm">No insights yet. Keep working!</p>}
          </div>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-orange-600" />
            Top Distractions
          </h2>
          {pieData.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={180}>
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" innerRadius={50} outerRadius={70} paddingAngle={5} dataKey="value">
                    {pieData.map((_: any, idx: number) => <Cell key={idx} fill={COLORS[idx % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              <div className="mt-4 space-y-2">
                {distractions?.slice(0, 5).map((d: DistractionSource, idx: number) => (
                  <div key={idx} className="flex justify-between items-center text-sm p-2 hover:bg-gray-50 rounded transition-colors">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[idx % COLORS.length] }}></div>
                      <span className="text-gray-700">{d.name}</span>
                    </div>
                    <span className="font-medium text-gray-900">{d.count}</span>
                  </div>
                ))}
              </div>
            </>
          ) : <p className="text-gray-500 text-sm">No distraction data yet.</p>}
        </div>
      </div>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5 text-indigo-600" />
          Recent Activity (ML Classified)
        </h2>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {activities?.slice(0, 10).map((activity: ActivityEvent, idx: number) => (
            <div key={idx} className="bg-gray-50 rounded-lg p-3 hover:bg-gray-100 transition-all animate-in fade-in" style={{ animationDelay: `${idx * 50}ms` }}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${activity.label === 'Focused' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                    {activity.label}
                  </span>
                  <span className="text-sm text-gray-600">{new Date(activity.timestamp).toLocaleTimeString()}</span>
                  {activity.context && <span className="text-xs text-gray-500 italic">{activity.context}</span>}
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-xs text-gray-500">Score: {activity.distraction_score.toFixed(2)}</span>
                  <button onClick={() => onFeedback(activity.timestamp, activity.label === 'Focused' ? 'Distracted' : 'Focused')} className="text-xs text-blue-600 hover:text-blue-700 font-medium px-2 py-1 hover:bg-blue-50 rounded transition-colors">
                    Correct
                  </button>
                </div>
              </div>
            </div>
          )) || <p className="text-gray-500 text-sm">No recent activity.</p>}
        </div>
      </div>
    </div>
  );
};

// Focus Deep Dive Page
const DeepDivePage = ({ activities, session }: any) => {
  const hours = Array.from({ length: 24 }, (_, i) => i);
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  
  const heatmapData = hours.map(hour => ({
    hour,
    focus: Math.random() * 100
  }));

  const contextCorrelation = activities?.reduce((acc: any, a: ActivityEvent) => {
    const ctx = a.context || 'Unknown';
    if (!acc[ctx]) acc[ctx] = { focused: 0, distracted: 0 };
    if (a.label === 'Focused') acc[ctx].focused++;
    else acc[ctx].distracted++;
    return acc;
  }, {}) || {};

  const correlationData = Object.entries(contextCorrelation).map(([context, data]: any) => ({
    context,
    focusRate: (data.focused / (data.focused + data.distracted)) * 100
  }));

  const timeToFirstDistraction = activities?.findIndex((a: ActivityEvent) => a.label === 'Distracted') || 0;
  const focusPersona = session?.focus_score > 80 ? '🏃 The Morning Sprinter' : session?.focus_score > 60 ? '🏃‍♀️ The Consistent Cruiser' : '🐢 The Steady Builder';

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
      <div className="bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl p-6 text-white shadow-lg">
        <h2 className="text-2xl font-bold mb-2">Your Focus Persona</h2>
        <p className="text-3xl mb-2">{focusPersona}</p>
        <p className="text-sm opacity-90">Based on your recent focus patterns</p>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        <StatCard icon={Timer} label="Time to First Distraction" value={`${timeToFirstDistraction} min`} color="bg-indigo-500" />
        <StatCard icon={Target} label="Focus Efficiency" value={`${session?.focus_score || 0}%`} color="bg-green-500" />
        <StatCard icon={BarChart3} label="Peak Hour" value="10 AM" subtitle="Your best time" color="bg-orange-500" />
      </div>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Calendar className="w-5 h-5 text-blue-600" />
          Focus Heatmap (Hour of Day)
        </h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={heatmapData}>
            <XAxis dataKey="hour" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="focus" fill="#3b82f6" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-green-600" />
          Context Correlation Analysis
        </h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={correlationData} layout="vertical">
            <XAxis type="number" domain={[0, 100]} />
            <YAxis type="category" dataKey="context" width={100} />
            <Tooltip />
            <Bar dataKey="focusRate" fill="#10b981" radius={[0, 8, 8, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <p className="text-sm text-gray-600 mt-4">Higher percentages indicate better focus in that context.</p>
      </div>
    </div>
  );
};

// Goals & Streaks Page
const GoalsPage = ({ session }: any) => {
  const [dailyGoal, setDailyGoal] = useState(120);
  const [weeklyGoal, setWeeklyGoal] = useState(600);
  const [bestStreak, setBestStreak] = useState(7);
  const [currentStreak, setCurrentStreak] = useState(3);

  const dailyProgress = ((session?.focused_minutes || 0) / dailyGoal) * 100;
  const weeklyProgress = ((session?.focused_minutes || 0) * 5 / weeklyGoal) * 100;

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
      <div className="grid gap-6 md:grid-cols-2">
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Target className="w-5 h-5 text-blue-600" />
            Daily Goal
          </h2>
          <div className="flex justify-center mb-4">
            <ProgressRing score={Math.min(dailyProgress, 100)} size={140} />
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-gray-900">{session?.focused_minutes || 0} / {dailyGoal} min</p>
            <p className="text-sm text-gray-600 mt-1">{Math.round(dailyProgress)}% completed</p>
          </div>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Calendar className="w-5 h-5 text-purple-600" />
            Weekly Goal
          </h2>
          <div className="flex justify-center mb-4">
            <ProgressRing score={Math.min(weeklyProgress, 100)} size={140} />
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-gray-900">{(session?.focused_minutes || 0) * 5} / {weeklyGoal} min</p>
            <p className="text-sm text-gray-600 mt-1">{Math.round(weeklyProgress)}% completed</p>
          </div>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <div className="bg-gradient-to-br from-orange-400 to-red-500 rounded-xl p-6 text-white shadow-lg">
          <div className="flex items-center gap-2 mb-4">
            <Flame className="w-6 h-6" />
            <h3 className="text-lg font-semibold">Current Streak</h3>
          </div>
          <p className="text-5xl font-bold text-center">{currentStreak}</p>
          <p className="text-center text-sm opacity-90 mt-2">days in a row!</p>
        </div>

        <div className="bg-gradient-to-br from-yellow-400 to-orange-500 rounded-xl p-6 text-white shadow-lg">
          <div className="flex items-center gap-2 mb-4">
            <Trophy className="w-6 h-6" />
            <h3 className="text-lg font-semibold">Best Streak</h3>
          </div>
          <p className="text-5xl font-bold text-center">{bestStreak}</p>
          <p className="text-center text-sm opacity-90 mt-2">personal record!</p>
        </div>
      </div>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Award className="w-5 h-5 text-yellow-600" />
          Achievements
        </h2>
        <div className="grid gap-4 md:grid-cols-3">
          {[
            { name: 'Perfect Session', desc: '0 distractions', earned: true },
            { name: 'Focus Master', desc: '100+ min focus', earned: true },
            { name: 'Week Warrior', desc: '7 day streak', earned: false }
          ].map((achievement, idx) => (
            <div key={idx} className={`p-4 rounded-lg border-2 transition-all ${achievement.earned ? 'bg-yellow-50 border-yellow-300' : 'bg-gray-50 border-gray-200 opacity-50'}`}>
              <div className="text-3xl mb-2">{achievement.earned ? '🏆' : '🔒'}</div>
              <h3 className="font-semibold text-gray-900">{achievement.name}</h3>
              <p className="text-sm text-gray-600">{achievement.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Predictive Insights Page
const PredictivePage = ({ activities }: any) => {
  const currentHour = new Date().getHours();
  const distractionRisk = currentHour >= 14 && currentHour <= 16 ? 'High' : currentHour >= 10 && currentHour <= 12 ? 'Low' : 'Medium';
  const riskColor = distractionRisk === 'High' ? 'text-red-600 bg-red-50' : distractionRisk === 'Low' ? 'text-green-600 bg-green-50' : 'text-yellow-600 bg-yellow-50';

  const hourlyPredictions = Array.from({ length: 24 }, (_, i) => ({
    hour: `${i}:00`,
    predicted: Math.random() * 100
  }));

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
      <div className={`${riskColor} rounded-xl p-6 border-2 shadow-lg`}>
        <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
          <AlertCircle className="w-6 h-6" />
          Distraction Risk: {distractionRisk}
        </h2>
        <p className="text-sm opacity-90">Based on your historical patterns and current time</p>
      </div>

      {distractionRisk === 'High' && (
        <div className="bg-blue-50 border-2 border-blue-200 rounded-xl p-6">
          <div className="flex items-start gap-3">
            <Lightbulb className="w-6 h-6 text-blue-600 flex-shrink-0" />
            <div>
              <h3 className="font-semibold text-gray-900 mb-2">Proactive Focus Tip</h3>
              <p className="text-sm text-gray-700">Your distraction risk is high for the next hour. Consider putting your phone on silent, closing unnecessary tabs, and using a Pomodoro timer.</p>
            </div>
          </div>
        </div>
      )}

      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-purple-600" />
          Focus Score Forecast (Next 24 Hours)
        </h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={hourlyPredictions}>
            <XAxis dataKey="hour" />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Line type="monotone" dataKey="predicted" stroke="#8b5cf6" strokeWidth={3} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Brain className="w-5 h-5 text-indigo-600" />
          Context-Specific Recommendations
        </h2>
        <div className="space-y-3">
          {[
            { context: 'Email', tip: 'Try time-boxing email checks to 15-minute blocks' },
            { context: 'Coding', tip: 'Your focus is highest in the morning - schedule complex tasks then' },
            { context: 'Meetings', tip: 'You tend to get distracted after 45 minutes - suggest shorter meetings' }
          ].map((rec, idx) => (
            <div key={idx} className="p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-semibold text-gray-900">{rec.context}</h3>
                  <p className="text-sm text-gray-600 mt-1">{rec.tip}</p>
                </div>
                <ChevronRight className="w-5 h-5 text-gray-400" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Main App with Sidebar
export default function FocusGuard() {
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [toast, setToast] = useState<string | null>(null);

  const { data: session, loading: sessionLoading } = useFetch<SessionStatus>(`${API_BASE}/api/session/status`);
  const { data: activities, refetch: refetchActivities } = useFetch<ActivityEvent[]>(`${API_BASE}/api/activity/recent`);
  const { data: insights } = useFetch<Insight[]>(`${API_BASE}/api/insights`);
  const { data: distractions } = useFetch<DistractionSource[]>(`${API_BASE}/api/distractions/top`);

  const handleFeedback = async (timestamp: string, newLabel: string) => {
    try {
      await fetch(`${API_BASE}/api/personalization/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ timestamp, label: newLabel })
      });
      setToast('Feedback submitted! ML models are learning...');
      setTimeout(() => setToast(null), 3000);
      refetchActivities();
    } catch (err) {
      setToast('Failed to submit feedback');
      setTimeout(() => setToast(null), 3000);
    }
  };

  const navItems = [
    { id: 'dashboard', icon: Home, label: 'Dashboard' },
    { id: 'deepdive', icon: BarChart3, label: 'Focus Deep Dive' },
    { id: 'goals', icon: Target, label: 'Goals & Streaks' },
    { id: 'predictive', icon: Lightbulb, label: 'Predictive Insights' }
  ];

  if (sessionLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Brain className="w-12 h-12 text-blue-600 animate-pulse mx-auto mb-4" />
          <p className="text-gray-600">Loading AI insights...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <aside className={`${sidebarOpen ? 'w-64' : 'w-20'} bg-white border-r border-gray-200 transition-all duration-300 ease-in-out flex flex-col shadow-lg`}>
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            {sidebarOpen ? (
              <div className="flex items-center gap-3">
                <Brain className="w-8 h-8 text-blue-600" />
                <div>
                  <h1 className="text-xl font-bold text-gray-900">FocusGuard</h1>
                  <p className="text-xs text-gray-500">AI-Powered</p>
                </div>
              </div>
            ) : (
              <Brain className="w-8 h-8 text-blue-600 mx-auto" />
            )}
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              aria-label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
            >
              {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        <nav className="flex-1 p-4 space-y-2">
          {navItems.map(({ id, icon: Icon, label }) => (
            <button
              key={id}
              onClick={() => setCurrentPage(id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                currentPage === id
                  ? 'bg-blue-50 text-blue-600 font-medium shadow-sm'
                  : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
              }`}
              title={!sidebarOpen ? label : undefined}
            >
              <Icon className="w-5 h-5 flex-shrink-0" />
              {sidebarOpen && <span className="text-sm">{label}</span>}
              {currentPage === id && sidebarOpen && (
                <ChevronRight className="w-4 h-4 ml-auto" />
              )}
            </button>
          ))}
        </nav>

        <div className="p-4 border-t border-gray-200">
          {sidebarOpen ? (
            <div className="bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg p-4 text-white">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-5 h-5" />
                <span className="text-sm font-semibold">Current State</span>
              </div>
              <p className="text-xs opacity-90">
                {session?.current_state === 'focused' ? '🎯 Focused' : 
                 session?.current_state === 'distracted' ? '⚠️ Distracted' : '💤 Idle'}
              </p>
            </div>
          ) : (
            <div className="text-2xl text-center">
              {session?.current_state === 'focused' ? '🎯' : 
               session?.current_state === 'distracted' ? '⚠️' : '💤'}
            </div>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        <header className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-10">
          <div className="px-8 py-4">
            <h2 className="text-2xl font-bold text-gray-900">
              {navItems.find(item => item.id === currentPage)?.label}
            </h2>
            <p className="text-sm text-gray-600 mt-1">
              {currentPage === 'dashboard' && 'Your real-time focus overview'}
              {currentPage === 'deepdive' && 'Explore your productivity patterns in depth'}
              {currentPage === 'goals' && 'Track your progress and achievements'}
              {currentPage === 'predictive' && 'AI-powered focus predictions and recommendations'}
            </p>
          </div>
        </header>

        <div className="p-8">
          {currentPage === 'dashboard' && (
            <DashboardPage
              session={session}
              activities={activities}
              insights={insights}
              distractions={distractions}
              onFeedback={handleFeedback}
            />
          )}
          {currentPage === 'deepdive' && (
            <DeepDivePage activities={activities} session={session} />
          )}
          {currentPage === 'goals' && (
            <GoalsPage session={session} />
          )}
          {currentPage === 'predictive' && (
            <PredictivePage activities={activities} />
          )}
        </div>
      </main>

      {toast && <Toast message={toast} onClose={() => setToast(null)} />}
    </div>
  );
}