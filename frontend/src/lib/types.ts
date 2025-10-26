// src/lib/types.ts

export interface TodayStats {
  focus_score: number;
  focused_time: number;
  distracted_time: number;
  anomalies: number;
  sessions: number;
  change?: number; // Optional property for trend
  context_switches?: number;
  session_count?: number;
}

export interface WeeklyStats {
  days: string[];
  scores: number[];
}

export interface HourlyStats {
  hours: string[];
  pattern: number[];
}

export interface Insight {
  type: 'success' | 'warning' | 'danger' | 'info';
  title: string;
  text: string;
  action: string;
}

export interface SessionStatus {
  active: boolean;
  is_running?: boolean; // Backend might send this
  start_time: string | null;
  stats: {
    total_events: number;
    anomalies: number;
    focused_time: number;
    distracted_time: number;
    elapsed_time?: number;
    combined_score?: number;
    anomaly_score?: number;
    classifier_probability?: number | null;
    confidence?: number;
    heuristic_triggered?: boolean;
    prediction_timestamp?: string | null;
  };
  alerts: any[];
  prediction?: PredictionSummary | null;
}

export interface PredictionSummary {
  combined_score?: number;
  anomaly_score?: number;
  classifier_probability?: number | null;
  confidence?: number;
  heuristic_triggered?: boolean;
  timestamp?: string | null;
}

export interface HealthStatus {
  status: string;
  app: string;
  version: string;
  timestamp: string;
}

export interface ActivityEvent {
  timestamp: string;
  type: string;
  app: string;
  title?: string;
  detail?: string;
}
