import axios from 'axios'
import type {
  TodayStats,
  WeeklyStats,
  HourlyStats,
  Insight,
  SessionStatus,
  HealthStatus,
  ActivityEvent,
  PredictionSummary,
  PersonalFeedbackRequest,
  DistractionStat,
  CognitiveTwinSnapshot,
} from './types'

const API_BASE_URL = 'http://127.0.0.1:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
})

// API functions
export const apiService = {
  // Health check
  async getHealth(): Promise<HealthStatus> {
    const response = await api.get('/health')
    return response.data
  },

  // Statistics
  async getTodayStats(): Promise<TodayStats> {
    const response = await api.get('/api/stats/today')
    const data = response.data
    // Provide a safe default so dashboards render gracefully when backend returns empty
    if (!data) {
      return {
        focus_score: 0,
        focused_time: 0,
        distracted_time: 0,
        anomalies: 0,
        sessions: 0,
      }
    }
    return data
  },

  async getWeeklyStats(): Promise<WeeklyStats> {
    const response = await api.get('/api/stats/weekly')
    const data = response.data
    if (!data || !Array.isArray(data.days) || !Array.isArray(data.scores)) {
      return { days: [], scores: [] }
    }
    return data
  },

  async getHourlyStats(): Promise<HourlyStats> {
    const response = await api.get('/api/stats/hourly')
    const data = response.data
    if (!data || !Array.isArray(data.hours) || !Array.isArray(data.pattern)) {
      return { hours: [], pattern: [] }
    }
    return data
  },

  async getWhatIf(hour: number): Promise<{ hour: string; predicted_focus: number | null; hours: string[]; pattern: number[] }> {
    const response = await api.get('/api/predict/whatif', { params: { hour } })
    return response.data
  },

  // Insights and analysis
  async getInsights(): Promise<Insight[]> {
    const response = await api.get('/api/insights')
    const data = response.data
    return Array.isArray(data) ? data : []
  },

  async getTopDistractions(): Promise<Record<string, DistractionStat>> {
    const response = await api.get('/api/distractions/top')
    const data = response.data
    return data && typeof data === 'object' ? data : {}
  },

  async getFeatureImportance(): Promise<Record<string, number>> {
    const response = await api.get('/api/features/importance')
    const data = response.data
    return data && typeof data === 'object' ? data : {}
  },

  // Session management
  async getSessionStatus(): Promise<SessionStatus> {
    const response = await api.get('/api/session/status')
    return response.data
  },

  async startSession(): Promise<{ status: string; session: SessionStatus }> {
    const response = await api.post('/api/session/start')
    return response.data
  },

  async stopSession(): Promise<{ status: string; session: SessionStatus }> {
    const response = await api.post('/api/session/stop')
    return response.data
  },

  async updateSession(data: Partial<SessionStatus['stats']>): Promise<{ status: string; session: SessionStatus }> {
    const response = await api.post('/api/session/update', data)
    return response.data
  },

  // Data export
  async exportData(): Promise<any> {
    const response = await api.get('/api/export')
    return response.data
  },

  // Activity feed
  async getRecentActivity(): Promise<ActivityEvent[]> {
    const response = await api.get('/api/activity/recent')
    return response.data
  },

  async submitPersonalFeedback(payload: PersonalFeedbackRequest): Promise<{ status: string }> {
    const response = await api.post('/api/personalization/feedback', payload)
    return response.data
  },

}

// Error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error)
    
    if (error.code === 'ECONNREFUSED') {
      throw new Error('FocusGuard backend is not running. Please start the server.')
    }
    
    if (error.response?.status === 404) {
      throw new Error('API endpoint not found')
    }
    
    if (error.response?.status >= 500) {
      throw new Error('Server error. Please try again.')
    }
    
    throw error
  }
)

export default api

export type {
  TodayStats,
  WeeklyStats,
  HourlyStats,
  Insight,
  CognitiveTwinSnapshot,
  SessionStatus,
  HealthStatus,
  ActivityEvent,
  PredictionSummary,
  PersonalFeedbackRequest,
  DistractionStat,
} from './types'