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
    return response.data
  },

  async getWeeklyStats(): Promise<WeeklyStats> {
    const response = await api.get('/api/stats/weekly')
    return response.data
  },

  async getHourlyStats(): Promise<HourlyStats> {
    const response = await api.get('/api/stats/hourly')
    return response.data
  },

  // Insights and analysis
  async getInsights(): Promise<Insight[]> {
    const response = await api.get('/api/insights')
    return response.data
  },

  async getTopDistractions(): Promise<Record<string, number>> {
    const response = await api.get('/api/distractions/top')
    return response.data
  },

  async getFeatureImportance(): Promise<Record<string, number>> {
    const response = await api.get('/api/features/importance')
    return response.data
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
  SessionStatus,
  HealthStatus,
  ActivityEvent,
  PredictionSummary,
  PersonalFeedbackRequest,
} from './types'