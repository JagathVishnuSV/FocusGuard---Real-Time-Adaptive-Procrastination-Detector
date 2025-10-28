import React, { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Eye,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Monitor,
  Coffee,
  Brain,
  Activity,
  ThumbsUp,
  ThumbsDown,
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { apiService, type ActivityEvent } from '@/lib/api'
import type { PredictionSummary } from '@/lib/types'

const normalizeFeatures = (input?: Record<string, unknown>) => {
  if (!input) {
    return undefined
  }

  const entries = Object.entries(input)
    .map<[string, number] | null>(([key, value]) => {
      const parsed = typeof value === 'number' ? value : Number(value)
      if (Number.isFinite(parsed)) {
        return [key, parsed]
      }
      return null
    })
    .filter((entry): entry is [string, number] => entry !== null)

  return entries.length > 0 ? Object.fromEntries(entries) : undefined
}

interface PredictionMetadata {
  combinedScore?: number
  anomalyScore?: number
  classifierProbability?: number | null
  confidence?: number
  heuristicTriggered?: boolean
  sessionId?: string | null
  features?: Record<string, number>
  rawPrediction?: PredictionSummary | null
  predictionLabel?: 'focused' | 'distracted' | null
  contextLabel?: string | null
  contextConfidence?: number | null
  contextCounts?: Record<string, number> | null
  distractionScore?: number | null
  url?: string | null
}

interface ProcessedEvent extends ActivityEvent, PredictionMetadata {
  id: string
  displayType: 'focus_start' | 'distraction' | 'app_switch' | 'idle' | 'session_end' | 'prediction'
  description: string
  requiresFeedback?: boolean
}

export const ActivityFeed: React.FC = () => {
  const [events, setEvents] = useState<ProcessedEvent[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [feedbackStatus, setFeedbackStatus] = useState<Record<string, 'idle' | 'pending' | 'success' | 'error'>>({})
  const lastPredictionLabel = React.useRef<string | null>(null)

  const processRawEvent = (rawEvent: ActivityEvent, index: number): ProcessedEvent => {
    let displayType: ProcessedEvent['displayType'] = 'app_switch'
    let description = ''

    // Process different event types
    switch (rawEvent.type) {
      case 'prediction': {
        displayType = 'prediction'
        const predictionPayload = rawEvent.prediction ?? safeParseDetail(rawEvent.detail)
        const payloadContext = predictionPayload?.context
        const eventContext = (rawEvent as any).context
        const detailContextLabel = predictionPayload?.dominant_context ?? predictionPayload?.dominantContext
        const detailContextConfidence = predictionPayload?.context_confidence ?? predictionPayload?.contextConfidence
        const detailContextCounts = predictionPayload?.context_counts ?? predictionPayload?.contextCounts
        const combined = predictionPayload?.combined_score ?? predictionPayload?.combinedScore
        const confidence = predictionPayload?.confidence
        const anomaly = predictionPayload?.anomaly_score ?? predictionPayload?.anomalyScore
        const classifier = predictionPayload?.classifier_probability ?? predictionPayload?.classifierProbability
        const heuristic = predictionPayload?.heuristic_triggered ?? predictionPayload?.heuristicTriggered
        const sessionId = predictionPayload?.session_id ?? rawEvent.session_id ?? null
        const features = normalizeFeatures(predictionPayload?.features ?? (rawEvent as any).features)
        const explicitLabel = predictionPayload?.prediction_label ?? predictionPayload?.predictionLabel

        let predictedLabel: 'focused' | 'distracted' | null = null
        if (typeof explicitLabel === 'string') {
          const normalized = explicitLabel.toLowerCase()
          if (normalized === 'focused' || normalized === 'focus') {
            predictedLabel = 'focused'
          } else if (normalized === 'distracted' || normalized === 'distraction') {
            predictedLabel = 'distracted'
          }
        }

        if (!predictedLabel && typeof combined === 'number') {
          predictedLabel = combined >= 0.6 ? 'distracted' : 'focused'
        }

        const combinedDisplay = typeof combined === 'number' ? combined.toFixed(2) : null
        const confidenceDisplay = typeof confidence === 'number' ? `${Math.round(confidence * 100)}%` : null

        if (predictedLabel === 'distracted') {
          description = 'Model flagged distraction'
        } else if (predictedLabel === 'focused') {
          description = 'Model confirmed focus'
        } else {
          description = 'Ensemble prediction generated'
        }
        if (combinedDisplay) {
          description += ` • Combined ${combinedDisplay}`
        }
        if (confidenceDisplay) {
          description += ` • Confidence ${confidenceDisplay}`
        }
        if (heuristic) {
          description += ' • Heuristic triggered'
        }
        const resolvedContextLabel = payloadContext?.label ?? eventContext?.label ?? detailContextLabel ?? null
        const resolvedContextConfidence = payloadContext?.confidence ?? eventContext?.confidence ?? detailContextConfidence ?? null
        if (resolvedContextLabel) {
          const confidenceText = typeof resolvedContextConfidence === 'number' ? ` (${Math.round(resolvedContextConfidence * 100)}%)` : ''
          description += ` • Context: ${resolvedContextLabel}${confidenceText}`
        }
        const resolvedDistractionScore = typeof predictionPayload?.distraction_score === 'number'
          ? predictionPayload.distraction_score
          : typeof (rawEvent as any).distraction_score === 'number'
            ? (rawEvent as any).distraction_score
            : null
        if (typeof resolvedDistractionScore === 'number') {
          description += ` • Score ${resolvedDistractionScore.toFixed(1)}`
        }

        return {
          id: `${rawEvent.timestamp}-${index}`,
          timestamp: rawEvent.timestamp,
          type: rawEvent.type,
          app: rawEvent.app,
          title: rawEvent.title,
          detail: rawEvent.detail,
          displayType,
          description,
          combinedScore: combined,
          anomalyScore: anomaly,
          classifierProbability: classifier,
          confidence,
          heuristicTriggered: heuristic,
          sessionId,
          features,
          rawPrediction: predictionPayload ?? null,
          predictionLabel: predictedLabel,
          requiresFeedback: false,
          contextLabel: resolvedContextLabel,
          contextConfidence: resolvedContextConfidence,
          contextCounts: payloadContext?.counts ?? eventContext?.counts ?? detailContextCounts ?? null,
          distractionScore: resolvedDistractionScore,
          url: (rawEvent as any).url ?? null,
        }
      }
      case 'app_switch':
        displayType = 'app_switch'
        description = `Switched to ${rawEvent.app}`
        break
      case 'click':
        displayType = 'focus_start'
        description = `Clicked in ${rawEvent.app}`
        break
      case 'keystroke':
        displayType = 'focus_start'
        description = `Typing in ${rawEvent.app}`
        break
      case 'idle':
        displayType = 'idle'
        description = 'System idle detected'
        break
      default:
        displayType = 'app_switch'
        description = `${rawEvent.type} in ${rawEvent.app}`
    }

    // Add window title if available
    if (rawEvent.title && rawEvent.title.length > 0) {
      description += ` - ${rawEvent.title.substring(0, 50)}${rawEvent.title.length > 50 ? '...' : ''}`
    }

    return {
      id: `${rawEvent.timestamp}-${index}`,
      timestamp: rawEvent.timestamp,
      type: rawEvent.type,
      app: rawEvent.app,
      title: rawEvent.title,
      detail: rawEvent.detail,
      displayType,
      description,
      requiresFeedback: false,
      contextLabel: rawEvent.context?.label ?? null,
      contextConfidence: rawEvent.context?.confidence ?? null,
      contextCounts: rawEvent.context?.counts ?? null,
      url: rawEvent.url ?? null,
    }
  }

  const safeParseDetail = (detail?: string | null) => {
    if (!detail) {
      return undefined
    }

    try {
      return JSON.parse(detail)
    } catch (err) {
      console.debug('Could not parse prediction detail payload', err)
      return undefined
    }
  }

  const fetchActivityData = async () => {
    try {
      setError(null)
      const rawEvents = await apiService.getRecentActivity()
      const processedEvents = rawEvents.map(processRawEvent)

      let previousLabel = lastPredictionLabel.current
      const augmented = processedEvents.map((event) => {
        if (event.displayType !== 'prediction') {
          return event
        }

        const confidence = typeof event.confidence === 'number' ? event.confidence : null
        const combined = typeof event.combinedScore === 'number' ? event.combinedScore : null
        const heuristic = Boolean(event.heuristicTriggered)
        const currentLabel = event.predictionLabel ?? (combined !== null && combined >= 0.6 ? 'distracted' : 'focused')

        const lowConfidence = confidence !== null && confidence >= 0.35 && confidence <= 0.7
        const labelFlip = previousLabel !== null && previousLabel !== currentLabel
        const shouldAsk = lowConfidence || heuristic || labelFlip

        previousLabel = currentLabel

        return {
          ...event,
          requiresFeedback: shouldAsk,
        }
      })

      lastPredictionLabel.current = previousLabel ?? lastPredictionLabel.current
      setEvents(augmented.reverse())
    } catch (err) {
      console.error('Failed to fetch activity data:', err)
      setError('Failed to load activity data')
      setEvents([])
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchActivityData()
    const interval = setInterval(fetchActivityData, 5000) // Refresh every 5 seconds
    return () => clearInterval(interval)
  }, [])

  const handleFeedback = useCallback(
    async (event: ProcessedEvent, label: 'focused' | 'distracted') => {
      if (feedbackStatus[event.id] === 'pending') {
        return
      }

      setFeedbackStatus((prev) => ({ ...prev, [event.id]: 'pending' }))

  const combinedScore = typeof event.combinedScore === 'number' ? event.combinedScore : null
  const predictedLabel = event.predictionLabel ?? (combinedScore !== null && combinedScore >= 0.6 ? 'distracted' : 'focused')

      try {
        await apiService.submitPersonalFeedback({
          user_label: label,
          predicted_label: predictedLabel,
          timestamp: event.timestamp,
          session_id: event.sessionId ?? null,
          prediction: event.rawPrediction ?? {
            combined_score: combinedScore ?? undefined,
            anomaly_score: typeof event.anomalyScore === 'number' ? event.anomalyScore : undefined,
            classifier_probability:
              typeof event.classifierProbability === 'number' ? event.classifierProbability : undefined,
            confidence: typeof event.confidence === 'number' ? event.confidence : undefined,
            heuristic_triggered: event.heuristicTriggered,
            session_id: event.sessionId ?? null,
            features: event.features,
          },
          features: event.features,
          app_name: event.app,
        })
        setFeedbackStatus((prev) => ({ ...prev, [event.id]: 'success' }))
      } catch (submitError) {
        console.error('Failed to submit personalization feedback', submitError)
        setFeedbackStatus((prev) => ({ ...prev, [event.id]: 'error' }))
      }
    },
    [feedbackStatus]
  )

  const getEventIcon = (event: ProcessedEvent) => {
    switch (event.displayType) {
      case 'focus_start':
        return <CheckCircle className="w-4 h-4 text-emerald-500" />
      case 'distraction':
        return <AlertTriangle className="w-4 h-4 text-red-500" />
      case 'app_switch':
        return <Monitor className="w-4 h-4 text-blue-500" />
      case 'idle':
        return <Coffee className="w-4 h-4 text-gray-500" />
      case 'session_end':
        return <XCircle className="w-4 h-4 text-gray-600" />
      case 'prediction':
        if (event.predictionLabel === 'distracted') {
          return <AlertTriangle className="w-4 h-4 text-red-500" />
        }
        if (event.predictionLabel === 'focused') {
          return <CheckCircle className="w-4 h-4 text-emerald-500" />
        }
        return <Brain className="w-4 h-4 text-purple-400" />
      default:
        return <Eye className="w-4 h-4 text-gray-400" />
    }
  }

  const getEventBadge = (event: ProcessedEvent) => {
    switch (event.displayType) {
      case 'focus_start':
        return <Badge variant="outline" className="border-emerald-500/40 bg-emerald-500/10 text-emerald-200">Interaction</Badge>
      case 'distraction':
        return <Badge variant="outline" className="border-rose-500/40 bg-rose-500/10 text-rose-200">Distraction</Badge>
      case 'app_switch':
        return <Badge variant="outline" className="border-sky-500/40 bg-sky-500/10 text-sky-200">Switch</Badge>
      case 'idle':
        return <Badge variant="outline" className="border-slate-500/40 bg-slate-500/10 text-slate-300">Idle</Badge>
      case 'session_end':
        return <Badge variant="outline" className="border-slate-500/40 bg-slate-500/10 text-slate-300">Session</Badge>
      case 'prediction':
        if (event.predictionLabel === 'distracted') {
          return <Badge variant="outline" className="border-rose-500/40 bg-rose-500/10 text-rose-200">Distraction</Badge>
        }
        if (event.predictionLabel === 'focused') {
          return <Badge variant="outline" className="border-emerald-500/40 bg-emerald-500/10 text-emerald-200">Focus</Badge>
        }
        return <Badge variant="outline" className="border-purple-500/40 bg-purple-500/10 text-purple-200">Ensemble</Badge>
      default:
        return <Badge variant="outline" className="border-slate-500/40 bg-slate-500/10 text-slate-300">Activity</Badge>
    }
  }

  if (isLoading) {
    return (
      <Card className="h-full bg-black/40 border-white/10">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Activity Feed
          </CardTitle>
          <CardDescription>Real-time activity monitoring</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="flex items-center gap-3">
              <div className="w-4 h-4 bg-white/10 rounded-full animate-pulse" />
              <div className="flex-1 space-y-2">
                <div className="h-3 bg-white/10 rounded animate-pulse" />
                <div className="h-2 bg-white/5 rounded animate-pulse w-3/4" />
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="h-full bg-black/40 border-white/10">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Activity Feed
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse ml-auto" />
        </CardTitle>
        <CardDescription>
          {error ? 'Activity monitoring unavailable' : 'Real-time activity monitoring'}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 max-h-96 overflow-y-auto">
        <AnimatePresence mode="popLayout">
          {events.length === 0 ? (
            <div className="text-center py-8 text-slate-300">
              <Brain className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>{error ? 'Unable to retrieve recent events' : 'No recent activity'}</p>
              <p className="text-sm text-slate-400">Start a session to see real-time data</p>
            </div>
          ) : (
            events.map((event) => (
              <motion.div
                key={event.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="flex items-start gap-3 p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
              >
                <div className="mt-1">
                  {getEventIcon(event)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    {getEventBadge(event)}
                    <span className="text-xs text-slate-400">
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-sm text-slate-100 truncate">
                    {event.description}
                  </p>
                  {event.detail && event.displayType !== 'prediction' && (
                    <p className="text-xs text-slate-400 mt-1 truncate">
                      {event.detail}
                    </p>
                  )}
                  {event.contextLabel && (
                    <p className="text-xs text-slate-400 mt-1 truncate">
                      Context: {event.contextLabel}
                      {typeof event.contextConfidence === 'number' ? ` (${Math.round(event.contextConfidence * 100)}%)` : ''}
                    </p>
                  )}
                  {typeof event.distractionScore === 'number' && (
                    <p className="text-xs text-slate-400 mt-1 truncate">
                      Distraction score: {event.distractionScore.toFixed(1)}
                    </p>
                  )}
                  {event.displayType === 'prediction' && event.requiresFeedback && (
                    <div className="grid grid-cols-2 gap-2 mt-3 text-xs text-slate-300">
                      {typeof event.combinedScore === 'number' && (
                        <div>
                          <span className="font-semibold text-slate-100">Combined:</span>{' '}
                          {event.combinedScore.toFixed(2)}
                        </div>
                      )}
                      {typeof event.confidence === 'number' && (
                        <div>
                          <span className="font-semibold text-slate-100">Confidence:</span>{' '}
                          {(event.confidence * 100).toFixed(0)}%
                        </div>
                      )}
                      {typeof event.anomalyScore === 'number' && (
                        <div>
                          <span className="font-semibold text-slate-100">Anomaly:</span>{' '}
                          {event.anomalyScore.toFixed(2)}
                        </div>
                      )}
                      {typeof event.classifierProbability === 'number' && (
                        <div>
                          <span className="font-semibold text-slate-100">Classifier:</span>{' '}
                          {(event.classifierProbability * 100).toFixed(0)}%
                        </div>
                      )}
                      {event.predictionLabel && (
                        <div>
                          <span className="font-semibold text-slate-100">Label:</span>{' '}
                          {event.predictionLabel === 'distracted' ? 'Distraction' : 'Focus'}
                        </div>
                      )}
                      {typeof event.distractionScore === 'number' && (
                        <div>
                          <span className="font-semibold text-slate-100">Score:</span>{' '}
                          {event.distractionScore.toFixed(1)}
                        </div>
                      )}
                      {event.heuristicTriggered !== undefined && (
                        <div className="col-span-2">
                          <span className="font-semibold text-slate-100">Heuristic:</span>{' '}
                          {event.heuristicTriggered ? 'Triggered safeguards' : 'No heuristic trigger'}
                        </div>
                      )}
                      <div className="col-span-2 flex items-center gap-2 pt-2 border-t border-white/10 mt-2">
                        <button
                          type="button"
                          onClick={() => handleFeedback(event, 'focused')}
                          disabled={feedbackStatus[event.id] === 'pending'}
                          className="inline-flex items-center gap-1 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-2 py-1 text-emerald-100 text-xs hover:bg-emerald-500/20 disabled:opacity-50"
                          aria-label="Mark as focused"
                        >
                          <ThumbsUp className="w-3.5 h-3.5" />
                          Focused
                        </button>
                        <button
                          type="button"
                          onClick={() => handleFeedback(event, 'distracted')}
                          disabled={feedbackStatus[event.id] === 'pending'}
                          className="inline-flex items-center gap-1 rounded-md border border-rose-500/30 bg-rose-500/10 px-2 py-1 text-rose-100 text-xs hover:bg-rose-500/20 disabled:opacity-50"
                          aria-label="Mark as distracted"
                        >
                          <ThumbsDown className="w-3.5 h-3.5" />
                          Distracted
                        </button>
                        {feedbackStatus[event.id] === 'success' && (
                          <span className="text-[11px] text-slate-300">Thanks for the feedback!</span>
                        )}
                        {feedbackStatus[event.id] === 'error' && (
                          <span className="text-[11px] text-rose-300">Submission failed. Try again.</span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </CardContent>
    </Card>
  )
}