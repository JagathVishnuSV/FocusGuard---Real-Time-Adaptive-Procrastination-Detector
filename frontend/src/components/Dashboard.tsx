import React, { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import {
  Target,
  Clock,
  AlertTriangle,
  Activity as ActivityIcon,
  Play,
  Square,
  Brain
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { StatusDisplay } from '@/components/ui/StatusDisplay'
import { MetricsCard } from './MetricsCard'
import { FocusChart } from './FocusChart'
import { ActivityFeed } from './ActivityFeed'
import { InsightsPanel } from './InsightsPanel'
import { cn, formatDuration } from '@/lib/utils'
import { useApi, useSessionStatus } from '@/hooks/useApi'
import type { TodayStats, SessionStatus } from '@/lib/types'

const Dashboard: React.FC = () => {
  const [actionError, setActionError] = useState<string | null>(null)
  const [pendingAction, setPendingAction] = useState<'start' | 'stop' | null>(null)

  const {
    sessionStatus,
    isLoading: isSessionLoading,
    error: sessionError,
    startSession,
    stopSession,
    isMutating,
  } = useSessionStatus()

  const {
    data: todayStats,
    error: todayError,
    isLoading: isTodayLoading,
  } = useApi<TodayStats>('/api/stats/today')

  const isSessionActive = Boolean(sessionStatus?.active)

  const sessionStats = useMemo<SessionStatus['stats']>(() => (
    sessionStatus?.stats ?? {
      total_events: 0,
      anomalies: 0,
      focused_time: 0,
      distracted_time: 0,
      elapsed_time: 0,
    }
  ), [sessionStatus])

  const resolvedTodayStats: TodayStats = useMemo(() => {
    const baseStats: TodayStats = todayStats ?? {
      focus_score: 0,
      focused_time: 0,
      distracted_time: 0,
      anomalies: 0,
      sessions: 0,
    }

    const hasRecordedStats = Boolean(
      todayStats && (
        (todayStats.sessions ?? 0) > 0 ||
        todayStats.focused_time > 0 ||
        todayStats.distracted_time > 0 ||
        todayStats.anomalies > 0 ||
        todayStats.focus_score > 0
      )
    )

    if (hasRecordedStats) {
      return {
        ...baseStats,
        focus_score: Number(baseStats.focus_score.toFixed(1)),
      }
    }

    const focusedSeconds = sessionStats.focused_time ?? 0
    const distractedSeconds = sessionStats.distracted_time ?? 0
    const totalSeconds = focusedSeconds + distractedSeconds
    const hasLiveStats = totalSeconds > 0 || sessionStats.total_events > 0 || isSessionActive

    if (hasLiveStats) {
      const focusScore = totalSeconds > 0 ? (focusedSeconds / totalSeconds) * 100 : 0

      // Derive a live snapshot so the dashboard reflects the current session instead of zeros.
      return {
        focus_score: Number(focusScore.toFixed(1)),
        focused_time: focusedSeconds / 60,
        distracted_time: distractedSeconds / 60,
        anomalies: sessionStats.anomalies ?? 0,
        sessions: Math.max(baseStats.sessions ?? 0, isSessionActive ? 1 : 0),
      }
    }

    return baseStats
  }, [todayStats, sessionStats, isSessionActive])

  const isStarting = pendingAction === 'start' && isMutating
  const isStopping = pendingAction === 'stop' && isMutating

  const handleStartSession = async () => {
    setActionError(null)
    setPendingAction('start')
    try {
      await startSession()
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error'
      setActionError(`Failed to start session: ${message}`)
    } finally {
      setPendingAction(null)
    }
  }

  const handleStopSession = async () => {
    setActionError(null)
    setPendingAction('stop')
    try {
      await stopSession()
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error'
      setActionError(`Failed to stop session: ${message}`)
    } finally {
      setPendingAction(null)
    }
  }

  const fatalError = actionError
    ? new Error(actionError)
    : (!sessionStatus && sessionError) || (!todayStats && todayError)
      ? (sessionError ?? todayError ?? null)
      : null

  const focusScoreProgress = Math.min(Math.max(resolvedTodayStats.focus_score, 0), 100)
  const predictionMeta = sessionStatus?.prediction ?? null
  const combinedScore = predictionMeta?.combined_score ?? sessionStats?.combined_score ?? null
  const anomalyScore = predictionMeta?.anomaly_score ?? sessionStats?.anomaly_score ?? null
  const classifierProb = predictionMeta?.classifier_probability ?? sessionStats?.classifier_probability ?? null
  const confidence = predictionMeta?.confidence ?? sessionStats?.confidence ?? null
  const heuristicTriggered = predictionMeta?.heuristic_triggered ?? sessionStats?.heuristic_triggered ?? false

  if ((isSessionLoading && !sessionStatus) || (isTodayLoading && !todayStats)) {
    return (
      <StatusDisplay
        isLoading
        data={null}
        emptyMessage="Preparing FocusGuard dashboard..."
        className="min-h-screen"
      />
    )
  }

  if (fatalError) {
    return (
      <StatusDisplay
        error={fatalError}
        data={null}
        errorMessage="Unable to load FocusGuard dashboard"
        className="min-h-screen"
      />
    )
  }

  return (
    <div className="min-h-screen">
      <header className="border-b border-white/5 bg-black/60 backdrop-blur-md sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <motion.div
              className="flex items-center space-x-3"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="p-2 bg-primary/10 rounded-lg">
                <Target className="w-6 h-6 text-primary" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-purple-600 bg-clip-text text-transparent">
                  FocusGuard
                </h1>
                <p className="text-sm text-muted-foreground">Real-time Procrastination Detection</p>
              </div>
            </motion.div>

            <motion.div
              className="flex items-center space-x-3"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="flex items-center space-x-2">
                <div
                  className={cn(
                    'w-2 h-2 rounded-full animate-pulse',
                    isSessionActive ? 'bg-green-500' : 'bg-gray-400'
                  )}
                />
                <div
                  className={cn(
                    'px-3 py-1 rounded-full text-xs font-medium backdrop-blur',
                    isSessionActive
                      ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/20'
                      : 'bg-slate-500/10 text-slate-300 border border-slate-500/20'
                  )}
                >
                  {isSessionActive ? '🟢 Monitoring Active' : '⏸️ Not Monitoring'}
                </div>
                {isSessionActive && sessionStatus?.start_time && (
                  <div className="text-xs text-muted-foreground">
                    {new Date(sessionStatus.start_time).toLocaleTimeString()}
                  </div>
                )}
                {combinedScore !== null && (
                  <div className="px-3 py-1 rounded-full text-xs font-medium bg-purple-500/10 border border-purple-500/20 text-purple-200">
                    Latest score {(combinedScore * 100).toFixed(1)}%
                  </div>
                )}
              </div>

              <Button
                onClick={isSessionActive ? handleStopSession : handleStartSession}
                variant={isSessionActive ? 'destructive' : 'default'}
                size="sm"
                className="min-w-[120px]"
                disabled={isStarting || isStopping}
              >
                {isStarting ? (
                  <>
                    <div className="w-4 h-4 mr-2 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    Starting...
                  </>
                ) : isStopping ? (
                  <>
                    <div className="w-4 h-4 mr-2 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    Stopping...
                  </>
                ) : isSessionActive ? (
                  <>
                    <Square className="w-4 h-4 mr-2" />
                    Stop Session
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    Start Session
                  </>
                )}
              </Button>
            </motion.div>
          </div>
        </div>
      </header>

      {(sessionError || todayError) && (
        <div className="border-b border-amber-500/20 bg-amber-500/10">
          <div className="container mx-auto px-6 py-3">
            <div className="flex items-center space-x-2 text-amber-200 text-sm">
              <AlertTriangle className="w-4 h-4" />
              <span>
                {sessionError instanceof Error ? sessionError.message : todayError instanceof Error ? todayError.message : 'Some data may be unavailable.'}
              </span>
            </div>
          </div>
        </div>
      )}

      {actionError && (
        <div className="border-b border-red-500/30 bg-red-500/10">
          <div className="container mx-auto px-6 py-3">
            <div className="flex items-center space-x-2 text-red-200 text-sm">
              <AlertTriangle className="w-4 h-4" />
              <span>{actionError}</span>
              <Button variant="ghost" size="sm" onClick={() => setActionError(null)} className="ml-auto">
                Dismiss
              </Button>
            </div>
          </div>
        </div>
      )}

  <main className="container mx-auto px-6 py-8 space-y-8 text-foreground">
        <StatusDisplay
          isLoading={isTodayLoading}
          error={todayError}
          data={todayStats ?? resolvedTodayStats}
          emptyMessage="No activity recorded today. Start a session to generate insights."
        >
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <MetricsCard
              title="Focus Score"
              value={`${resolvedTodayStats.focus_score.toFixed(1)}%`}
              icon={Target}
              color="primary"
              description="Today's productivity level"
              progress={focusScoreProgress}
            />

            <MetricsCard
              title="Focused Time"
              value={formatDuration(resolvedTodayStats.focused_time * 60)}
              icon={Clock}
              color="success"
              description="Productive work time"
            />

            <MetricsCard
              title="Distractions"
              value={resolvedTodayStats.anomalies.toString()}
              icon={AlertTriangle}
              color="warning"
              description="Interruptions detected"
            />

            <MetricsCard
              title="Sessions"
              value={resolvedTodayStats.sessions.toString()}
              icon={ActivityIcon}
              color="info"
              description="Work sessions today"
            />
          </motion.div>
        </StatusDisplay>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <motion.div
            className="lg:col-span-2"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <FocusChart />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <ActivityFeed />
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.15 }}
        >
          <InsightsPanel />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm font-semibold text-muted-foreground">
                <Brain className="w-4 h-4" />
                Live Session Summary
              </CardTitle>
              <CardDescription>
                Realtime counters from the active monitoring session.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-xs uppercase text-muted-foreground">Elapsed Time</p>
                <p className="text-lg font-semibold">{formatDuration(sessionStats.elapsed_time ?? 0)}</p>
              </div>
              <div>
                <p className="text-xs uppercase text-muted-foreground">Events</p>
                <p className="text-lg font-semibold">{sessionStats.total_events}</p>
              </div>
              <div>
                <p className="text-xs uppercase text-muted-foreground">Focused</p>
                <p className="text-lg font-semibold">{formatDuration(sessionStats.focused_time ?? 0)}</p>
              </div>
              <div>
                <p className="text-xs uppercase text-muted-foreground">Distracted</p>
                <p className="text-lg font-semibold">{formatDuration(sessionStats.distracted_time ?? 0)}</p>
              </div>
            </CardContent>
            {(combinedScore !== null || anomalyScore !== null || classifierProb !== null || confidence !== null || heuristicTriggered) && (
              <CardContent className="border-t border-white/5 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {combinedScore !== null && (
                    <div>
                      <p className="text-xs uppercase text-muted-foreground">Combined Score</p>
                      <p className="text-lg font-semibold">{(combinedScore * 100).toFixed(1)}%</p>
                    </div>
                  )}
                  {anomalyScore !== null && (
                    <div>
                      <p className="text-xs uppercase text-muted-foreground">Anomaly Score</p>
                      <p className="text-lg font-semibold">{anomalyScore.toFixed(3)}</p>
                    </div>
                  )}
                  {classifierProb !== null && (
                    <div>
                      <p className="text-xs uppercase text-muted-foreground">Classifier Probability</p>
                      <p className="text-lg font-semibold">{(classifierProb * 100).toFixed(1)}%</p>
                    </div>
                  )}
                  {confidence !== null && (
                    <div>
                      <p className="text-xs uppercase text-muted-foreground">Decision Confidence</p>
                      <p className="text-lg font-semibold">{(confidence * 100).toFixed(1)}%</p>
                    </div>
                  )}
                  {heuristicTriggered && (
                    <div className="md:col-span-3">
                      <p className="text-xs uppercase text-muted-foreground">Heuristic Override</p>
                      <p className="text-sm text-amber-300">Rule-based override boosted the latest prediction.</p>
                    </div>
                  )}
                </div>
              </CardContent>
            )}
          </Card>
        </motion.div>
      </main>
    </div>
  )
}

export default Dashboard