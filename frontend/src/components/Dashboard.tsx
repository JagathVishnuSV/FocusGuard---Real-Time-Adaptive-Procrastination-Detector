import React, { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import {
  Target,
  Clock,
  AlertTriangle,
  Play,
  Square,
  Brain,
  Shield,
  Zap,
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { StatusDisplay } from '@/components/ui/StatusDisplay'
import { MetricsCard } from './MetricsCard'
import { FocusChart } from './FocusChart'
import { ActivityFeed } from './ActivityFeed'
import { CognitiveTwinPanel } from './CognitiveTwinPanel'
import { cn, formatDuration } from '@/lib/utils'
import { useApi, useSessionStatus } from '@/hooks/useApi'
import type { TodayStats, SessionStatus } from '@/lib/types'

const FOCUS_GOAL_MINUTES = 120

type HeroStat = {
  label: string
  value: string
  hint: string
  isContext?: boolean
}

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
  const cognitiveTwin = predictionMeta?.cognitive_twin ?? null
  const focusDelta = todayStats?.change ?? 0
  const deepWorkMinutes = resolvedTodayStats.focused_time
  const deepWorkProgress = FOCUS_GOAL_MINUTES > 0
    ? Math.min(100, Math.round((deepWorkMinutes / FOCUS_GOAL_MINUTES) * 100))
    : 0
  const eventsPerMinute = sessionStats.elapsed_time && sessionStats.elapsed_time > 0
    ? sessionStats.total_events / Math.max(1, sessionStats.elapsed_time / 60)
    : 0
  const distractionRatio = (() => {
    const focusSeconds = sessionStats.focused_time ?? 0
    const distractSeconds = sessionStats.distracted_time ?? 0
    const total = focusSeconds + distractSeconds
    if (total === 0) {
      return 0
    }
    return distractSeconds / total
  })()
  const distractionRate = sessionStats.elapsed_time && sessionStats.elapsed_time > 0
    ? (sessionStats.anomalies ?? 0) / Math.max(1, sessionStats.elapsed_time / 3600)
    : (sessionStats.anomalies ?? 0)
  const riskScore = sessionStatus?.prediction?.distraction_score ?? sessionStatus?.stats?.distraction_score ?? distractionRatio
  const riskLevel = riskScore >= 0.6 ? 'High' : riskScore >= 0.35 ? 'Medium' : 'Low'
  const riskNarrative = riskLevel === 'High'
    ? 'Spike in distraction signals — tighten rituals now.'
    : riskLevel === 'Medium'
      ? 'Maintain cadence with short resets between blocks.'
      : 'Prime window for deep work. Lock in a long block.'
  const monitoringSince = sessionStatus?.start_time ? new Date(sessionStatus.start_time) : null
  const heroStats = useMemo<HeroStat[]>(() => {
    const rawContext = predictionMeta?.dominant_context ?? sessionStatus?.stats?.dominant_context ?? ''
    const contextLabel = typeof rawContext === 'string' && rawContext.trim().length > 0 ? rawContext : 'Unknown'
    return [
      {
        label: 'Dominant Context',
        value: contextLabel,
        hint: 'Most confident classifier state',
        isContext: true,
      },
      {
        label: 'Elapsed Session',
        value: formatDuration(sessionStats.elapsed_time ?? 0),
        hint: monitoringSince
          ? `Started ${monitoringSince.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
          : 'Start a session to begin tracking',
      },
      {
        label: 'Signals Processed',
        value: sessionStats.total_events.toString(),
        hint: `${eventsPerMinute.toFixed(1)} events / min`,
      },
    ]
  }, [eventsPerMinute, monitoringSince, predictionMeta?.dominant_context, sessionStats.elapsed_time, sessionStats.total_events, sessionStatus?.stats?.dominant_context])
  const metricCards = useMemo(() => ([
    {
      title: 'Focus Score',
      value: `${resolvedTodayStats.focus_score.toFixed(1)}%`,
      icon: Target,
      color: 'primary' as const,
      description: 'vs rolling baseline',
      progress: focusScoreProgress,
      trend: {
        value: Math.abs(Number(focusDelta.toFixed(1))),
        isPositive: focusDelta >= 0,
      },
    },
    {
      title: 'Deep Work Minutes',
      value: `${Math.round(deepWorkMinutes)} / ${FOCUS_GOAL_MINUTES}`,
      icon: Clock,
      color: 'success' as const,
      description: 'Daily target 2h',
      progress: deepWorkProgress,
    },
    {
      title: 'Distractions / hr',
      value: distractionRate.toFixed(1),
      icon: AlertTriangle,
      color: 'warning' as const,
      description: 'Based on anomalies this session',
    },
    {
      title: 'Event Velocity',
      value: `${eventsPerMinute.toFixed(1)}/min`,
      icon: Zap,
      color: 'info' as const,
      description: 'Signals processed each minute',
    },
  ]), [resolvedTodayStats.focus_score, focusScoreProgress, focusDelta, deepWorkMinutes, deepWorkProgress, distractionRate, eventsPerMinute])
  const alerts = Array.isArray(sessionStatus?.alerts) ? sessionStatus.alerts : []
  const latestAlert = alerts.length ? alerts[0] : null
  const focusConfidence = typeof predictionMeta?.confidence === 'number'
    ? Math.round(predictionMeta.confidence * 100)
    : null
  const lastUpdatedAt = predictionMeta?.timestamp ? new Date(predictionMeta.timestamp) : null
  const focusUptime = sessionStats.elapsed_time && sessionStats.elapsed_time > 0
    ? Math.round(((sessionStats.focused_time ?? 0) / Math.max(1, sessionStats.elapsed_time)) * 100)
    : 0

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
      <header className="sticky top-0 z-50 border-b border-white/5 bg-black/70 backdrop-blur">
        <div className="container mx-auto px-6 py-4">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            <motion.div
              className="flex flex-col gap-3 sm:flex-row sm:items-center sm:gap-4"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4 }}
            >
              <div className="rounded-lg bg-primary/10 p-2">
                <Target className="h-6 w-6 text-primary" />
              </div>
              <div className="space-y-1">
                <h1 className="text-2xl font-bold text-white">FocusGuard</h1>
                <p className="text-sm text-muted-foreground">Adaptive procrastination control</p>
              </div>
            </motion.div>
            <motion.div
              className="flex flex-col-reverse gap-3 sm:flex-row sm:items-center sm:justify-end sm:gap-4"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4 }}
            >
              <div className="flex items-center gap-3 text-xs font-medium">
                <span className={cn('inline-flex h-2.5 w-2.5 rounded-full', isSessionActive ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500')} />
                <span className={cn('rounded-full border px-3 py-1 backdrop-blur', isSessionActive ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200' : 'border-slate-600/40 bg-slate-600/10 text-slate-200')}>
                  {isSessionActive ? 'Monitoring active' : 'Monitoring paused'}
                </span>
                <span className={cn('flex items-center gap-1 rounded-full border px-3 py-1 uppercase tracking-wide', riskLevel === 'High' ? 'border-red-500/40 text-red-200' : riskLevel === 'Medium' ? 'border-amber-500/40 text-amber-200' : 'border-emerald-500/40 text-emerald-200')}>
                  <Shield className="h-3.5 w-3.5" />
                  {riskLevel}
                </span>
              </div>
              <Button
                onClick={isSessionActive ? handleStopSession : handleStartSession}
                variant={isSessionActive ? 'destructive' : 'default'}
                size="sm"
                className="min-w-[130px]"
                disabled={isStarting || isStopping}
              >
                {isStarting ? (
                  <>
                    <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    Starting...
                  </>
                ) : isStopping ? (
                  <>
                    <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    Stopping...
                  </>
                ) : isSessionActive ? (
                  <>
                    <Square className="mr-2 h-4 w-4" />
                    Stop Session
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Start Session
                  </>
                )}
              </Button>
            </motion.div>
          </div>
        </div>
      </header>

      {(sessionError || todayError) && (
        <div className="border-b border-amber-500/30 bg-amber-500/10">
          <div className="container mx-auto px-6 py-3 text-sm text-amber-100">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              <span>
                {sessionError instanceof Error ? sessionError.message : todayError instanceof Error ? todayError.message : 'Some data may be delayed.'}
              </span>
            </div>
          </div>
        </div>
      )}

      {actionError && (
        <div className="border-b border-red-500/30 bg-red-500/10">
          <div className="container mx-auto px-6 py-3 text-sm text-red-100">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              <span>{actionError}</span>
              <Button variant="ghost" size="sm" onClick={() => setActionError(null)} className="ml-auto text-red-100">
                Dismiss
              </Button>
            </div>
          </div>
        </div>
      )}

      <main className="container mx-auto px-6 py-8 text-foreground">
        <div className="space-y-8">
          <StatusDisplay
            isLoading={isTodayLoading}
            error={todayError}
            data={todayStats ?? resolvedTodayStats}
            emptyMessage="No activity recorded today. Start a session to generate insights."
          >
            <div className="space-y-6">
              <section className="rounded-2xl border border-white/10 bg-gradient-to-br from-indigo-900/60 via-slate-950 to-black p-6 shadow-xl">
                <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
                  <div className="flex-1 space-y-4">
                    <div className="flex flex-wrap items-center gap-3 text-sm text-slate-200/80">
                      <span className={cn('inline-flex h-2.5 w-2.5 rounded-full', isSessionActive ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500')} />
                      <span>{isSessionActive ? 'Live telemetry streaming' : 'Session idle'}</span>
                      {monitoringSince && isSessionActive && (
                        <span className="text-xs text-slate-300/70">since {monitoringSince.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                      )}
                      <span className={riskLevel === 'High' ? 'ml-auto text-xs font-semibold text-red-200' : riskLevel === 'Medium' ? 'ml-auto text-xs font-semibold text-amber-200' : 'ml-auto text-xs font-semibold text-emerald-200'}>
                        {riskNarrative}
                      </span>
                    </div>
                    <dl className="grid gap-4 sm:grid-cols-3">
                      {heroStats.map((stat) => (
                        <div key={stat.label} className="rounded-xl border border-white/10 bg-black/40 p-4">
                          <dt className="text-xs uppercase tracking-wide text-slate-400">{stat.label}</dt>
                          <dd className="mt-2 text-white">
                            <p
                              className={cn('text-2xl font-semibold leading-snug', stat.isContext && 'text-xl font-medium capitalize text-slate-100')}
                              title={stat.value}
                            >
                              {stat.value}
                            </p>
                            <p className="mt-1 text-xs text-slate-300/80">{stat.hint}</p>
                          </dd>
                        </div>
                      ))}
                    </dl>
                  </div>
                  <div className="w-full max-w-sm rounded-2xl border border-white/10 bg-black/50 p-5">
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-400">Focus score</p>
                      <p className="mt-2 text-4xl font-semibold text-white">{resolvedTodayStats.focus_score.toFixed(1)}%</p>
                      <p className="mt-1 text-sm text-slate-300">
                        {focusDelta >= 0 ? '+' : ''}{focusDelta.toFixed(1)} pts vs yesterday
                      </p>
                    </div>
                    <div className="mt-5">
                      <p className="text-xs uppercase tracking-wide text-slate-400">Deep work target</p>
                      <p className="mt-1 text-sm text-white/80">{Math.round(deepWorkMinutes)} / {FOCUS_GOAL_MINUTES} min</p>
                      <progress
                        className="mt-2 h-2 w-full overflow-hidden rounded-full bg-white/10 [color-scheme:dark] [&::-webkit-progress-bar]:bg-transparent [&::-webkit-progress-value]:bg-emerald-400 [&::-moz-progress-bar]:bg-emerald-400"
                        value={deepWorkProgress}
                        max={100}
                        aria-label="Deep work progress"
                      />
                    </div>
                    <dl className="mt-5 grid grid-cols-2 gap-4 text-sm text-white/80">
                      <div>
                        <dt className="text-xs uppercase text-white/50">Focused</dt>
                        <dd className="mt-1 text-lg font-semibold">{formatDuration(sessionStats.focused_time ?? 0)}</dd>
                      </div>
                      <div>
                        <dt className="text-xs uppercase text-white/50">Distracted</dt>
                        <dd className="mt-1 text-lg font-semibold">{formatDuration(sessionStats.distracted_time ?? 0)}</dd>
                      </div>
                      <div>
                        <dt className="text-xs uppercase text-white/50">Events / min</dt>
                        <dd className="mt-1 text-lg font-semibold">{eventsPerMinute.toFixed(1)}</dd>
                      </div>
                      <div>
                        <dt className="text-xs uppercase text-white/50">Distractions / hr</dt>
                        <dd className="mt-1 text-lg font-semibold">{distractionRate.toFixed(1)}</dd>
                      </div>
                    </dl>
                    <div className="mt-4 rounded-lg border border-white/10 bg-white/5 p-3 text-xs text-slate-200">
                      {latestAlert?.message ?? 'No escalations. FocusGuard will notify you if patterns drift.'}
                    </div>
                  </div>
                </div>
              </section>

              <motion.div
                className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
              >
                {metricCards.map((metric) => (
                  <MetricsCard key={metric.title} {...metric} />
                ))}
              </motion.div>
            </div>
          </StatusDisplay>

          <section className="space-y-6">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Focus signal</p>
                <h3 className="text-lg font-semibold text-white">Quality & activity at a glance</h3>
              </div>
              <div className="text-xs text-muted-foreground">
                Last update {lastUpdatedAt ? lastUpdatedAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—'}
              </div>
            </div>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
              <motion.div
                className="lg:col-span-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.05 }}
              >
                <FocusChart />
              </motion.div>
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.05 }}
              >
                <ActivityFeed />
              </motion.div>
            </div>
          </section>

          {cognitiveTwin && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.07 }}
            >
              <CognitiveTwinPanel
                data={cognitiveTwin}
                isActive={isSessionActive}
                lastUpdated={predictionMeta?.timestamp ?? null}
              />
            </motion.div>
          )}

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.11 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm font-semibold text-muted-foreground">
                  <Brain className="h-4 w-4" />
                  Live Session Summary
                </CardTitle>
                <CardDescription>
                  Real-time counters and model telemetry from the active window.
                </CardDescription>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-4 md:grid-cols-4">
                {[
                  { label: 'Elapsed Time', value: formatDuration(sessionStats.elapsed_time ?? 0) },
                  { label: 'Events', value: sessionStats.total_events.toString() },
                  { label: 'Events / min', value: eventsPerMinute.toFixed(1) },
                  { label: 'Focused', value: formatDuration(sessionStats.focused_time ?? 0) },
                  { label: 'Distracted', value: formatDuration(sessionStats.distracted_time ?? 0) },
                  { label: 'Distraction Ratio', value: `${Math.round(distractionRatio * 100)}%` },
                  { label: 'Focus Uptime', value: `${focusUptime}%` },
                  focusConfidence != null ? { label: 'Model Confidence', value: `${focusConfidence}%` } : null,
                ].filter(Boolean).map((stat) => (
                  <div key={(stat as { label: string }).label} className="rounded-xl border border-muted/40 p-4">
                    <p className="text-xs uppercase text-muted-foreground">{(stat as { label: string }).label}</p>
                    <p className="mt-1 text-lg font-semibold">{(stat as { value: string }).value}</p>
                  </div>
                ))}
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </main>
    </div>
  )
}

export default Dashboard