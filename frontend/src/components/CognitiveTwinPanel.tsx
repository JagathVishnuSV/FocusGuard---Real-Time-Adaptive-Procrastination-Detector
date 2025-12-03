import React, { useMemo } from 'react'
import { Brain, Sparkles, History, Activity, Radar } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import type { CognitiveTwinSnapshot } from '@/lib/types'
import { cn } from '@/lib/utils'

interface CognitiveTwinPanelProps {
  data?: CognitiveTwinSnapshot | null
  isActive: boolean
  lastUpdated?: number | string | null
}

const resolveTimestamp = (value?: number | string | null) => {
  if (value == null) {
    return null
  }

  if (typeof value === 'number' && Number.isFinite(value)) {
    return new Date(value * 1000)
  }

  if (typeof value === 'string') {
    const numeric = Number(value)
    if (!Number.isNaN(numeric)) {
      return new Date(numeric * 1000)
    }

    const parsed = new Date(value)
    return Number.isNaN(parsed.getTime()) ? null : parsed
  }

  return null
}

const MAX_TARGET_LENGTH = 48

const prettyTarget = (token?: string | null) => {
  if (!token) {
    return 'Unknown'
  }
  const trimmed = token.trim()
  if (!trimmed || trimmed === 'unknown') {
    return 'Unknown'
  }
  const normalized = trimmed.length > MAX_TARGET_LENGTH ? `${trimmed.slice(0, MAX_TARGET_LENGTH - 1).trim()}…` : trimmed
  if (/^https?:\/\//i.test(trimmed)) {
    try {
      const url = new URL(trimmed)
      return url.host || trimmed.replace(/^https?:\/\//i, '')
    } catch {
      return normalized.replace(/^https?:\/\//i, '')
    }
  }
  return normalized
}

const formatPercent = (value?: number | null) => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return null
  }
  const bounded = Math.min(Math.max(value * 100, 0), 100)
  return `${bounded.toFixed(1)}%`
}

export const CognitiveTwinPanel: React.FC<CognitiveTwinPanelProps> = ({ data, isActive, lastUpdated }) => {
  const probability = formatPercent(data?.prob_distracted)
  const updatedAt = useMemo(() => resolveTimestamp(lastUpdated), [lastUpdated])
  const statusLabel = isActive ? 'Live session' : 'Waiting for session'
  const predictedNext = prettyTarget(data?.predicted_next)
  const lastObserved = prettyTarget(data?.last_app)
  const confidenceLevel = data?.prob_distracted ?? null
  const hasData = Boolean(data)
  const gaugeClass = useMemo(() => {
    if (confidenceLevel == null) {
      return 'from-slate-500/20 via-slate-600/10 to-slate-700/20'
    }
    if (confidenceLevel >= 0.6) {
      return 'from-rose-500/30 via-purple-500/20 to-indigo-500/10'
    }
    if (confidenceLevel >= 0.35) {
      return 'from-amber-500/25 via-orange-500/15 to-purple-500/10'
    }
    return 'from-emerald-500/25 via-teal-500/10 to-cyan-500/10'
  }, [confidenceLevel])

  const metricValue = (value?: number | null) => {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value
    }
    return '--'
  }

  return (
    <Card className="border-white/10 bg-black/40">
      <CardHeader className="space-y-1">
        <CardTitle className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          <Brain className="h-4 w-4 text-purple-300" />
          Cognitive Twin
        </CardTitle>
        <CardDescription className="text-xs text-slate-300">
          Predictive insight into your next likely context and distraction risk.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-[180px_1fr] gap-6 items-center">
          <div className={cn('relative flex h-32 w-32 items-center justify-center rounded-full border border-white/10 bg-gradient-to-br', gaugeClass)}>
            <div className="absolute inset-2 rounded-full bg-black/60 backdrop-blur" />
            <div className="relative z-10 flex flex-col items-center justify-center text-center text-purple-100">
              <span className="text-[11px] uppercase tracking-widest text-slate-300">Risk</span>
              <span className="text-2xl font-semibold">
                {probability ?? '--'}
              </span>
              <span className="text-[11px] text-slate-400">Distraction</span>
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <p className="text-xs uppercase text-slate-400">Predicted Next Focus</p>
              <div className="mt-1 flex flex-col gap-1 text-slate-50">
                <p className="text-lg font-semibold flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-purple-300" />
                  {predictedNext}
                </p>
                {data?.is_stale && (
                  <span className="inline-flex items-center gap-1 rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[11px] uppercase tracking-wide text-slate-400">
                    Waiting for new signals
                  </span>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-slate-300">
              <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                <p className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-slate-400">
                  <Activity className="h-3.5 w-3.5" /> Support
                </p>
                <p className="mt-1 text-base font-semibold text-slate-50">{metricValue(data?.support)}</p>
              </div>
              <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                <p className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-slate-400">
                  <History className="h-3.5 w-3.5" /> History Depth
                </p>
                <p className="mt-1 text-base font-semibold text-slate-50">{metricValue(data?.history_size)}</p>
              </div>
              <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                <p className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-slate-400">
                  <Radar className="h-3.5 w-3.5" /> Horizon
                </p>
                <p className="mt-1 text-base font-semibold text-slate-50">
                  {metricValue(data?.horizon_seconds)}
                  {typeof data?.horizon_seconds === 'number' ? 's' : ''}
                </p>
              </div>
              <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                <p className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-slate-400">
                  <Activity className="h-3.5 w-3.5" /> Transitions
                </p>
                <p className="mt-1 text-base font-semibold text-slate-50">{metricValue(data?.transitions_observed)}</p>
              </div>
            </div>

            {(typeof data?.buffer_events === 'number' || typeof data?.new_events_considered === 'number' || (lastObserved && lastObserved !== 'Unknown')) && (
              <div className="flex flex-wrap gap-2 text-[11px] text-slate-300">
                {typeof data?.buffer_events === 'number' && (
                  <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                    Buffer {data.buffer_events} events
                  </span>
                )}
                {typeof data?.new_events_considered === 'number' && (
                  <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                    +{data.new_events_considered} new
                  </span>
                )}
                {lastObserved && lastObserved !== 'Unknown' && (
                  <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                    Last seen: {lastObserved}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center justify-between border-t border-white/10 pt-4 text-[11px] text-slate-400">
          <span>{hasData ? statusLabel : isActive ? 'Collecting signals…' : 'Start a session to populate insights'}</span>
          <span>{hasData && updatedAt ? `Updated ${updatedAt.toLocaleTimeString()}` : 'Awaiting update'}</span>
        </div>
      </CardContent>
    </Card>
  )
}
