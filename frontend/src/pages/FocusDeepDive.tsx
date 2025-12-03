import React, { useMemo } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts'
import { Calendar, TrendingUp, Timer } from 'lucide-react'

import { useApi, useSessionStatus } from '../hooks/useApi'
import type {
  ActivityEvent,
  HourlyStats,
  DistractionStat,
  SessionStatus,
} from '@/lib/types'

interface HeatmapPoint {
  hour: string
  focus: number
}

interface CorrelationPoint {
  context: string
  focusRate: number
  total: number
}

const PIE_COLOURS = ['#22d3ee', '#a855f7', '#fbbf24', '#f97316', '#c084fc']
const PIE_COLOUR_CLASSES = ['bg-cyan-300', 'bg-purple-400', 'bg-amber-300', 'bg-orange-400', 'bg-fuchsia-400']

const normaliseScore = (value: number | null | undefined) => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 0
  }
  if (value <= 1) {
    return Math.round(value * 100)
  }
  return Math.round(value)
}

const buildHeatmap = (stats?: HourlyStats | null): HeatmapPoint[] => {
  if (!stats || !Array.isArray(stats.hours) || !Array.isArray(stats.pattern)) {
    return []
  }

  return stats.hours.map((hour, idx) => {
    const raw = stats.pattern[idx] ?? 0
    return { hour, focus: normaliseScore(raw) }
  })
}

const findFirstDistractionMinutes = (events: ActivityEvent[], session?: SessionStatus | null): string | null => {
  if (!session?.start_time || events.length === 0) {
    return null
  }

  const firstDistraction = events
    .map((evt) => ({ evt, ts: new Date(evt.timestamp).getTime() }))
    .filter(({ evt }) => {
      const label = `${evt.type ?? ''}`.toLowerCase()
      if (label.includes('distract') || label.includes('procrast')) {
        return true
      }
      const distraction = evt.prediction?.distraction_score ?? evt.distraction_score ?? null
      return typeof distraction === 'number' && distraction >= 0.6
    })
    .sort((a, b) => a.ts - b.ts)[0]

  if (!firstDistraction) {
    return null
  }

  const sessionStart = new Date(session.start_time).getTime()
  if (Number.isNaN(sessionStart)) {
    return null
  }

  const minutes = Math.max(0, Math.round((firstDistraction.ts - sessionStart) / 60000))
  return `${minutes} min`
}

const computePersona = (score: number) => {
  if (score >= 0.8) {
    return { label: 'The Morning Sprinter', emoji: '🏃‍♂️', description: 'You ramp into deep work quickly when sessions begin.' }
  }
  if (score >= 0.6) {
    return { label: 'The Consistent Cruiser', emoji: '🚴‍♀️', description: 'You maintain a steady focus cadence across the session.' }
  }
  return { label: 'The Steady Builder', emoji: '🏗️', description: 'Focus is building gradually; micro-habits will help accelerate momentum.' }
}

const buildContextCorrelation = (events: ActivityEvent[]): CorrelationPoint[] => {
  const counts = events.reduce<Record<string, { focused: number; distracted: number }>>((acc, evt) => {
    const key =
      evt.context?.label ||
      evt.prediction?.dominant_context ||
      evt.app ||
      evt.title ||
      'Unknown'
    if (!acc[key]) {
      acc[key] = { focused: 0, distracted: 0 }
    }

    const distractionScore = evt.prediction?.distraction_score ?? evt.distraction_score ?? null
    const isDistracted = typeof distractionScore === 'number'
      ? distractionScore >= 0.6
      : (evt.type ?? '').toLowerCase().includes('distract')
    if (isDistracted) {
      acc[key].distracted += 1
    } else {
      acc[key].focused += 1
    }
    return acc
  }, {})

  return Object.entries(counts)
    .map(([context, data]) => {
      const total = data.focused + data.distracted
      return {
        context,
        focusRate: total > 0 ? Math.round((data.focused / total) * 100) : 0,
        total,
      }
    })
    .sort((a, b) => b.total - a.total)
}

const computeFingerprint = (events: ActivityEvent[]) => {
  const distractionTimestamps = events
    .filter((evt) => {
      const label = `${evt.type ?? ''}`.toLowerCase()
      const distraction = evt.prediction?.distraction_score ?? evt.distraction_score ?? null
      return label.includes('distract') || (typeof distraction === 'number' && distraction >= 0.6)
    })
    .map((evt) => new Date(evt.timestamp).getTime())
    .filter((ts) => !Number.isNaN(ts))
    .sort((a, b) => a - b)

  if (distractionTimestamps.length === 0) {
    return { type: 'None', description: 'Not enough distraction signals yet.' }
  }

  const gaps: number[] = []
  for (let i = 1; i < distractionTimestamps.length; i += 1) {
    gaps.push((distractionTimestamps[i] - distractionTimestamps[i - 1]) / 60000)
  }

  if (gaps.length === 0) {
    return { type: 'Isolated', description: 'Only a single distraction observed so far.' }
  }

  const averageGap = gaps.reduce((sum, gap) => sum + gap, 0) / gaps.length
  const clusterRatio = gaps.filter((gap) => gap <= 5).length / gaps.length

  if (averageGap <= 5 || clusterRatio > 0.5) {
    return { type: 'Clustered', description: 'Distractions arrive in bursts — address trigger events before they cascade.' }
  }
  if (averageGap >= 45) {
    return { type: 'Scattered', description: 'Distractions are rare but unpredictable. Use proactive breaks to stay ahead.' }
  }
  return { type: 'Mixed', description: 'A balance of brief runs and occasional bursts — buffer focused blocks with short resets.' }
}

const buildDistractionList = (stats?: Record<string, DistractionStat> | null) => {
  if (!stats) {
    return []
  }
  return Object.entries(stats)
    .map(([identifier, value]) => {
      const sampleTitle = value.sample_title?.trim()
      const rawLabel = value.label?.trim() || identifier
      const displayLabel = sampleTitle && rawLabel && !rawLabel.toLowerCase().includes(sampleTitle.toLowerCase())
        ? `${sampleTitle}${value.host ? ` · ${value.host}` : ''}`
        : rawLabel
      const detail = value.source_app || value.host || value.window_title || null
      return {
        id: identifier,
        label: displayLabel,
        hits: value.hits ?? value.avg_score ?? value.max_score ?? 0,
        dominantContext: value.dominant_context ?? null,
        detail,
      }
    })
    .sort((a, b) => Number(b.hits) - Number(a.hits))
}

const buildFeatureImportance = (featureMap?: Record<string, number> | null) => {
  if (!featureMap) {
    return []
  }
  return Object.entries(featureMap)
    .map(([feature, importance]) => ({ feature, importance }))
    .sort((a, b) => b.importance - a.importance)
}

export default function FocusDeepDive() {
  const { data: activities } = useApi<ActivityEvent[]>('/api/activity/recent')
  const { data: hourly } = useApi<HourlyStats>('/api/stats/hourly')
  const { data: featureImportance } = useApi<Record<string, number>>('/api/features/importance')
  const { data: distractionMap } = useApi<Record<string, DistractionStat>>('/api/distractions/top')
  const { sessionStatus } = useSessionStatus()

  const events = activities ?? []
  const heatmap = useMemo(() => buildHeatmap(hourly), [hourly])
  const firstDistraction = useMemo(() => findFirstDistractionMinutes(events, sessionStatus), [events, sessionStatus])
  const contextCorrelation = useMemo(() => buildContextCorrelation(events), [events])
  const fingerprint = useMemo(() => computeFingerprint(events), [events])
  const topContexts = contextCorrelation.slice(0, 5)
  const features = useMemo(() => buildFeatureImportance(featureImportance), [featureImportance])
  const topDistractions = useMemo(() => buildDistractionList(distractionMap).slice(0, 8), [distractionMap])

  const personaScore = sessionStatus?.prediction?.combined_score ?? sessionStatus?.stats?.combined_score ?? 0
  const persona = computePersona(personaScore)
  const focusEfficiency = normaliseScore(personaScore)

  const totalFocusMinutes = Math.round((sessionStatus?.stats?.focused_time ?? 0) / 60)
  const productiveShare = contextCorrelation.reduce((acc, entry) => acc + (entry.focusRate >= 60 ? entry.total : 0), 0)
  const contextDiversity = contextCorrelation.length

  return (
    <div className="space-y-6">
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-lg bg-gradient-to-br from-indigo-600 via-purple-600 to-slate-900 p-6 text-white shadow-lg">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold uppercase tracking-wide text-indigo-200">Focus Persona</h2>
              <p className="mt-2 text-3xl font-bold">{persona.emoji} {persona.label}</p>
              <p className="mt-3 max-w-lg text-sm text-indigo-100/80">{persona.description}</p>
            </div>
            <div className="flex h-20 w-20 items-center justify-center rounded-full bg-indigo-500/30 text-2xl font-semibold">
              {focusEfficiency}%
            </div>
          </div>
          <dl className="mt-6 grid grid-cols-3 gap-4 text-xs uppercase tracking-wide text-indigo-100/70">
            <div>
              <dt className="text-indigo-200/70">Focus Minutes Today</dt>
              <dd className="mt-1 text-base font-semibold text-white">{totalFocusMinutes}</dd>
            </div>
            <div>
              <dt className="text-indigo-200/70">High-Intent Contexts</dt>
              <dd className="mt-1 text-base font-semibold text-white">{productiveShare}</dd>
            </div>
            <div>
              <dt className="text-indigo-200/70">Context Diversity</dt>
              <dd className="mt-1 text-base font-semibold text-white">{contextDiversity}</dd>
            </div>
          </dl>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg bg-slate-900/70 p-4 text-white shadow-sm border border-white/5">
          <h3 className="mb-3 font-semibold">Feature Importance</h3>
          {features.length ? (
            <ul className="space-y-2 text-sm">
              {features.slice(0, 8).map((featureItem) => (
                <li key={featureItem.feature} className="flex items-center gap-3">
                  <div className="flex-1">
                    <div className="font-medium text-slate-100">{featureItem.feature}</div>
                    <progress
                      className="mt-1 h-2 w-full overflow-hidden rounded bg-white/10 accent-sky-400"
                      value={Math.round((featureItem.importance ?? 0) * 100)}
                      max={100}
                      aria-hidden
                    />
                  </div>
                  <span className="w-12 text-right text-xs font-semibold text-slate-200">
                    {Math.round((featureItem.importance ?? 0) * 100)}%
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-slate-300">No feature importance data available yet.</p>
          )}
        </div>

        <div className="rounded-lg bg-slate-900/70 p-4 text-white shadow-sm border border-white/5">
          <h3 className="mb-3 font-semibold">Top Distractions</h3>
          {topDistractions.length ? (
            <ul className="space-y-3 text-sm">
              {topDistractions.map((item) => (
                <li key={item.id} className="flex items-center justify-between gap-3">
                  <div>
                    <div className="font-medium text-slate-100">{item.label}</div>
                    {item.detail && (
                      <p className="text-xs text-slate-400">{item.detail}</p>
                    )}
                    {item.dominantContext && (
                      <p className="text-xs text-slate-500">Context: {item.dominantContext}</p>
                    )}
                  </div>
                  <span className="text-xs font-semibold text-slate-200">{item.hits}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-slate-300">No distraction signals collected yet.</p>
          )}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-4">
        <div className="rounded-lg bg-slate-900/70 p-4 text-white shadow-sm border border-white/5">
          <div className="flex items-center gap-3">
            <Timer className="h-6 w-6 text-sky-300" />
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wide">Time to First Distraction</p>
              <p className="text-lg font-semibold text-white">{firstDistraction ?? '—'}</p>
            </div>
          </div>
        </div>

        <div className="rounded-lg bg-slate-900/70 p-4 text-white shadow-sm border border-white/5">
          <div className="flex items-center gap-3">
            <TrendingUp className="h-6 w-6 text-emerald-300" />
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wide">Focus Efficiency</p>
              <p className="text-lg font-semibold text-white">{focusEfficiency}%</p>
            </div>
          </div>
        </div>

        <div className="rounded-lg bg-slate-900/70 p-4 text-white shadow-sm border border-white/5">
          <div className="flex items-center gap-3">
            <Calendar className="h-6 w-6 text-amber-300" />
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wide">Peak Hour</p>
              <p className="text-lg font-semibold text-white">
                {hourly?.hours && hourly.pattern?.length
                  ? hourly.hours[hourly.pattern.indexOf(Math.max(...hourly.pattern))] ?? '—'
                  : `${new Date().getHours()}:00`}
              </p>
            </div>
          </div>
        </div>

        <div className="rounded-lg bg-slate-900/70 p-4 text-white shadow-sm border border-white/5">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Distraction Fingerprint</p>
          <p className="mt-2 text-lg font-semibold text-white">{fingerprint.type}</p>
          <p className="mt-2 text-xs text-slate-300 leading-relaxed">{fingerprint.description}</p>
        </div>
      </div>

      <div className="rounded-lg bg-slate-900/70 p-4 text-white shadow-sm border border-white/5">
        <h3 className="mb-3 font-semibold">Focus Heatmap (hour of day)</h3>
        {heatmap.length ? (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={heatmap}>
              <XAxis dataKey="hour" tick={{ fill: '#cbd5e1' }} interval={0} angle={-35} height={70} textAnchor="end" />
              <YAxis tick={{ fill: '#cbd5e1' }} domain={[0, 100]} />
              <Tooltip cursor={{ fill: 'rgba(148, 163, 184, 0.15)' }} />
              <Bar dataKey="focus" fill="#6366f1" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-slate-300">No hourly focus profile available yet.</p>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-lg bg-slate-900/70 p-4 text-white shadow-sm border border-white/5">
          <h3 className="mb-3 font-semibold">Context Correlation</h3>
          {contextCorrelation.length ? (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={contextCorrelation}>
                <XAxis dataKey="context" tick={{ fill: '#cbd5e1' }} interval={0} angle={-25} height={80} textAnchor="end" />
                <YAxis tick={{ fill: '#cbd5e1' }} domain={[0, 100]} />
                <Tooltip cursor={{ stroke: '#0ea5e9', strokeWidth: 1 }} />
                <Line type="monotone" dataKey="focusRate" stroke="#34d399" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-slate-300">No context correlation data captured yet.</p>
          )}
        </div>

        <div className="rounded-lg bg-slate-900/70 p-4 text-white shadow-sm border border-white/5">
          <h3 className="mb-3 font-semibold">Top Context Mix</h3>
          {topContexts.length ? (
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={topContexts.map((ctx) => ({ name: ctx.context, value: ctx.total }))}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={2}
                >
                  {topContexts.map((ctx, index) => (
                    <Cell key={ctx.context} fill={PIE_COLOURS[index % PIE_COLOURS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number, _, payload) => [`${value} events`, payload?.payload?.name ?? '']} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-slate-300">No dominant contexts yet. Keep logging sessions.</p>
          )}
          <ul className="mt-4 space-y-2 text-sm">
            {topContexts.map((ctx, index) => (
              <li key={ctx.context} className="flex items-center gap-3">
                <span
                  className={`inline-flex h-3 w-3 flex-shrink-0 rounded-full ${PIE_COLOUR_CLASSES[index % PIE_COLOUR_CLASSES.length]}`}
                />
                <span className="flex-1 text-slate-200">{ctx.context}</span>
                <span className="text-xs text-slate-400">{ctx.total} events</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}
