import React, { useMemo, useState } from 'react'
import { ResponsiveContainer, Line, XAxis, YAxis, Tooltip, Area, ComposedChart } from 'recharts'
import { AlertTriangle, Brain, TrendingUp } from 'lucide-react'

import { useApi, useSessionStatus } from '../hooks/useApi'
import type { HourlyStats, SessionStatus } from '@/lib/types'

interface WhatIfResponse {
  hour: string
  predicted_focus: number | null
  hours: string[]
  pattern: number[]
  confidence?: number[]
}

interface ForecastPoint {
  hourLabel: string
  baseline: number
  adjusted: number
  selected: boolean
}

interface TriggerSignal {
  context: string
  weight: number
  recommendation?: string | null
}

const percent = (value: number | null | undefined): number | null => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return null
  }
  if (value <= 1) {
    return Math.round(value * 100)
  }
  return Math.round(value)
}

const deriveRiskMeta = (session?: SessionStatus | null) => {
  const liveScore = percent(session?.prediction?.distraction_score ?? session?.prediction?.combined_score ?? null)
  const baselineScore = percent(session?.stats?.combined_score ?? null)
  const score = liveScore ?? baselineScore ?? 55
  const delta = liveScore !== null && baselineScore !== null ? liveScore - baselineScore : null

  const band = score >= 70 ? 'High' : score >= 45 ? 'Medium' : 'Low'
  const descriptions: Record<string, string> = {
    High: 'Model sees elevated distraction pressure — tighten guardrails now.',
    Medium: 'You are in a fluctuating zone; stay proactive to avoid dips.',
    Low: 'Strong focus momentum detected — lean into deep work blocks.',
  }

  return { score, band, delta, description: descriptions[band] }
}

const resolveSelectedIndex = (hours: string[], selectedHour: number) => {
  if (!hours.length) return 0
  const padded = selectedHour.toString().padStart(2, '0')
  const matching = hours.findIndex((label) => label.startsWith(padded))
  if (matching >= 0) {
    return matching
  }
  return Math.min(hours.length - 1, Math.max(0, selectedHour))
}

const buildForecast = (
  stats: HourlyStats | null | undefined,
  whatIf: WhatIfResponse | null | undefined,
  selectedHour: number,
): { points: ForecastPoint[]; selectedIdx: number } => {
  const hours = whatIf?.hours ?? stats?.hours ?? []
  const pattern = whatIf?.pattern ?? stats?.pattern ?? []
  const idx = resolveSelectedIndex(hours, selectedHour)
  if (!hours.length || !pattern.length) {
    return { points: [], selectedIdx: idx }
  }

  const adjustedValue = percent(whatIf?.predicted_focus ?? null)

  const points = hours.map((label, i) => {
    const baseline = percent(pattern[i] ?? null) ?? 0
    const adjusted = i === idx && adjustedValue !== null ? adjustedValue : baseline
    return {
      hourLabel: label ?? `${i}:00`,
      baseline,
      adjusted,
      selected: i === idx,
    }
  })

  return { points, selectedIdx: idx }
}

const buildScenarioTable = (points: ForecastPoint[]): ForecastPoint[] => {
  return [...points]
    .sort((a, b) => b.adjusted - a.adjusted)
    .slice(0, 4)
}

const buildTriggerSignals = (session?: SessionStatus | null, insights?: any[] | null): TriggerSignal[] => {
  const counts = session?.prediction?.context_counts ?? session?.prediction?.context?.counts ?? {}
  const entries = Object.entries(counts ?? {})
    .map(([context, value]) => ({
      context,
      weight: typeof value === 'number' ? value : Number(value) || 0,
      recommendation: insights?.find((ins) => ins.context === context)?.text ?? insights?.find((ins) => ins.context === context)?.message ?? null,
    }))
    .filter((item) => item.weight > 0)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 4)

  return entries
}

const riskColour = (band: string) => {
  if (band === 'High') return 'from-rose-600 via-red-600 to-rose-900 text-white'
  if (band === 'Low') return 'from-emerald-500 via-emerald-600 to-teal-900 text-white'
  return 'from-amber-500 via-yellow-500 to-orange-900 text-slate-900'
}

export default function PredictiveInsights() {
  const [selectedHour, setSelectedHour] = useState<number>(new Date().getHours())
  const { data: insights } = useApi<any[]>('/api/insights')
  const { data: hourlyStats } = useApi<HourlyStats>('/api/stats/hourly')
  const { data: whatIf } = useApi<WhatIfResponse>(`/api/predict/whatif?hour=${selectedHour}`)
  const { sessionStatus } = useSessionStatus()

  const riskMeta = useMemo(() => deriveRiskMeta(sessionStatus), [sessionStatus])
  const forecast = useMemo(() => buildForecast(hourlyStats, whatIf, selectedHour), [hourlyStats, whatIf, selectedHour])
  const scenarioRows = useMemo(() => buildScenarioTable(forecast.points), [forecast.points])
  const triggers = useMemo(() => buildTriggerSignals(sessionStatus, insights), [sessionStatus, insights])
  const cognitiveTwin = sessionStatus?.prediction?.cognitive_twin ?? null

  const currentHourForecast = forecast.points.find((point) => point.selected)

  return (
    <div className="space-y-6">
      {/* 1. Risk Band Overview */}
      <div className={`rounded-xl bg-gradient-to-br p-6 shadow-xl ${riskColour(riskMeta.band)}`}>
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] opacity-80">Distraction Risk</p>
            <div className="mt-3 flex items-baseline gap-3">
              <span className="text-5xl font-bold">{riskMeta.score}%</span>
              <span className="text-base font-semibold">{riskMeta.band}</span>
              {riskMeta.delta !== null && (
                <span className={`text-sm font-semibold ${riskMeta.delta >= 0 ? 'text-rose-100' : 'text-emerald-100'}`}>
                  {riskMeta.delta >= 0 ? '+' : ''}{riskMeta.delta}% vs baseline
                </span>
              )}
            </div>
            <p className="mt-4 text-sm leading-relaxed opacity-90">{riskMeta.description}</p>
          </div>
          {currentHourForecast && (
            <div className="rounded-lg bg-white/15 p-4 text-sm">
              <p className="text-xs uppercase tracking-wide opacity-80">Current Hour Outlook</p>
              <p className="mt-2 text-2xl font-semibold">{currentHourForecast.adjusted}%</p>
              <p className="text-xs opacity-80">Simulated what-if for {currentHourForecast.hourLabel}</p>
            </div>
          )}
        </div>
      </div>

      {/* 2. Forecast + Scenario Simulation */}
      <div className="grid gap-4 lg:grid-cols-[2fr,1fr]">
        <div className="rounded-xl border border-white/5 bg-slate-900/70 p-5 text-slate-100 shadow-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold uppercase tracking-wide text-slate-300">Focus Forecast</p>
              <p className="text-xs text-slate-400">Baseline vs what-if intervention</p>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <label htmlFor="what-if-select" className="text-slate-400">Simulate hour</label>
              <select
                id="what-if-select"
                className="rounded bg-slate-800 px-2 py-1 text-white"
                value={selectedHour}
                onChange={(event) => setSelectedHour(Number(event.target.value))}
                aria-label="Choose hour for what-if forecast"
              >
                {Array.from({ length: 24 }, (_, hour) => (
                  <option key={hour} value={hour}>{`${hour.toString().padStart(2, '0')}:00`}</option>
                ))}
              </select>
            </div>
          </div>

          {forecast.points.length ? (
            <ResponsiveContainer width="100%" height={280}>
              <ComposedChart data={forecast.points} margin={{ top: 20, right: 15, bottom: 5, left: 0 }}>
                <defs>
                  <linearGradient id="forecastFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="hourLabel" tick={{ fill: '#94a3b8', fontSize: 12 }} interval={1} angle={-25} height={70} tickMargin={12} textAnchor="end" />
                <YAxis tick={{ fill: '#94a3b8' }} domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid rgba(148,163,184,0.2)', borderRadius: '0.5rem' }} formatter={(value) => [`${value}%`, 'Focus']} />
                <Area type="monotone" dataKey="baseline" stroke="#475569" fillOpacity={1} fill="url(#forecastFill)" strokeDasharray="4 4" />
                <Line type="monotone" dataKey="adjusted" stroke="#8b5cf6" strokeWidth={2} dot={(props) => {
                  const point = props.payload as ForecastPoint
                  if (point.selected) {
                    return <circle cx={props.cx} cy={props.cy} r={5} fill="#fef3c7" stroke="#a855f7" strokeWidth={2} />
                  }
                  return <circle cx={props.cx} cy={props.cy} r={2} fill="#c4b5fd" opacity={0.5} />
                }} />
              </ComposedChart>
            </ResponsiveContainer>
          ) : (
            <p className="mt-6 text-sm text-slate-400">Forecast unavailable — run a session to populate hourly stats.</p>
          )}
        </div>

        <div className="rounded-xl border border-white/5 bg-slate-900/60 p-5 text-slate-100 shadow-lg">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-sky-300" />
            <p className="text-sm font-semibold uppercase tracking-wide text-slate-300">Upcoming risk pockets</p>
          </div>
          <div className="mt-4 space-y-3">
            {scenarioRows.length ? scenarioRows.map((row) => {
              const band = row.adjusted >= 70 ? 'High' : row.adjusted >= 45 ? 'Medium' : 'Low'
              return (
                <div key={row.hourLabel} className="rounded-lg border border-white/5 bg-slate-950/50 p-3">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-semibold">{row.hourLabel}</span>
                    <span className={`text-xs font-semibold ${band === 'High' ? 'text-rose-300' : band === 'Low' ? 'text-emerald-300' : 'text-amber-200'}`}>
                      {band} risk
                    </span>
                  </div>
                  <div className="mt-2 flex items-center justify-between text-xs text-slate-400">
                    <span>Baseline {row.baseline}%</span>
                    <span>What-if {row.adjusted}%</span>
                  </div>
                </div>
              )
            }) : (
              <p className="text-sm text-slate-400">No forecast points to rank yet.</p>
            )}
          </div>
        </div>
      </div>

      {/* 3. Trigger Radar + Cognitive Twin */}
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-white/5 bg-slate-900/70 p-5 text-slate-100 shadow-lg">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-amber-300" />
            <p className="text-sm font-semibold uppercase tracking-wide text-slate-300">Trigger Radar</p>
          </div>
          <div className="mt-4 space-y-3">
            {triggers.length ? triggers.map((trigger) => (
              <div key={trigger.context} className="rounded-lg border border-white/5 bg-slate-950/50 p-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-semibold">{trigger.context}</span>
                  <span className="text-xs text-slate-400">Signal {trigger.weight}</span>
                </div>
                {trigger.recommendation && (
                  <p className="mt-2 text-xs text-slate-400">{trigger.recommendation}</p>
                )}
              </div>
            )) : (
              <p className="text-sm text-slate-400">Model has not detected repeating triggers yet.</p>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-white/5 bg-slate-900/70 p-5 text-slate-100 shadow-lg">
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-purple-300" />
            <p className="text-sm font-semibold uppercase tracking-wide text-slate-300">Cognitive Twin Outlook</p>
          </div>
          {cognitiveTwin ? (
            <div className="mt-4 space-y-3 text-sm">
              <div>
                <p className="text-xs uppercase tracking-wide text-slate-400">Predicted next context</p>
                <p className="mt-1 text-lg font-semibold text-white">{cognitiveTwin.predicted_next ?? 'Undetermined'}</p>
              </div>
              <div className="grid grid-cols-3 gap-3 text-xs text-slate-400">
                <div className="rounded-lg border border-white/5 bg-slate-950/50 p-3">
                  <p className="text-[10px] uppercase tracking-wide">Probability</p>
                  <p className="mt-1 text-base font-semibold text-white">{cognitiveTwin.prob_distracted !== undefined ? `${Math.round(cognitiveTwin.prob_distracted * 100)}%` : '—'}</p>
                </div>
                <div className="rounded-lg border border-white/5 bg-slate-950/50 p-3">
                  <p className="text-[10px] uppercase tracking-wide">Support</p>
                  <p className="mt-1 text-base font-semibold text-white">{cognitiveTwin.support ?? '—'}</p>
                </div>
                <div className="rounded-lg border border-white/5 bg-slate-950/50 p-3">
                  <p className="text-[10px] uppercase tracking-wide">Horizon</p>
                  <p className="mt-1 text-base font-semibold text-white">{cognitiveTwin.horizon_seconds ? `${Math.round(cognitiveTwin.horizon_seconds / 60)} min` : '—'}</p>
                </div>
              </div>
              <p className="text-xs text-slate-400">Observed apps: {cognitiveTwin.last_app ?? 'unknown'} · buffer {cognitiveTwin.buffer_events ?? 0} events</p>
            </div>
          ) : (
            <p className="mt-4 text-sm text-slate-400">Cognitive twin needs a few more sessions before forecasting reliably.</p>
          )}
        </div>
      </div>
    </div>
  )
}
