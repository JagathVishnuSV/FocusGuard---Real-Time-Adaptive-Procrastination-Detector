import React from 'react'
import { useApi, useSessionStatus } from '../hooks/useApi'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts'
import { Calendar, TrendingUp, Timer } from 'lucide-react'

export default function FocusDeepDive() {
  const { data: activities } = useApi<any[]>('/api/activity/recent')
  const { data: hourly } = useApi<{ hours: string[]; pattern: number[] }>('/api/stats/hourly')
  const { data: features } = useApi<{ feature: string; importance: number }[]>('/api/features/importance')
  const { data: topDistractions } = useApi<any[]>('/api/distractions/top')
  const { sessionStatus } = useSessionStatus()

  // Use backend hourly stats if available, otherwise empty array
  const heatmap = (hourly?.hours ?? []).map((hour, idx) => {
    const raw = hourly?.pattern?.[idx] ?? 0
    // normalize if pattern looks like 0..1
    const value = raw <= 1 ? Math.round(raw * 100) : Math.round(raw)
    return { hour, focus: value }
  })

  // Time to first distraction in minutes since session start (if available)
  let timeToFirstDistraction: string | null = null
  try {
    const first = (activities ?? []).find((a: any) => ((a.label || a.type) || '').toString().toLowerCase().includes('distract'))
    if (first && sessionStatus?.start_time) {
      const started = new Date(sessionStatus.start_time as string)
      const firstT = new Date(first.timestamp)
      const minutes = Math.max(0, Math.round((firstT.getTime() - started.getTime()) / 60000))
      timeToFirstDistraction = `${minutes} min`
    }
  } catch (e) {
    // ignore parsing issues
  }

  // Persona using latest prediction combined_score if available
  const personaScore = sessionStatus?.prediction?.combined_score ?? sessionStatus?.stats?.combined_score ?? 0
  const persona = personaScore > 0.8 ? '🏃 The Morning Sprinter' : personaScore > 0.6 ? '🏃‍♀️ The Consistent Cruiser' : '🐢 The Steady Builder'

  // Build context counts from activity list
  const contextCorrelation = (activities ?? []).reduce((acc: any, a: any) => {
    const ctx = (a.context && (a.context.label || a.context)) || a.context || a.detail || a.app || 'Unknown'
    if (!acc[ctx]) acc[ctx] = { focused: 0, distracted: 0 }
    const isFocused = ((a.label || a.type) === 'Focused') || ((a.prediction?.combined_score ?? a.distraction_score ?? 0) < 0.5)
    if (isFocused) acc[ctx].focused++
    else acc[ctx].distracted++
    return acc
  }, {})

  const correlationData = Object.entries(contextCorrelation).map(([context, data]: any) => ({ context, focusRate: data.focused + data.distracted > 0 ? Math.round((data.focused / (data.focused + data.distracted)) * 100) : 0 }))

  // Distraction fingerprint: detect whether distractions are clustered or scattered.
  const distractionEvents = (activities ?? []).filter((a: any) => {
    const lbl = (a.label || a.type || '').toString().toLowerCase()
    const predScore = a.prediction?.combined_score ?? a.distraction_score ?? a.prediction?.distraction_score
    return lbl.includes('distract') || lbl.includes('procrast') || (predScore !== undefined && predScore >= 0.5)
  }).map((a: any) => new Date(a.timestamp).getTime()).sort((x: number, y: number) => x - y)

  let fingerprint = { type: 'None', description: 'Not enough data' }
  if (distractionEvents.length > 0) {
    const gaps: number[] = []
    for (let i = 1; i < distractionEvents.length; i++) gaps.push((distractionEvents[i] - distractionEvents[i - 1]) / 60000)
    const avgGap = gaps.length ? gaps.reduce((s, g) => s + g, 0) / gaps.length : Infinity
    const clusters = gaps.filter(g => g <= 5).length // gaps <= 5 minutes indicates cluster
    if (avgGap <= 5 || clusters / Math.max(1, gaps.length) > 0.5) {
      fingerprint = { type: 'Clustered', description: 'Distractions happen in bursts — likely interruptions that trigger cascades.' }
    } else if (avgGap > 30) {
      fingerprint = { type: 'Scattered', description: 'Distractions are spread out — frequent minor interruptions.' }
    } else {
      fingerprint = { type: 'Mixed', description: 'A mix of bursts and isolated distractions.' }
    }
  }

  // Top contexts
  const topContexts = Object.entries(contextCorrelation).sort((a: any, b: any) => (b[1].focused + b[1].distracted) - (a[1].focused + a[1].distracted)).slice(0, 5)

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-br from-indigo-800 to-zinc-900 p-6 rounded-lg text-white">
        <h2 className="text-xl font-semibold">AI-Powered Focus Persona</h2>
        <p className="mt-2 text-2xl">{persona}</p>
        <p className="text-sm mt-1 opacity-90">A quick summary derived from your recent sessions</p>
      </div>
      
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-zinc-800 p-4 rounded-lg shadow-sm text-white">
          <h3 className="font-semibold mb-3">Feature Importance</h3>
          {features && features.length ? (
            <ul className="space-y-2">
              {features.slice(0, 8).map((f, i) => (
                <li key={i} className="flex items-center gap-3">
                  <div className="flex-1">
                    <div className="text-sm">{f.feature}</div>
                    <div className="w-full bg-white/5 h-2 rounded mt-1">
                      <div className="h-2 rounded bg-indigo-500" style={{ width: `${Math.round(f.importance * 100)}%` }} />
                    </div>
                  </div>
                  <div className="w-12 text-right text-sm font-semibold">{Math.round(f.importance * 100)}%</div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-300">No feature importance data available.</p>
          )}
        </div>

        <div className="bg-zinc-800 p-4 rounded-lg shadow-sm text-white">
          <h3 className="font-semibold mb-3">Top Distractions</h3>
          {topDistractions && topDistractions.length ? (
            <ul className="space-y-2">
              {topDistractions.slice(0, 8).map((d: any, i: number) => (
                <li key={i} className="flex items-center justify-between">
                  <div className="text-sm">{d.label || d.context || d.app || d.type || 'Unknown'}</div>
                  <div className="text-sm font-semibold">{d.count ?? d.hits ?? d.score ?? 0}</div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-300">No distraction data available.</p>
          )}
        </div>
      </div>

      <div className="grid md:grid-cols-4 gap-4">
        <div className="bg-zinc-800 p-4 rounded-lg shadow-sm text-white">
          <div className="flex items-center gap-3"><Timer className="w-6 h-6 text-indigo-300" /><div>
            <p className="text-xs text-gray-300">Time to First Distraction</p>
            <p className="text-lg font-semibold">{timeToFirstDistraction ?? '—'}</p>
          </div></div>
        </div>

        <div className="bg-zinc-800 p-4 rounded-lg shadow-sm text-white">
          <div className="flex items-center gap-3"><TrendingUp className="w-6 h-6 text-emerald-300" /><div>
            <p className="text-xs text-gray-300">Focus Efficiency</p>
            <p className="text-lg font-semibold">{Math.round((sessionStatus?.prediction?.combined_score ?? sessionStatus?.stats?.combined_score ?? 0) * 100) || 0}%</p>
          </div></div>
        </div>

        <div className="bg-zinc-800 p-4 rounded-lg shadow-sm text-white">
          <div className="flex items-center gap-3"><Calendar className="w-6 h-6 text-orange-300" /><div>
            <p className="text-xs text-gray-300">Peak Hour</p>
            <p className="text-lg font-semibold">{(hourly?.hours && hourly.hours.length) ? `${hourly.hours[hourly.pattern.indexOf(Math.max(...(hourly.pattern || [0])))]}` : `${new Date().getHours()}:00`}</p>
          </div></div>
        </div>

        <div className="bg-zinc-800 p-4 rounded-lg shadow-sm text-white">
          <p className="text-xs text-gray-300">Distraction Fingerprint</p>
          <p className="text-lg font-semibold">{fingerprint.type}</p>
          <p className="text-sm text-gray-400 mt-2">{fingerprint.description}</p>
        </div>
      </div>

      <div className="bg-zinc-800 p-4 rounded-lg shadow-sm text-white">
        <h3 className="font-semibold mb-3">Focus Heatmap (hour of day)</h3>
        {heatmap.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={heatmap}>
              <XAxis dataKey="hour" tick={{ fill: '#cbd5e1' }} />
              <YAxis tick={{ fill: '#cbd5e1' }} />
              <Tooltip />
              <Bar dataKey="focus" fill="#6366f1" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-gray-300">No hourly data available yet.</p>
        )}
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-zinc-800 p-4 rounded-lg shadow-sm text-white">
          <h3 className="font-semibold mb-3">Context Correlation</h3>
          {correlationData.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={correlationData}>
                <XAxis dataKey="context" tick={{ fill: '#cbd5e1' }} />
                <YAxis tick={{ fill: '#cbd5e1' }} />
                <Tooltip />
                <Line type="monotone" dataKey="focusRate" stroke="#10b981" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-gray-300">No context data available.</p>
          )}
        </div>

        <div className="bg-zinc-800 p-4 rounded-lg shadow-sm text-white">
          <h3 className="font-semibold mb-3">Top Contexts</h3>
          <ul className="space-y-2">
            {topContexts.length ? topContexts.map(([ctx, counts]: any, idx: number) => (
              <li key={idx} className="flex items-center justify-between">
                <span className="text-sm">{ctx}</span>
                <span className="text-sm font-semibold">{counts.focused + counts.distracted}</span>
              </li>
            )) : (<li className="text-sm text-gray-300">No contexts yet.</li>)}
          </ul>
        </div>
      </div>
    </div>
  )
}
