import React, { useMemo, useState } from 'react'
import { useApi, useSessionStatus } from '../hooks/useApi'
import { Lightbulb } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

export default function PredictiveInsights() {
  const { data: insights } = useApi<any[]>('/api/insights')
  const { data: hourlyStats } = useApi<{ hours: string[]; pattern: number[] }>('/api/stats/hourly')
  const { sessionStatus } = useSessionStatus()

  const risk = useMemo(() => {
    const score = sessionStatus?.prediction?.combined_score ?? sessionStatus?.stats?.combined_score
    if (typeof score === 'number') return score > 0.7 ? 'High' : score > 0.4 ? 'Medium' : 'Low'
    const h = new Date().getHours()
    return h >= 14 && h <= 16 ? 'High' : h >= 10 && h <= 12 ? 'Low' : 'Medium'
  }, [sessionStatus])

  const [selectedHour, setSelectedHour] = useState<number>(() => new Date().getHours())
  // Use backend what-if prediction endpoint when available
  const { data: whatif } = useApi<{ hour: string; predicted_focus: number | null; hours: string[]; pattern: number[] }>(
    `/api/predict/whatif?hour=${selectedHour}`
  )

  // Build baseline forecast from backend pattern (prefer whatif.pattern when available)
  const basePattern = whatif?.pattern ?? hourlyStats?.pattern ?? []
  const hours = whatif?.hours ?? hourlyStats?.hours ?? []
  const baselineForecast = hours.map((h, i) => ({ h, v: Math.round(basePattern?.[i] ?? 0) }))

  // Create adjusted forecast that applies the what-if predicted focus for the selected hour (if provided)
  const adjustedForecast = baselineForecast.map((d, i) => ({
    ...d,
    v: i === selectedHour && typeof whatif?.predicted_focus === 'number' ? Math.round(whatif!.predicted_focus as number) : d.v,
    selected: i === selectedHour,
  }))

  const tipsForRisk: Record<string, string[]> = {
    High: [
      'Your distraction risk is high. Try muting notifications and closing non-essential tabs.',
      'Use a short Pomodoro (25/5) to regain control.'
    ],
    Medium: [
      'Light distractions expected. Time-box small tasks and batch notifications.'
    ],
    Low: [
      'Low distraction risk — schedule deep work now for best results.'
    ]
  }

  const predictedForSelected = whatif?.predicted_focus ?? hourlyStats?.pattern?.[selectedHour] ?? null

  return (
    <div className="space-y-6">
      <div className={`p-6 rounded-lg ${risk === 'High' ? 'bg-red-700 text-white' : risk === 'Low' ? 'bg-emerald-700 text-white' : 'bg-yellow-600 text-black'}`}>
        <div className="flex items-center gap-3">
          <Lightbulb className="w-6 h-6" />
          <div>
            <h3 className="font-semibold">Distraction Risk: {risk}</h3>
            <p className="text-sm opacity-90">Based on latest model prediction</p>
            <ul className="mt-2 list-disc list-inside text-sm">
              {(tipsForRisk[risk] || []).map((t, i) => <li key={i}>{t}</li>)}
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-zinc-800 p-4 rounded-lg shadow-sm text-white">
        <h3 className="font-semibold">Focus Forecast (hourly)</h3>
        {baselineForecast.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={adjustedForecast}>
              <XAxis dataKey="h" tick={{ fill: '#cbd5e1' }} />
              <YAxis tick={{ fill: '#cbd5e1' }} />
              <Tooltip />
              {/* Baseline (dashed) */}
              <Line type="monotone" data={baselineForecast} dataKey="v" stroke="#475569" strokeWidth={1} strokeDasharray="4 4" dot={false} />
              {/* Adjusted / what-if (solid) */}
              <Line
                type="monotone"
                data={adjustedForecast}
                dataKey="v"
                stroke="#8b5cf6"
                strokeWidth={2}
                dot={(props: any) => {
                  const payload = props.payload as any
                  const cx = props.cx as number
                  const cy = props.cy as number
                  if (payload?.selected) return <circle cx={cx} cy={cy} r={5} fill="#ffdd57" stroke="#fff" />
                  // Always return an SVG element (small, low-opacity circle) to satisfy Recharts typing
                  return <circle cx={cx} cy={cy} r={2} fill="#8b5cf6" opacity={0.35} />
                }}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-gray-300">No hourly forecast available.</p>
        )}

        <div className="mt-4 flex items-center gap-3">
          <label htmlFor="whatif-hour" className="text-sm text-gray-300">What-if: Choose an hour</label>
            <select id="whatif-hour" aria-label="Choose hour for what-if simulation" value={selectedHour} onChange={(e) => setSelectedHour(Number(e.target.value))} className="bg-zinc-900 text-white px-2 py-1 rounded">
            {Array.from({ length: 24 }, (_, i) => <option key={i} value={i}>{i}:00</option>)}
          </select>
          <div className="ml-auto text-sm">
            {predictedForSelected !== null ? (
              <span>Predicted focus: <strong>{Math.round(predictedForSelected)}%</strong></span>
            ) : <span className="text-gray-400">No data</span>}
          </div>
        </div>
      </div>

      <div className="bg-zinc-800 p-4 rounded-lg shadow-sm text-white">
        <h3 className="font-semibold">Context-Specific Recommendations</h3>
        <div className="mt-3 space-y-2">
          {(insights || []).slice(0, 5).map((ins: any, i: number) => (
            <div key={i} className="p-3 rounded border border-white/5 bg-zinc-900">
              <div className="text-sm font-medium">{ins.title || ins.type || 'Recommendation'}</div>
              <div className="text-xs text-gray-300 mt-1">{ins.text || ins.message}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
