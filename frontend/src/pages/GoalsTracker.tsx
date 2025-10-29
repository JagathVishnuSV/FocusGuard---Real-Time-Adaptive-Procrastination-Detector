import React, { useState, useEffect } from 'react'
import { useApi } from '../hooks/useApi'
import { Target, Calendar } from 'lucide-react'

export default function GoalsTracker() {
  const { data: todayStats } = useApi<any>('/api/stats/today')
  const { data: weekly } = useApi<{ days: string[]; scores: number[] }>('/api/stats/weekly')

  const focusedMinutes = Math.round((todayStats?.focused_time ?? 0))

  const [dailyGoal, setDailyGoal] = useState<number>(() => {
    try { return Number(localStorage.getItem('fg_daily_goal')) || 120 } catch { return 120 }
  })
  const [weeklyGoal, setWeeklyGoal] = useState<number>(() => {
    try { return Number(localStorage.getItem('fg_weekly_goal')) || 600 } catch { return 600 }
  })
  const [currentStreak, setCurrentStreak] = useState(0)

  useEffect(() => {
    if (todayStats?.sessions) {
      // naive streak: count days with focused_time > 0 in weekly data
      try {
        const scores = weekly?.scores ?? []
        const streak = scores.reduce((acc, s) => (s > 0 ? acc + 1 : 0), 0)
        setCurrentStreak(streak)
      } catch { setCurrentStreak(0) }
    }
  }, [weekly, todayStats])

  useEffect(() => {
    try { localStorage.setItem('fg_daily_goal', String(dailyGoal)) } catch {}
  }, [dailyGoal])

  useEffect(() => {
    try { localStorage.setItem('fg_weekly_goal', String(weeklyGoal)) } catch {}
  }, [weeklyGoal])

  const dailyProgress = Math.min(100, Math.round((focusedMinutes / dailyGoal) * 100))

  return (
    <div className="space-y-6">
      <div className="bg-zinc-800 p-6 rounded-lg shadow-sm text-white">
        <h2 className="text-lg font-semibold flex items-center gap-2"><Target className="w-5 h-5 text-blue-400" /> Daily Goal</h2>
        <div className="mt-4 flex items-center gap-6">
          <div className="w-40 h-40 rounded-full bg-zinc-900 flex items-center justify-center text-3xl font-bold">{dailyProgress}%</div>
          <div>
            <p className="text-sm text-gray-300">{focusedMinutes} / {dailyGoal} min</p>
            <div className="mt-4 flex gap-2">
              <button onClick={() => setDailyGoal(d => d + 15)} className="px-3 py-1 bg-indigo-600 text-white rounded">Increase</button>
              <button onClick={() => setDailyGoal(d => Math.max(15, d - 15))} className="px-3 py-1 bg-gray-700 text-white rounded">Decrease</button>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-zinc-800 p-6 rounded-lg shadow-sm text-white">
        <h2 className="text-lg font-semibold flex items-center gap-2"><Calendar className="w-5 h-5 text-purple-400" /> Weekly Goal</h2>
        <p className="mt-2 text-sm text-gray-300">{Math.round((focusedMinutes * 5))} / {weeklyGoal} min</p>
      </div>

      <div className="bg-zinc-800 p-6 rounded-lg shadow-sm text-white">
        <h3 className="font-semibold">Streaks & Achievements</h3>
        <p className="text-sm text-gray-300 mt-2">Current streak: <span className="font-semibold">{currentStreak} days</span></p>
      </div>
    </div>
  )
}
