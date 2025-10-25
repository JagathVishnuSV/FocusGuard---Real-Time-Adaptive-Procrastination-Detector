import React, { useState } from 'react'
import { motion } from 'framer-motion'
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { TrendingUp, BarChart3, PieChart as PieChartIcon } from 'lucide-react'
import { useApi, useSessionStatus } from '@/hooks/useApi'
import type { WeeklyStats, HourlyStats, TodayStats } from '@/lib/types'

type ChartType = 'weekly' | 'hourly' | 'focus-breakdown'

interface FocusBreakdownData {
  name: string
  value: number
  color: string
  colorClass: string
}

const COLORS = {
  primary: '#8b5cf6',
  success: '#10b981',
  warning: '#f59e0b',
  danger: '#ef4444',
}

export const FocusChart: React.FC = () => {
  const [chartType, setChartType] = useState<ChartType>('weekly')
  const { data: weeklyData, isLoading: loadingWeekly, error: weeklyError } = useApi<WeeklyStats>('/api/stats/weekly')
  const { data: hourlyData, isLoading: loadingHourly, error: hourlyError } = useApi<HourlyStats>('/api/stats/hourly')
  const { data: todayStats } = useApi<TodayStats>('/api/stats/today')
  const { sessionStatus } = useSessionStatus()

  const isLoading = loadingWeekly || loadingHourly
  const hasError = Boolean(weeklyError || hourlyError)

  const getChartData = () => {
    switch (chartType) {
      case 'weekly':
        return weeklyData ? weeklyData.days.map((day, index) => ({
          name: day,
          value: weeklyData.scores[index] || 0,
        })) : []
      
      case 'hourly':
        return hourlyData ? hourlyData.hours.map((hour, index) => ({
          name: hour,
          value: hourlyData.pattern[index] || 0,
        })) : []
      
      case 'focus-breakdown':
        return buildFocusBreakdownData()
    }
  }

  const buildFocusBreakdownData = (): FocusBreakdownData[] => {
    const sessionStats = sessionStatus?.stats
    const isActive = Boolean(sessionStatus?.active)

    const focusedSeconds = isActive
      ? sessionStats?.focused_time ?? 0
      : (todayStats?.focused_time ?? 0) * 60

    const distractedSeconds = isActive
      ? sessionStats?.distracted_time ?? 0
      : (todayStats?.distracted_time ?? 0) * 60

    const elapsedSeconds = isActive
      ? sessionStats?.elapsed_time ?? 0
      : focusedSeconds + distractedSeconds

    const idleSeconds = Math.max(elapsedSeconds - focusedSeconds - distractedSeconds, 0)
    const total = focusedSeconds + distractedSeconds + idleSeconds

    const toPercent = (value: number) => Number(((value / total) * 100).toFixed(2))

    const baseData: Array<{ name: string; value: number; color: string; colorClass: string }> = [
      {
        name: 'Focused',
        value: total <= 0 ? 0 : toPercent(focusedSeconds),
        color: COLORS.success,
        colorClass: 'bg-emerald-400',
      },
      {
        name: 'Distracted',
        value: total <= 0 ? 0 : toPercent(distractedSeconds),
        color: COLORS.danger,
        colorClass: 'bg-rose-400',
      },
      {
        name: 'Idle',
        value: total <= 0 ? 0 : toPercent(idleSeconds),
        color: COLORS.warning,
        colorClass: 'bg-amber-300',
      },
    ]

    return baseData
  }

  const renderChart = () => {
    if (chartType === 'focus-breakdown') {
      const breakdownData = buildFocusBreakdownData()
      const hasBreakdown = breakdownData.some((item) => item.value > 0)

      if (!hasBreakdown) {
        return (
          <div className="flex items-center justify-center h-[300px] text-sm text-muted-foreground">
            No focus analytics available yet.
          </div>
        )
      }

      const pieData = breakdownData.map(({ name, value, color }) => ({ name, value, color }))

      return (
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={pieData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={100}
              paddingAngle={5}
              dataKey="value"
            >
              {breakdownData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              formatter={(value) => [`${value}%`, 'Percentage']}
              labelStyle={{ color: 'hsl(var(--foreground))' }}
              contentStyle={{
                backgroundColor: 'hsl(var(--card))',
                border: '1px solid hsl(var(--border))',
                borderRadius: '8px',
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      )
    }

    const data = getChartData()
    const hasData = Array.isArray(data) && data.some((item: any) => Number(item?.value ?? 0) > 0)

    if (!hasData) {
      return (
        <div className="flex items-center justify-center h-[300px] text-sm text-muted-foreground">
          No focus analytics available yet.
        </div>
      )
    }

    if (chartType === 'hourly') {
      return (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis 
              dataKey="name" 
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
            />
            <YAxis 
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
            />
            <Tooltip
              formatter={(value) => [`${value}%`, 'Focus Score']}
              labelStyle={{ color: 'hsl(var(--foreground))' }}
              contentStyle={{
                backgroundColor: 'hsl(var(--card))',
                border: '1px solid hsl(var(--border))',
                borderRadius: '8px',
              }}
            />
            <Bar 
              dataKey="value" 
              fill={COLORS.primary}
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      )
    }

    return (
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="colorFocus" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={COLORS.primary} stopOpacity={0.3}/>
              <stop offset="95%" stopColor={COLORS.primary} stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis 
            dataKey="name" 
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
          />
          <YAxis 
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
          />
          <Tooltip
            formatter={(value) => [`${value}%`, 'Focus Score']}
            labelStyle={{ color: 'hsl(var(--foreground))' }}
            contentStyle={{
              backgroundColor: 'hsl(var(--card))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '8px',
            }}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke={COLORS.primary}
            fillOpacity={1}
            fill="url(#colorFocus)"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    )
  }

  return (
    <Card className="h-full bg-black/40 border-white/10">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="w-5 h-5" />
              <span>Focus Analytics</span>
            </CardTitle>
            <CardDescription>
              {chartType === 'weekly' && 'Weekly focus score trend'}
              {chartType === 'hourly' && 'Focus patterns by hour'}
              {chartType === 'focus-breakdown' && 'Time distribution breakdown'}
            </CardDescription>
          </div>
          
          <div className="flex space-x-1">
            <Button
              variant={chartType === 'weekly' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setChartType('weekly')}
            >
              <TrendingUp className="w-4 h-4" />
            </Button>
            <Button
              variant={chartType === 'hourly' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setChartType('hourly')}
            >
              <BarChart3 className="w-4 h-4" />
            </Button>
            <Button
              variant={chartType === 'focus-breakdown' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setChartType('focus-breakdown')}
            >
              <PieChartIcon className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center h-[300px]">
            <div className="flex space-x-2">
              <div className="w-2 h-2 bg-primary rounded-full animate-bounce" />
              <div className="w-2 h-2 bg-primary rounded-full animate-bounce delay-75" />
              <div className="w-2 h-2 bg-primary rounded-full animate-bounce delay-150" />
            </div>
          </div>
        ) : hasError ? (
          <div className="flex items-center justify-center h-[300px] text-slate-300">
            Unable to load focus analytics.
          </div>
        ) : (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
          >
            {renderChart()}
          </motion.div>
        )}
        
        {chartType === 'focus-breakdown' && (
          <div className="flex justify-center space-x-6 mt-4">
            {buildFocusBreakdownData().map((item, index) => (
              <div key={index} className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${item.colorClass}`} />
                <span className="text-sm text-muted-foreground">
                  {item.name} ({item.value}%)
                </span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}