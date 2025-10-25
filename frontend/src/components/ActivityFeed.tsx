import React, { useState, useEffect } from 'react'
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
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { apiService, type ActivityEvent } from '@/lib/api'

interface ProcessedEvent extends ActivityEvent {
  id: string
  displayType: 'focus_start' | 'distraction' | 'app_switch' | 'idle' | 'session_end'
  description: string
}

export const ActivityFeed: React.FC = () => {
  const [events, setEvents] = useState<ProcessedEvent[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const processRawEvent = (rawEvent: ActivityEvent, index: number): ProcessedEvent => {
    let displayType: ProcessedEvent['displayType'] = 'app_switch'
    let description = ''

    // Process different event types
    switch (rawEvent.type) {
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
      description
    }
  }

  const fetchActivityData = async () => {
    try {
      setError(null)
      const rawEvents = await apiService.getRecentActivity()
      const processedEvents = rawEvents.map(processRawEvent).reverse()
      setEvents(processedEvents)
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

  const getEventIcon = (type: ProcessedEvent['displayType']) => {
    switch (type) {
      case 'focus_start':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'distraction':
        return <AlertTriangle className="w-4 h-4 text-red-500" />
      case 'app_switch':
        return <Monitor className="w-4 h-4 text-blue-500" />
      case 'idle':
        return <Coffee className="w-4 h-4 text-gray-500" />
      case 'session_end':
        return <XCircle className="w-4 h-4 text-gray-600" />
      default:
        return <Eye className="w-4 h-4 text-gray-400" />
    }
  }

  const getEventBadge = (type: ProcessedEvent['displayType']) => {
    switch (type) {
      case 'focus_start':
        return <Badge variant="outline" className="border-emerald-500/40 bg-emerald-500/10 text-emerald-200">Focus</Badge>
      case 'distraction':
        return <Badge variant="outline" className="border-rose-500/40 bg-rose-500/10 text-rose-200">Distraction</Badge>
      case 'app_switch':
        return <Badge variant="outline" className="border-sky-500/40 bg-sky-500/10 text-sky-200">Switch</Badge>
      case 'idle':
        return <Badge variant="outline" className="border-slate-500/40 bg-slate-500/10 text-slate-300">Idle</Badge>
      case 'session_end':
        return <Badge variant="outline" className="border-slate-500/40 bg-slate-500/10 text-slate-300">Session</Badge>
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
                  {getEventIcon(event.displayType)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    {getEventBadge(event.displayType)}
                    <span className="text-xs text-slate-400">
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-sm text-slate-100 truncate">
                    {event.description}
                  </p>
                  {event.detail && (
                    <p className="text-xs text-slate-400 mt-1 truncate">
                      {event.detail}
                    </p>
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