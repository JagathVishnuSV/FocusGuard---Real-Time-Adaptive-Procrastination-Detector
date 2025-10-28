import React, { useMemo, useState, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Lightbulb, 
  AlertTriangle, 
  Target, 
  Brain,
  ChevronRight,
  BarChart3,
  Zap,
  CalendarClock,
  ShieldCheck,
  ListChecks,
  ExternalLink,
  X
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import type { Insight, DistractionStat } from '@/lib/types'
import { useApi } from '@/hooks/useApi'
import { cn } from '@/lib/utils'

interface FeatureImportanceRow {
  name: string
  importance: number
  description: string
}

interface DistractionTrigger {
  app: string
  hits: number
  avgScore?: number
  maxScore?: number
  dominantContext?: string | null
  lastSeen?: string | null
}

const FEATURE_DESCRIPTION_MAP: Record<string, string> = {
  keystrokes_per_sec: 'Rate of keyboard input during the window',
  clicks_per_sec: 'Mouse click frequency',
  app_switches: 'Number of application switches detected',
  app_entropy: 'Variety of applications used (higher = more contexts)',
  idle_time_ratio: 'Fraction of the window detected as idle',
  productive_app_ratio: 'Proportion of time in productive applications',
  distraction_app_ratio: 'Proportion of time in known distracting apps',
  keystroke_burst_score: 'Burstiness in typing cadence',
  click_burst_score: 'Burstiness in clicking patterns',
  app_switch_frequency: 'Rate of context shifts per second',
  keystroke_variance: 'Variance in typing intervals',
  click_variance: 'Variance in clicking intervals',
  keystroke_click_ratio: 'Typing vs clicking balance',
  idle_transitions: 'Number of idle sessions detected',
  app_focus_duration: 'Average uninterrupted app focus time',
  context_switch_cost: 'Relative cost of rapid context changes',
}

type DrawerType = 'recommendations' | 'focus-block' | 'blocker'

interface ActionConfig {
  type: 'drawer' | 'tab'
  drawer?: DrawerType
  tab?: 'insights' | 'analysis' | 'triggers'
  highlight?: 'analysis' | 'triggers'
}

const INSIGHT_TYPE_LABEL: Record<Insight['type'], string> = {
  danger: 'Critical',
  warning: 'Warning',
  info: 'Opportunity',
  success: 'Win',
}

const INSIGHT_SEVERITY_ORDER: Record<Insight['type'], number> = {
  danger: 0,
  warning: 1,
  info: 2,
  success: 3,
}

export const InsightsPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'insights' | 'analysis' | 'triggers'>('insights')
  const [actionFeedback, setActionFeedback] = useState<string | null>(null)
  const [activeDrawer, setActiveDrawer] = useState<DrawerType | null>(null)
  const [highlightTarget, setHighlightTarget] = useState<'analysis' | 'triggers' | null>(null)
  const { data: insights, isLoading: loadingInsights, error: insightsError } = useApi<Insight[]>('/api/insights')
  const { data: featureImportanceRaw, isLoading: loadingFeatures, error: featureError } = useApi<Record<string, number>>('/api/features/importance')
  const { data: distractionsRaw, isLoading: loadingDistractions, error: distractionError } = useApi<Record<string, DistractionStat>>('/api/distractions/top')

  const actionTabMap = useMemo<Record<string, 'insights' | 'analysis' | 'triggers'>>(() => ({
    'View weekly comparison': 'analysis',
    'View weekly stats': 'analysis',
  }), [])

  const ctaBehavior = useMemo<Record<string, ActionConfig>>(
    () => ({
      'Get recommendations': { type: 'drawer', drawer: 'recommendations' },
      'Schedule focus block': { type: 'drawer', drawer: 'focus-block' },
      'Enable blocker': { type: 'drawer', drawer: 'blocker' },
      'View weekly stats': { type: 'tab', tab: 'analysis', highlight: 'analysis' },
      'View weekly comparison': { type: 'tab', tab: 'analysis', highlight: 'analysis' },
      'View distraction triggers': { type: 'tab', tab: 'triggers', highlight: 'triggers' },
    }),
    []
  )

  const featureImportance: FeatureImportanceRow[] = useMemo(() => {
    if (!featureImportanceRaw) return []
    return Object.entries(featureImportanceRaw)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([key, value]) => ({
        name: key,
        importance: value,
        description: FEATURE_DESCRIPTION_MAP[key] ?? 'Model feature',
      }))
  }, [featureImportanceRaw])

  const distractionTriggers: DistractionTrigger[] = useMemo(() => {
    if (!distractionsRaw) return []
    return Object.entries(distractionsRaw)
      .map(([app, payload]) => {
        if (typeof payload === 'number') {
          return { app, hits: payload }
        }
        return {
          app,
          hits: payload?.hits ?? 0,
          avgScore: payload?.avg_score,
          maxScore: payload?.max_score,
          dominantContext: payload?.dominant_context ?? null,
          lastSeen: payload?.last_seen ?? null,
        }
      })
      .sort((a, b) => {
        const scoreA = typeof a.avgScore === 'number' ? a.avgScore : 0
        const scoreB = typeof b.avgScore === 'number' ? b.avgScore : 0
        if (scoreA !== scoreB) {
          return scoreB - scoreA
        }
        return b.hits - a.hits
      })
  }, [distractionsRaw])

  const sortedInsights = useMemo(() => {
    if (!insights) {
      return []
    }

    return [...insights].sort((a, b) => {
      const aScore = INSIGHT_SEVERITY_ORDER[a.type] ?? 99
      const bScore = INSIGHT_SEVERITY_ORDER[b.type] ?? 99
      return aScore - bScore
    })
  }, [insights])

  const closeDrawer = useCallback(() => setActiveDrawer(null), [])

  const handleInsightAction = useCallback(
    (action: string) => {
      const config = ctaBehavior[action] ?? (actionTabMap[action] ? { type: 'tab', tab: actionTabMap[action] } : null)

      if (!config) {
        setActionFeedback('This recommendation is coming soon.')
        return
      }

      if (config.type === 'drawer') {
        setActiveDrawer(config.drawer ?? null)
        setHighlightTarget(null)
        setActionFeedback(null)
        return
      }

      const targetTab = config.tab ?? 'insights'
      let feedbackMessage: string | null = null

      if (targetTab === 'analysis' && featureImportance.length === 0) {
        feedbackMessage = 'Train the classifier to unlock feature analysis.'
      }

      if (targetTab === 'triggers' && distractionTriggers.length === 0) {
        feedbackMessage = 'No distraction telemetry yet. Keep the session running.'
      }

      if (targetTab !== activeTab) {
        setActiveTab(targetTab)
      }

      if (!feedbackMessage && config.highlight) {
        setHighlightTarget(config.highlight)
      }

      closeDrawer()
      setActionFeedback(feedbackMessage)
    },
    [actionTabMap, activeTab, ctaBehavior, closeDrawer, distractionTriggers.length, featureImportance.length]
  )

  const scheduleFocusBlock = useCallback(
    (minutes: number) => {
      const start = new Date()
      const end = new Date(start.getTime() + minutes * 60000)
      const pad = (value: number) => value.toString().padStart(2, '0')
      const format = (date: Date) =>
        `${date.getUTCFullYear()}${pad(date.getUTCMonth() + 1)}${pad(date.getUTCDate())}` +
        `T${pad(date.getUTCHours())}${pad(date.getUTCMinutes())}${pad(date.getUTCSeconds())}Z`

      const calendarUrl = `https://calendar.google.com/calendar/render?action=TEMPLATE&text=Focus+Guard+Block&details=Deep+work+scheduled+with+FocusGuard&dates=${format(start)}/${format(end)}`

      if (typeof window !== 'undefined') {
        window.open(calendarUrl, '_blank', 'noopener,noreferrer')
        setActionFeedback(`Opened a ${minutes}-minute focus block in your calendar.`)
      } else {
        setActionFeedback('Open your calendar to schedule this focus block manually.')
      }

      closeDrawer()
    },
    [closeDrawer]
  )

  const openRecommendationsResource = useCallback(
    (url: string) => {
      if (typeof window !== 'undefined') {
        window.open(url, '_blank', 'noopener,noreferrer')
        setActionFeedback('Opened recommended resources in a new tab.')
      }
      closeDrawer()
    },
    [closeDrawer]
  )

  const copyBlockerName = useCallback(async () => {
    const appName = 'FocusGuard Ensemble'

    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      try {
        await navigator.clipboard.writeText(appName)
        setActionFeedback('Copied “FocusGuard Ensemble” to your clipboard.')
        return
      } catch (err) {
        console.debug('Clipboard copy failed', err)
      }
    }

    setActionFeedback(`Copy manually: ${appName}`)
  }, [])

  useEffect(() => {
    if (!actionFeedback) {
      return
    }

    const timeout = setTimeout(() => setActionFeedback(null), 4000)
    return () => clearTimeout(timeout)
  }, [actionFeedback])

  useEffect(() => {
    if (!highlightTarget) {
      return
    }

    const timeout = setTimeout(() => setHighlightTarget(null), 3200)
    return () => clearTimeout(timeout)
  }, [highlightTarget])

  const isLoading = loadingInsights || loadingFeatures || loadingDistractions
  const hasError = Boolean(insightsError || featureError || distractionError)

  const getInsightIcon = (type: Insight['type']) => {
    switch (type) {
      case 'success':
        return Target
      case 'warning':
        return AlertTriangle
      case 'danger':
        return AlertTriangle
      case 'info':
        return Lightbulb
      default:
        return Brain
    }
  }

  const getInsightColor = (type: Insight['type']) => {
    switch (type) {
      case 'success':
        return 'text-green-500 bg-green-500/10 border-green-500/20'
      case 'warning':
        return 'text-yellow-500 bg-yellow-500/10 border-yellow-500/20'
      case 'danger':
        return 'text-red-500 bg-red-500/10 border-red-500/20'
      case 'info':
        return 'text-blue-500 bg-blue-500/10 border-blue-500/20'
      default:
        return 'text-gray-500 bg-gray-500/10 border-gray-500/20'
    }
  }

  const renderInsights = () => (
    <div className="space-y-4">
      {sortedInsights.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No personalized insights yet. Keep a session running and provide feedback to unlock this pane.
        </p>
      ) : (
        sortedInsights.map((insight, index) => {
          const Icon = getInsightIcon(insight.type)
          const colorClass = getInsightColor(insight.type)
          const badgeLabel = INSIGHT_TYPE_LABEL[insight.type]

          return (
            <motion.div
              key={`${insight.title}-${index}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.08 }}
            >
              <Card className="hover:shadow-lg hover:shadow-primary/10 transition-shadow border-white/10 bg-white/[0.02] backdrop-blur">
                <CardContent className="p-4">
                  <div className="flex items-start space-x-3">
                    <div className={cn('p-2 rounded-lg', colorClass)}>
                      <Icon className="w-4 h-4" />
                    </div>

                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge
                          variant="outline"
                          className={cn('uppercase tracking-wide text-[10px] font-semibold border px-2 py-0.5', colorClass)}
                        >
                          {badgeLabel}
                        </Badge>
                        {insight.type === 'danger' && (
                          <span className="text-[11px] text-red-300/90">Action needed soon</span>
                        )}
                        {insight.type === 'warning' && (
                          <span className="text-[11px] text-amber-300/90">Monitor closely</span>
                        )}
                      </div>

                      <h4 className="font-semibold text-sm mb-1">{insight.title}</h4>
                      <p className="text-sm text-muted-foreground mb-3 leading-relaxed">
                        {insight.text}
                      </p>

                      <Button
                        variant="ghost"
                        size="sm"
                        className="p-0 h-auto text-primary"
                        onClick={() => handleInsightAction(insight.action)}
                      >
                        {insight.action}
                        <ChevronRight className="w-3 h-3 ml-1" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )
        })
      )}
      {actionFeedback && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground mt-2">
          <Lightbulb className="w-3 h-3" />
          <span>{actionFeedback}</span>
        </div>
      )}
    </div>
  )

  const renderAnalysis = () => (
    <div className="space-y-4">
      {highlightTarget === 'analysis' && (
        <div className="rounded-lg border border-emerald-400/40 bg-emerald-500/10 p-3 text-xs text-emerald-200/90">
          Jumped here from the insights pane. Review the drivers behind today’s predictions below.
        </div>
      )}
      <div>
        <h4 className="font-medium text-sm mb-3 flex items-center">
          <BarChart3 className="w-4 h-4 mr-2" />
          Feature Importance
        </h4>
        <div className="space-y-3">
          {featureImportance.length === 0 ? (
            <p className="text-sm text-muted-foreground">{featureError ? 'Feature importance data unavailable.' : 'No feature insights yet. Train the classifier to populate this section.'}</p>
          ) : featureImportance.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              className="flex items-center space-x-3 p-3 rounded-lg bg-muted/50"
            >
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium font-mono lowercase">{feature.name}</span>
                  <Badge variant="secondary" className="text-xs">
                    {Math.round(feature.importance * 100)}%
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground">{feature.description}</p>
                <div className="w-full bg-muted rounded-full h-1.5 mt-2">
                  <motion.div
                    className="h-1.5 rounded-full bg-primary"
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(feature.importance * 100, 100)}%` }}
                    transition={{ duration: 0.8, delay: index * 0.1 }}
                  />
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  )

  const renderTriggers = () => (
    <div className="space-y-4">
      {highlightTarget === 'triggers' && (
        <div className="rounded-lg border border-rose-400/40 bg-rose-500/10 p-3 text-xs text-rose-200/90">
          These are the distraction sources that triggered your latest alerts.
        </div>
      )}
      <div>
        <h4 className="font-medium text-sm mb-3 flex items-center">
          <Zap className="w-4 h-4 mr-2" />
          Top Distraction Triggers
        </h4>
        <div className="space-y-3">
          {distractionTriggers.length === 0 ? (
            <p className="text-sm text-muted-foreground">{distractionError ? 'No distraction telemetry available.' : 'No distractions recorded yet.'}</p>
          ) : distractionTriggers.map((trigger, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              className="flex items-center justify-between p-3 rounded-lg bg-red-500/5 border border-red-500/20"
            >
              <div>
                <div className="font-medium text-sm">{trigger.app}</div>
                <div className="text-xs text-muted-foreground">
                  {trigger.dominantContext ? `Context: ${trigger.dominantContext}` : 'Tracked distraction source'}
                </div>
                {typeof trigger.avgScore === 'number' && (
                  <div className="text-xs text-muted-foreground mt-1">
                    Avg score: {trigger.avgScore.toFixed(1)}
                    {typeof trigger.maxScore === 'number' ? ` • Max ${trigger.maxScore.toFixed(1)}` : ''}
                  </div>
                )}
                {trigger.lastSeen && (
                  <div className="text-[11px] text-muted-foreground/80 mt-1">
                    Last seen: {new Date(trigger.lastSeen).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                )}
              </div>
              
              <div className="text-right">
                <div className="text-sm font-medium text-red-400">
                  {trigger.hits} hits
                </div>
                <div className="text-xs text-muted-foreground">recent window</div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  )

  const renderDrawer = () => (
    <AnimatePresence>
      {activeDrawer && (
        <motion.div
          key={activeDrawer}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-40 flex justify-end bg-black/60 backdrop-blur-sm"
          onClick={closeDrawer}
          aria-label="Close insight drawer"
        >
          <motion.div
            initial={{ x: 320 }}
            animate={{ x: 0 }}
            exit={{ x: 320 }}
            transition={{ type: 'spring', stiffness: 280, damping: 28 }}
            className="relative h-full w-full max-w-sm bg-slate-950/95 border border-white/10 shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <button
              type="button"
              onClick={closeDrawer}
              className="absolute right-4 top-4 text-muted-foreground hover:text-white"
              aria-label="Close insights panel"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="h-full overflow-y-auto p-6 space-y-4">
              {activeDrawer === 'recommendations' && (
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="rounded-lg bg-primary/10 p-2 text-primary">
                      <ListChecks className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="text-base font-semibold">Personalized Playbook</h3>
                      <p className="text-sm text-muted-foreground">
                        Quick resets to recover focus in the next 5 minutes.
                      </p>
                    </div>
                  </div>
                  <ul className="space-y-3 text-sm text-muted-foreground">
                    <li className="rounded-md border border-white/5 bg-white/[0.03] p-3">
                      <span className="font-medium text-white block">Micro-break reset</span>
                      Step away for two minutes, stretch, and refill water to disrupt distraction loops.
                    </li>
                    <li className="rounded-md border border-white/5 bg-white/[0.03] p-3">
                      <span className="font-medium text-white block">Trigger review</span>
                      Glance at the distraction feed and mark any false positives to sharpen the model.
                    </li>
                    <li className="rounded-md border border-white/5 bg-white/[0.03] p-3">
                      <span className="font-medium text-white block">Re-entry plan</span>
                      Commit to a single task for the next 25 minutes; silence secondary monitors if possible.
                    </li>
                  </ul>
                  <Button
                    variant="secondary"
                    className="w-full"
                    onClick={() => openRecommendationsResource('https://todoist.com/productivity-methods/pomodoro-technique')}
                  >
                    Explore deep work tactics
                    <ExternalLink className="w-3 h-3 ml-2" />
                  </Button>
                </div>
              )}

              {activeDrawer === 'focus-block' && (
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="rounded-lg bg-emerald-500/10 p-2 text-emerald-300">
                      <CalendarClock className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="text-base font-semibold">Schedule a Focus Block</h3>
                      <p className="text-sm text-muted-foreground">
                        One click adds a calendar hold with room for deep work.
                      </p>
                    </div>
                  </div>
                  <div className="grid grid-cols-1 gap-2">
                    {[25, 45, 90].map((duration) => (
                      <Button
                        key={duration}
                        variant="outline"
                        className="justify-between"
                        onClick={() => scheduleFocusBlock(duration)}
                      >
                        {duration} minutes
                        <ChevronRight className="w-3 h-3" />
                      </Button>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Tip: pair the block with Windows Focus Assist or a phone do-not-disturb preset to stay in the zone.
                  </p>
                </div>
              )}

              {activeDrawer === 'blocker' && (
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="rounded-lg bg-rose-500/10 p-2 text-rose-300">
                      <ShieldCheck className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="text-base font-semibold">Reduce Distractions</h3>
                      <p className="text-sm text-muted-foreground">
                        Block repeat offenders in Windows Focus Assist or your browser.
                      </p>
                    </div>
                  </div>
                  <ol className="space-y-3 text-sm text-muted-foreground list-decimal list-inside">
                    <li>
                      Open <span className="text-white">Settings → System → Focus Assist</span> and add disruptive apps to the priority list.
                    </li>
                    <li>
                      For browsers, pin <span className="text-white">chrome://settings/content/siteDetails?site=</span> to quickly mute or block distracting sites.
                    </li>
                    <li>
                      Set a recurring “focus session” automation in PowerToys or your favourite productivity app.
                    </li>
                  </ol>
                  <Button variant="outline" className="justify-start" onClick={copyBlockerName}>
                    Copy “FocusGuard Ensemble” to block list
                  </Button>
                  <p className="text-xs text-muted-foreground">
                    Need inspiration? Pair this with <span className="text-white">LeechBlock</span> (browser) or <span className="text-white">Cold Turkey</span> (desktop) for time-based locks.
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )

  return (
    <>
      <Card className="bg-black/40 border-white/10">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="w-5 h-5" />
            <span>AI Insights & Analysis</span>
          </CardTitle>
          <CardDescription>
            Personalized recommendations based on your behavior patterns
          </CardDescription>
          
          <div className="flex space-x-1 mt-4">
            <Button
              variant={activeTab === 'insights' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setActiveTab('insights')}
            >
              <Lightbulb className="w-4 h-4 mr-2" />
              Insights
            </Button>
            <Button
              variant={activeTab === 'analysis' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setActiveTab('analysis')}
            >
              <BarChart3 className="w-4 h-4 mr-2" />
              Analysis
            </Button>
            <Button
              variant={activeTab === 'triggers' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setActiveTab('triggers')}
            >
              <Zap className="w-4 h-4 mr-2" />
              Triggers
            </Button>
          </div>
        </CardHeader>
        
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <div className="flex space-x-2">
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce delay-75"></div>
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce delay-150"></div>
              </div>
            </div>
          ) : hasError && (insights?.length ?? 0) === 0 && featureImportance.length === 0 && distractionTriggers.length === 0 ? (
            <div className="text-center text-sm text-muted-foreground py-6">
              Insights are unavailable. Train the models and ensure the backend is running.
            </div>
          ) : (
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3 }}
            >
              {activeTab === 'insights' && renderInsights()}
              {activeTab === 'analysis' && renderAnalysis()}
              {activeTab === 'triggers' && renderTriggers()}
            </motion.div>
          )}
        </CardContent>
      </Card>

      {renderDrawer()}
    </>
  )
}