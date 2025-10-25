import React, { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Lightbulb, 
  TrendingUp, 
  AlertTriangle, 
  Target, 
  Brain,
  ChevronRight,
  BarChart3,
  Clock,
  Zap
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import type { Insight } from '@/lib/types'
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

export const InsightsPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'insights' | 'analysis' | 'triggers'>('insights')
  const { data: insights, isLoading: loadingInsights, error: insightsError } = useApi<Insight[]>('/api/insights')
  const { data: featureImportanceRaw, isLoading: loadingFeatures, error: featureError } = useApi<Record<string, number>>('/api/features/importance')
  const { data: distractionsRaw, isLoading: loadingDistractions, error: distractionError } = useApi<Record<string, number>>('/api/distractions/top')

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
      .sort((a, b) => b[1] - a[1])
      .map(([app, count]) => ({ app, hits: count }))
  }, [distractionsRaw])

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
  {(insights ?? []).map((insight, index) => {
        const Icon = getInsightIcon(insight.type)
        const colorClass = getInsightColor(insight.type)
        
        return (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            <Card className="hover:shadow-md transition-shadow">
              <CardContent className="p-4">
                <div className="flex items-start space-x-3">
                  <div className={cn("p-2 rounded-lg", colorClass)}>
                    <Icon className="w-4 h-4" />
                  </div>
                  
                  <div className="flex-1">
                    <h4 className="font-medium text-sm mb-2">{insight.title}</h4>
                    <p className="text-sm text-muted-foreground mb-3">
                      {insight.text}
                    </p>
                    
                    <Button variant="ghost" size="sm" className="p-0 h-auto text-primary">
                      {insight.action}
                      <ChevronRight className="w-3 h-3 ml-1" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )
      })}
    </div>
  )

  const renderAnalysis = () => (
    <div className="space-y-4">
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
                <div className="text-xs text-muted-foreground">Tracked distraction source</div>
              </div>
              
              <div className="text-right">
                <div className="text-sm font-medium text-red-400">
                  {trigger.hits} hits
                </div>
                <div className="text-xs text-muted-foreground">this week</div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  )

  return (
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
  )
}