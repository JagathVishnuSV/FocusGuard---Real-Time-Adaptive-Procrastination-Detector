import React from 'react'
import { Sparkles, MessageSquareText, Bot } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { GeminiEnrichment } from '@/lib/types'
import { cn } from '@/lib/utils'

interface GeminiHighlightsProps {
  enabled: boolean
  enrichment?: GeminiEnrichment | null
  isActive: boolean
}

const formatGeneratedAt = (timestamp?: string | null) => {
  if (!timestamp) {
    return null
  }

  const parsed = new Date(timestamp)
  if (Number.isNaN(parsed.getTime())) {
    return null
  }

  return parsed.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const GeminiHighlights: React.FC<GeminiHighlightsProps> = ({ enabled, enrichment, isActive }) => {
  const hasEnrichment = Boolean(
    enrichment && (
      enrichment.context_summary ||
      enrichment.focus_insight ||
      (enrichment.prediction_explanation && (enrichment.prediction_explanation.summary || enrichment.prediction_explanation.ghost_narrative))
    )
  )

  const generatedAt = formatGeneratedAt(enrichment?.generated_at ?? undefined)
  const modelLabel = enrichment?.model ?? 'gemini'

  let body: React.ReactNode
  if (!enabled) {
    body = (
      <div className="flex flex-col items-start gap-2 text-sm text-slate-300">
        <p className="text-muted-foreground">
          Gemini enrichment is currently disabled. Set <code className="px-1 py-0.5 rounded bg-slate-900/80 text-xs">ENABLE_GEMINI=1</code> and provide
          an API key in your environment to unlock narrative insights.
        </p>
      </div>
    )
  } else if (!isActive) {
    body = (
      <div className="text-sm text-muted-foreground">
        Start a live session to generate Gemini-powered summaries for your current context and focus signals.
      </div>
    )
  } else if (!hasEnrichment) {
    body = (
      <div className="flex items-center gap-2 text-sm text-slate-300">
        <div className="h-2 w-2 rounded-full bg-primary animate-ping" aria-hidden />
        <span>Gathering context for Gemini…</span>
      </div>
    )
  } else {
    body = (
      <div className="space-y-4">
        {enrichment?.context_summary && (
          <div className="rounded-lg border border-white/5 bg-white/[0.03] p-3">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-300">
              <Sparkles className="h-3.5 w-3.5" />
              Context Snapshot
            </div>
            <p className="mt-2 text-sm text-slate-100 leading-relaxed">{enrichment.context_summary}</p>
          </div>
        )}

        {enrichment?.focus_insight && (
          <div className="rounded-lg border border-primary/20 bg-primary/10 p-3">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-primary-100">
              <Bot className="h-3.5 w-3.5" />
              Focus Insight
            </div>
            <p className="mt-2 text-sm text-primary-50 leading-relaxed">{enrichment.focus_insight}</p>
          </div>
        )}

        {enrichment?.prediction_explanation && (
          <div className="rounded-lg border border-purple-400/20 bg-purple-500/10 p-3 text-purple-50">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide">
              <MessageSquareText className="h-3.5 w-3.5" />
              Prediction Breakdown
            </div>
            {enrichment.prediction_explanation.summary && (
              <p className="mt-2 text-sm leading-relaxed">
                <span className="font-semibold">Summary: </span>
                {enrichment.prediction_explanation.summary}
              </p>
            )}
            {enrichment.prediction_explanation.ghost_narrative && (
              <p className="mt-2 text-sm leading-relaxed">
                <span className="font-semibold">Ghost: </span>
                {enrichment.prediction_explanation.ghost_narrative}
              </p>
            )}
          </div>
        )}
      </div>
    )
  }

  return (
    <Card className={cn('border-white/10 bg-black/40', !enabled && 'border-dashed border-slate-500/40')}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm font-semibold text-muted-foreground">
          <Sparkles className="h-4 w-4 text-primary" />
          Gemini Insights
          {enabled && (
            <Badge variant="outline" className="ml-2 border-primary/40 bg-primary/10 text-[11px] uppercase tracking-wide">
              {modelLabel}
            </Badge>
          )}
        </CardTitle>
        <CardDescription>
          Google Gemini narratives that translate raw telemetry into human-friendly context.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {body}
        {enabled && generatedAt && (
          <p className="text-xs text-muted-foreground">Generated at {generatedAt}</p>
        )}
      </CardContent>
    </Card>
  )
}

export default GeminiHighlights
