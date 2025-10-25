import React from 'react'
import { motion } from 'framer-motion'
import type { LucideIcon } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'

interface MetricsCardProps {
  title: string
  value: string
  icon: LucideIcon
  color: 'primary' | 'success' | 'warning' | 'danger' | 'info'
  description?: string
  progress?: number
  trend?: {
    value: number
    isPositive: boolean
  }
}

const colorClasses = {
  primary: {
    icon: 'text-primary bg-primary/10',
    progress: 'bg-primary',
  },
  success: {
    icon: 'text-green-500 bg-green-500/10',
    progress: 'bg-green-500',
  },
  warning: {
    icon: 'text-yellow-500 bg-yellow-500/10',
    progress: 'bg-yellow-500',
  },
  danger: {
    icon: 'text-red-500 bg-red-500/10',
    progress: 'bg-red-500',
  },
  info: {
    icon: 'text-blue-500 bg-blue-500/10',
    progress: 'bg-blue-500',
  },
}

export const MetricsCard: React.FC<MetricsCardProps> = ({
  title,
  value,
  icon: Icon,
  color,
  description,
  progress,
  trend,
}) => {
  const colorClass = colorClasses[color]

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
    >
      <Card className="relative overflow-hidden group hover:shadow-lg transition-all duration-300">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              {title}
            </CardTitle>
            <div className={cn("p-2 rounded-lg", colorClass.icon)}>
              <Icon className="w-4 h-4" />
            </div>
          </div>
        </CardHeader>
        
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-baseline space-x-2">
              <div className="text-2xl font-bold">{value}</div>
              {trend && (
                <div className={cn(
                  "text-xs font-medium flex items-center",
                  trend.isPositive ? "text-green-500" : "text-red-500"
                )}>
                  <span>{trend.isPositive ? '+' : '-'}{Math.abs(trend.value)}%</span>
                </div>
              )}
            </div>
            
            {description && (
              <CardDescription className="text-xs">
                {description}
              </CardDescription>
            )}
            
            {progress !== undefined && (
              <div className="space-y-1">
                <div className="w-full bg-muted rounded-full h-1.5">
                  <motion.div
                    className={cn("h-1.5 rounded-full", colorClass.progress)}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(Math.max(progress, 0), 100)}%` }}
                    transition={{ duration: 1, delay: 0.2 }}
                  />
                </div>
              </div>
            )}
          </div>
        </CardContent>
        
        {/* Hover effect overlay */}
        <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
      </Card>
    </motion.div>
  )
}