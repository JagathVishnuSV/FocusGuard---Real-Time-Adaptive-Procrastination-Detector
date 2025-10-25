import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)

  if (hours > 0) {
    return `${hours}h ${minutes}m`
  }
  if (minutes > 0) {
    return `${minutes}m ${secs}s`
  }
  return `${secs}s`
}

export function formatTime(timestamp: number | string): string {
  const date = new Date(timestamp)
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export function formatDate(timestamp: number | string): string {
  const date = new Date(timestamp)
  return date.toLocaleDateString()
}

export function calculateFocusScore(focusedTime: number, totalTime: number): number {
  if (totalTime === 0) return 0
  return Math.round((focusedTime / totalTime) * 100)
}

export function getStatusColor(status: string): string {
  switch (status.toLowerCase()) {
    case 'focused':
    case 'productive':
      return 'text-green-500'
    case 'distracted':
    case 'procrastinating':
      return 'text-red-500'
    case 'idle':
      return 'text-yellow-500'
    default:
      return 'text-gray-500'
  }
}

export function getStatusBgColor(status: string): string {
  switch (status.toLowerCase()) {
    case 'focused':
    case 'productive':
      return 'bg-green-500/10 border-green-500/20'
    case 'distracted':
    case 'procrastinating':
      return 'bg-red-500/10 border-red-500/20'
    case 'idle':
      return 'bg-yellow-500/10 border-yellow-500/20'
    default:
      return 'bg-gray-500/10 border-gray-500/20'
  }
}