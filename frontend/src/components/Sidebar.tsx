import React from 'react'
import { Home, BarChart3, Target, Lightbulb, Menu, X, Brain } from 'lucide-react'
import type { PageId } from '../App'

interface Props {
  currentPage: PageId
  setCurrentPage: (p: PageId) => void
  sidebarOpen: boolean
  setSidebarOpen: (v: boolean) => void
}

export default function Sidebar({ currentPage, setCurrentPage, sidebarOpen, setSidebarOpen }: Props) {
  const navItems: { id: PageId; label: string; icon: any }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: Home },
    { id: 'deepdive', label: 'Focus Deep Dive', icon: BarChart3 },
    { id: 'goals', label: 'Goals & Streaks', icon: Target },
    { id: 'predictive', label: 'Predictive Insights', icon: Lightbulb },
  ]

  const onKeyActivate = (e: React.KeyboardEvent, id: PageId) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      setCurrentPage(id)
    }
  }

  return (
    <aside
      role="navigation"
      aria-label="Main navigation"
      className={`${sidebarOpen ? 'w-64' : 'w-20'} bg-white/5 backdrop-blur-sm border-r border-white/6 transition-all duration-300 flex flex-col`}
    > 
      <div className="p-4 border-b border-white/6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Brain className="w-8 h-8 text-indigo-400" aria-hidden />
          {sidebarOpen && (
            <div>
              <h1 className="text-lg font-semibold">FocusGuard</h1>
              <p className="text-xs text-gray-300">AI-powered focus coach</p>
            </div>
          )}
        </div>
        <button
          aria-label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          aria-expanded={sidebarOpen}
          className="p-2 rounded-md hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          onClick={() => setSidebarOpen(!sidebarOpen)}
        >
          {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      <nav className="flex-1 p-3 space-y-2" aria-label="Primary">
        {navItems.map(({ id, label, icon: Icon }) => (
          <div key={id}>
            <button
              role="link"
              aria-current={currentPage === id ? 'page' : undefined}
              onClick={() => setCurrentPage(id)}
              onKeyDown={(e) => onKeyActivate(e, id)}
              title={!sidebarOpen ? label : undefined}
              tabIndex={0}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-md transition-all text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${currentPage === id ? 'bg-indigo-600/20 text-indigo-300' : 'text-gray-200 hover:bg-white/5'}`}
            >
              <Icon className="w-5 h-5" aria-hidden />
              {sidebarOpen && <span className="truncate">{label}</span>}
            </button>
          </div>
        ))}
      </nav>

      <div className="p-3 border-t border-white/6">
        {sidebarOpen ? (
          <div className="text-xs text-gray-300">Tips: Try the Focus Deep Dive for patterns & persona.</div>
        ) : (
          <div className="text-center text-xs text-gray-300" aria-hidden>Tips</div>
        )}
      </div>
    </aside>
  )
}
