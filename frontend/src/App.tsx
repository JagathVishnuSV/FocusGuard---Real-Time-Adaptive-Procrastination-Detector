import React, { useState, Suspense } from 'react'
import Sidebar from './components/Sidebar'
import './App.css'
import { AnimatePresence, motion } from 'framer-motion'

const Dashboard = React.lazy(() => import('./components/Dashboard'))
const FocusDeepDive = React.lazy(() => import('./pages/FocusDeepDive'))
const GoalsTracker = React.lazy(() => import('./pages/GoalsTracker'))
const PredictiveInsights = React.lazy(() => import('./pages/PredictiveInsights'))

export type PageId = 'dashboard' | 'deepdive' | 'goals' | 'predictive'

function App() {
  const getHashPage = (): PageId => {
    if (typeof window === 'undefined') return 'dashboard'
    const h = (window.location.hash || '').replace('#', '') as PageId
    return h === 'deepdive' || h === 'goals' || h === 'predictive' ? h : 'dashboard'
  }

  const [currentPage, setCurrentPageState] = useState<PageId>(() => getHashPage())
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // wrapper to keep hash in sync
  const setCurrentPage = (p: PageId) => {
    setCurrentPageState(p)
    try { window.location.hash = `#${p}` } catch (e) { /* ignore */ }
  }

  // listen to external hash changes (back/forward)
  React.useEffect(() => {
    const onHash = () => setCurrentPageState(getHashPage())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  // ensure initial hash is set
  React.useEffect(() => {
    if (!window.location.hash) window.location.hash = `#${currentPage}`
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-950 via-black to-zinc-900 text-foreground flex">
      <Sidebar currentPage={currentPage} setCurrentPage={setCurrentPage} sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />

      <main className="flex-1">
        <div className="p-6">
          <Suspense fallback={<div className="p-8 text-center text-sm text-zinc-400">Loading...</div>}>
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={currentPage}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0, transition: { duration: 0.28 } }}
                exit={{ opacity: 0, y: -8, transition: { duration: 0.18 } }}
              >
                {currentPage === 'dashboard' && <Dashboard />}
                {currentPage === 'deepdive' && <FocusDeepDive />}
                {currentPage === 'goals' && <GoalsTracker />}
                {currentPage === 'predictive' && <PredictiveInsights />}
              </motion.div>
            </AnimatePresence>
          </Suspense>
        </div>
      </main>
    </div>
  )
}

export default App
