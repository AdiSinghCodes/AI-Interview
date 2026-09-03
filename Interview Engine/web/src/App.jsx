import { Routes, Route, Navigate } from 'react-router-dom'
import Setup from './pages/Setup'
import Preflight from './pages/Preflight'
import InterviewRoom from './pages/InterviewRoom'
import Summary from './pages/Summary'
import './app.css'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Setup />} />
      <Route path="/preflight/:sessionId" element={<Preflight />} />
      <Route path="/interview/:sessionId" element={<InterviewRoom />} />
      <Route path="/summary/:roundId" element={<Summary />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
