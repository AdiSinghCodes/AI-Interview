import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'

const ROLES = [
  { id: 'technical', icon: '💻', name: 'General Technical', desc: 'Core CS, APIs, databases, concurrency, reliability' },
  { id: 'python', icon: '🐍', name: 'Python Backend', desc: 'Language internals, the data model, asyncio, testing' },
  { id: 'react', icon: '⚛️', name: 'React Frontend', desc: 'Rendering, hooks, state, performance, architecture' },
  { id: 'dsa', icon: '🧩', name: 'DSA / Problem Solving', desc: 'Arrays, hashing, trees, complexity — think aloud' },
  { id: 'hr', icon: '👥', name: 'HR / Behavioural', desc: 'Teamwork, ownership, conflict, motivation' },
]
const COUNTS = [5, 8, 10, 12]

export default function Setup() {
  const nav = useNavigate()
  const [role, setRole] = useState('technical')
  const [numQuestions, setNum] = useState(8)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const start = async () => {
    setBusy(true)
    setErr('')
    try {
      const s = await api.createSession({ role, num_questions: numQuestions })
      nav(`/preflight/${s.session_id}`)
    } catch (e) {
      setErr(String(e.message || e))
      setBusy(false)
    }
  }

  return (
    <div className="page">
      <p className="kicker">NTRVSTA · AI Voice Interview</p>
      <h1>Set up your interview</h1>
      <p className="dim">Pick a round. The interviewer talks to you by voice — answer out loud, naturally.</p>

      <h2 style={{ marginTop: '1.5rem' }}>Round</h2>
      <div className="roles">
        {ROLES.map((r) => (
          <button key={r.id} className={`role-btn ${role === r.id ? 'sel' : ''}`} onClick={() => setRole(r.id)}>
            <div className="icon">{r.icon}</div>
            <div className="name">{r.name}</div>
            <div className="desc">{r.desc}</div>
          </button>
        ))}
      </div>

      <h2>Questions</h2>
      <div className="row">
        {COUNTS.map((c) => (
          <button key={c} className={`pill ${numQuestions === c ? 'sel' : ''}`} onClick={() => setNum(c)}>{c}</button>
        ))}
      </div>

      {err && <p style={{ color: 'var(--bad)' }}>{err}</p>}
      <div style={{ marginTop: '1.8rem' }}>
        <button className="primary" onClick={start} disabled={busy}>
          {busy ? 'Preparing…' : 'Continue to device check'}
        </button>
      </div>
    </div>
  )
}
