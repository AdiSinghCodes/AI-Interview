import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'

export default function Summary() {
  const { roundId } = useParams()
  const nav = useNavigate()
  const [turns, setTurns] = useState([])
  const [summary, setSummary] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    ;(async () => {
      try {
        const t = await api.transcript(roundId)
        if (alive) setTurns(t.turns || [])
        const s = await api.summary(roundId)
        if (alive) setSummary(s.summary || '')
      } catch (e) {
        if (alive) setSummary(`Could not generate a summary: ${e.message || e}`)
      } finally {
        if (alive) setLoading(false)
      }
    })()
    return () => { alive = false }
  }, [roundId])

  return (
    <div className="page">
      <p className="kicker">Interview complete</p>
      <h1>How it went</h1>

      <div className="card" style={{ marginBottom: '1.2rem' }}>
        <h2>Interviewer's read</h2>
        {loading ? <p className="dim">Writing up the interview…</p> : <p style={{ lineHeight: 1.6 }}>{summary}</p>}
      </div>

      <div className="card">
        <h2>Transcript</h2>
        <div className="transcript" style={{ maxHeight: '55vh' }}>
          {turns.map((t) => (
            <div key={t.index} className={`bubble ${t.role === 'candidate' ? 'candidate' : 'interviewer'}`}>
              <span className="who">{t.role === 'candidate' ? 'You' : 'Interviewer'}</span>
              <p>{t.text}</p>
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginTop: '1.5rem' }}>
        <button className="primary" onClick={() => nav('/')}>New interview</button>
      </div>
    </div>
  )
}
