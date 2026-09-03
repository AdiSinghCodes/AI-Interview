import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import Avatar from '../components/Avatar'
import TranscriptPanel from '../components/TranscriptPanel'
import { api } from '../api'
import { useMediaStream } from '../hooks/useMediaStream'
import { useInterviewSession } from '../hooks/useInterviewSession'

const BADGE = {
  idle: 'Ready when you are',
  connecting: 'Connecting…',
  speaking: 'Interviewer is speaking',
  thinking: 'Thinking…',
  listening: 'Your turn — answer out loud',
  ended: 'Interview complete',
  error: 'Connection lost',
}

export default function InterviewRoom() {
  const { sessionId } = useParams()
  const nav = useNavigate()
  const { videoRef, stream, error: mediaError } = useMediaStream()
  const [info, setInfo] = useState(null)
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    api.session(sessionId).then(setInfo).catch(() => {})
  }, [sessionId])

  const { phase, turns, partial, amplitude, begin, end } = useInterviewSession(info?.round_id, stream)

  useEffect(() => {
    const t = setInterval(() => setElapsed((s) => s + 1), 1000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    if (phase === 'ended' && info?.round_id) {
      const t = setTimeout(() => nav(`/summary/${info.round_id}`), 3000)
      return () => clearTimeout(t)
    }
  }, [phase, info, nav])

  const asked = useMemo(
    () => turns.filter((t) => t.role === 'interviewer' && !['opening', 'closing'].includes(t.kind)).length,
    [turns],
  )
  const mm = String(Math.floor(elapsed / 60)).padStart(2, '0')
  const ss = String(elapsed % 60).padStart(2, '0')
  const ready = Boolean(info?.round_id && stream)

  if (mediaError) {
    return (
      <div className="page">
        <div className="card">
          <h1>Camera / microphone blocked</h1>
          <p className="dim">{mediaError}</p>
          <p>Allow access in your browser's address bar, then reload this page.</p>
          <button className="primary" onClick={() => nav('/')}>Back to start</button>
        </div>
      </div>
    )
  }

  return (
    <div className="room">
      <div className="left">
        <div className="avatar-wrap">
          <Avatar speaking={phase === 'speaking'} amplitude={amplitude} />
          {phase === 'idle' && (
            <div className="begin-overlay">
              <p className="dim">
                {info ? `${info.persona_name} is ready for your ${info.role} round.` : 'Getting things ready…'}
              </p>
              <p style={{ fontSize: '0.85rem', color: 'var(--warn)' }}>Headphones on. Answer out loud, naturally.</p>
              <button className="primary" disabled={!ready} onClick={begin}>
                {ready ? 'Begin interview' : 'Preparing…'}
              </button>
            </div>
          )}
        </div>

        <div className="statusbar card" style={{ padding: '0.8rem 1rem' }}>
          <span className={`badge ${phase}`}>
            {phase === 'thinking' && <span className="spinner" />}
            {phase === 'listening' && <span className="rec-dot" />}
            {BADGE[phase] || phase}
          </span>
          <span className="qmeta">
            {info ? `${asked}/${info.num_questions} questions` : ''} · {mm}:{ss}
          </span>
        </div>
      </div>

      <div className="right">
        <div className="statusbar">
          <video ref={videoRef} className="webcam-thumb" autoPlay playsInline muted />
          <div className="controls">
            {phase !== 'idle' && phase !== 'ended' && (
              <button onClick={() => { end(); nav(`/summary/${info?.round_id}`) }}>End interview</button>
            )}
          </div>
        </div>
        <div className="card" style={{ flex: 1, display: 'flex', minHeight: 0 }}>
          {turns.length === 0 && phase === 'idle' ? (
            <p className="dim" style={{ margin: 'auto' }}>The conversation will appear here.</p>
          ) : (
            <TranscriptPanel turns={turns} partial={partial} />
          )}
        </div>
      </div>
    </div>
  )
}
