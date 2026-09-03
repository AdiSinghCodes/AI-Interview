import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import { useMediaStream } from '../hooks/useMediaStream'

export default function Preflight() {
  const { sessionId } = useParams()
  const nav = useNavigate()
  const { videoRef, stream, hasPermission, error } = useMediaStream()

  const [level, setLevel] = useState(0)
  const [micPeak, setMicPeak] = useState(0)
  const [headphones, setHeadphones] = useState(false)
  const [prep, setPrep] = useState({ done: 0, total: 0, ready: false })
  const [msg, setMsg] = useState('')
  const rafRef = useRef(null)

  // mic level meter
  useEffect(() => {
    if (!stream) return
    const ctx = new AudioContext()
    const src = ctx.createMediaStreamSource(stream)
    const an = ctx.createAnalyser()
    an.fftSize = 512
    src.connect(an)
    const buf = new Uint8Array(an.fftSize)
    const loop = () => {
      an.getByteTimeDomainData(buf)
      let sum = 0
      for (let i = 0; i < buf.length; i++) {
        const v = (buf[i] - 128) / 128
        sum += v * v
      }
      const rms = Math.sqrt(sum / buf.length)
      setLevel(rms)
      setMicPeak((p) => Math.max(p * 0.995, rms))
      rafRef.current = requestAnimationFrame(loop)
    }
    loop()
    return () => {
      cancelAnimationFrame(rafRef.current)
      ctx.close().catch(() => {})
    }
  }, [stream])

  // poll prefetch progress
  useEffect(() => {
    let alive = true
    const poll = async () => {
      try {
        const st = await api.sessionStatus(sessionId)
        if (alive) setPrep(st)
        if (!st.ready && alive) setTimeout(poll, 1200)
      } catch {
        if (alive) setTimeout(poll, 2000)
      }
    }
    poll()
    return () => { alive = false }
  }, [sessionId])

  const micOk = micPeak > 0.02
  const canStart = hasPermission && micOk && headphones && prep.ready

  const start = async () => {
    setMsg('')
    try {
      const v = await api.preflight({ mic_rms: micPeak, headphones, camera_ok: hasPermission })
      if (!v.ok) return setMsg(v.reasons.join(' '))
      nav(`/interview/${sessionId}`)
    } catch (e) {
      setMsg(String(e.message || e))
    }
  }

  const pct = prep.total ? Math.round((prep.done / prep.total) * 100) : 0

  return (
    <div className="page">
      <p className="kicker">Device check</p>
      <h1>Quick setup before we begin</h1>

      <div className="pf-grid card">
        <div>
          <video ref={videoRef} className="pf-video" autoPlay playsInline muted />
          {error && <p style={{ color: 'var(--bad)' }}>Camera/mic: {error}</p>}
        </div>

        <div>
          <div className="check">
            <span className={`dot ${hasPermission ? 'ok' : 'bad'}`} />
            <span>Camera & microphone access</span>
          </div>

          <div className="check" style={{ display: 'block' }}>
            <div className="row" style={{ marginBottom: '0.35rem' }}>
              <span className={`dot ${micOk ? 'ok' : ''}`} />
              <span>Microphone — say a sentence out loud</span>
            </div>
            <div className="meter"><span style={{ width: `${Math.min(100, level * 400)}%` }} /></div>
          </div>

          <label className="check" style={{ cursor: 'pointer' }}>
            <input type="checkbox" checked={headphones} onChange={(e) => setHeadphones(e.target.checked)} />
            <span>I'm wearing headphones (required — stops the interviewer's voice leaking into the mic)</span>
          </label>

          <div className="check" style={{ display: 'block', borderBottom: 'none' }}>
            <div className="row" style={{ marginBottom: '0.35rem' }}>
              <span className={`dot ${prep.ready ? 'ok' : ''}`} />
              <span>{prep.ready ? 'Interview ready' : `Preparing your interview… ${pct}%`}</span>
            </div>
            <div className="bar"><span style={{ width: `${pct}%` }} /></div>
          </div>

          {msg && <p style={{ color: 'var(--bad)' }}>{msg}</p>}
          <button className="primary" style={{ marginTop: '1rem' }} onClick={start} disabled={!canStart}>
            Start interview
          </button>
        </div>
      </div>
    </div>
  )
}
