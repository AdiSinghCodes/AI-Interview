import { useEffect, useRef } from 'react'

export default function TranscriptPanel({ turns, partial }) {
  const endRef = useRef(null)
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns, partial])

  return (
    <div className="transcript">
      {turns.map((t, i) => (
        <div key={i} className={`bubble ${t.role}`}>
          <span className="who">{t.role === 'interviewer' ? 'Interviewer' : 'You'}</span>
          <p>{t.text}</p>
        </div>
      ))}
      {partial && (
        <div className="bubble candidate pending">
          <span className="who">You</span>
          <p>{partial}…</p>
        </div>
      )}
      <div ref={endRef} />
    </div>
  )
}
