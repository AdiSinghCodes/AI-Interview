import { useState, useRef, useCallback, useEffect } from 'react'
import { liveSocketUrl } from '../api'

const SEND_SR = 16000
const RECV_SR = 24000
const FRAME = 512

/**
 * The live interview. Must be started from a user gesture (`begin()`) so the
 * browser lets audio play. Streams the mic to the engine as 16 kHz PCM, plays
 * the interviewer's 24 kHz audio back gap-free, and tracks the turn phase.
 *
 * phase: idle | connecting | speaking | thinking | listening | ended | error
 */
export function useInterviewSession(roundId, stream) {
  const [phase, setPhase] = useState('idle')
  const [turns, setTurns] = useState([])
  const [partial, setPartial] = useState('')
  const [amplitude, setAmplitude] = useState(0)

  const wsRef = useRef(null)
  const sendCtxRef = useRef(null)
  const playCtxRef = useRef(null)
  const analyserRef = useRef(null)
  const nodesRef = useRef({})
  const playHeadRef = useRef(0)
  const pendingRef = useRef(0)
  const speakingDoneRef = useRef(true)
  const endedRef = useRef(false)
  const rafRef = useRef(null)
  const tailRef = useRef(new Float32Array(0))
  const startedRef = useRef(false)

  const pushTurn = (t) => setTurns((prev) => [...prev, t])

  const toListening = useCallback(() => {
    if (!endedRef.current && pendingRef.current <= 0 && speakingDoneRef.current) {
      setPhase('listening')
    }
  }, [])

  const playPCM = useCallback((arrayBuffer) => {
    const ctx = playCtxRef.current
    const f32 = new Float32Array(arrayBuffer)
    if (!ctx || !f32.length) return
    const buf = ctx.createBuffer(1, f32.length, RECV_SR)
    buf.copyToChannel(f32, 0)
    const node = ctx.createBufferSource()
    node.buffer = buf
    node.connect(analyserRef.current)
    playHeadRef.current = Math.max(playHeadRef.current, ctx.currentTime + 0.02)
    node.start(playHeadRef.current)
    playHeadRef.current += buf.duration
    pendingRef.current += 1
    setPhase('speaking')
    node.onended = () => {
      pendingRef.current -= 1
      toListening()
    }
  }, [toListening])

  const cleanup = useCallback(() => {
    cancelAnimationFrame(rafRef.current)
    const { proc, micSrc } = nodesRef.current
    try { proc?.disconnect(); micSrc?.disconnect() } catch { /* noop */ }
    sendCtxRef.current?.close().catch(() => {})
    playCtxRef.current?.close().catch(() => {})
    const ws = wsRef.current
    if (ws && ws.readyState <= 1) ws.close()
  }, [])

  const begin = useCallback(async () => {
    if (startedRef.current || !roundId || !stream) return
    startedRef.current = true
    endedRef.current = false
    setPhase('connecting')

    // outbound: mic -> downsample -> 512-sample frames
    const sendCtx = new AudioContext()
    sendCtxRef.current = sendCtx
    if (sendCtx.state === 'suspended') await sendCtx.resume()
    const micSrc = sendCtx.createMediaStreamSource(stream)
    const proc = sendCtx.createScriptProcessor(4096, 1, 1)
    proc.onaudioprocess = (e) => {
      const ws = wsRef.current
      if (!ws || ws.readyState !== 1) return
      const input = e.inputBuffer.getChannelData(0)
      const ratio = sendCtx.sampleRate / SEND_SR
      const outLen = Math.floor(input.length / ratio)
      const down = new Float32Array(outLen)
      for (let i = 0; i < outLen; i++) {
        const pos = i * ratio
        const j = Math.floor(pos)
        down[i] = (input[j] || 0) * (1 - (pos - j)) + (input[j + 1] || 0) * (pos - j)
      }
      const merged = new Float32Array(tailRef.current.length + down.length)
      merged.set(tailRef.current)
      merged.set(down, tailRef.current.length)
      let off = 0
      for (; off + FRAME <= merged.length; off += FRAME) ws.send(merged.slice(off, off + FRAME).buffer)
      tailRef.current = merged.slice(off)
    }
    micSrc.connect(proc)
    proc.connect(sendCtx.destination)
    nodesRef.current = { proc, micSrc }

    // inbound: playback + amplitude for the avatar
    const playCtx = new AudioContext({ sampleRate: RECV_SR })
    playCtxRef.current = playCtx
    if (playCtx.state === 'suspended') await playCtx.resume()
    const analyser = playCtx.createAnalyser()
    analyser.fftSize = 256
    analyser.connect(playCtx.destination)
    analyserRef.current = analyser

    const tick = () => {
      const a = analyserRef.current
      if (a) {
        const data = new Uint8Array(a.frequencyBinCount)
        a.getByteFrequencyData(data)
        let sum = 0
        for (let i = 0; i < data.length; i++) sum += data[i]
        setAmplitude(Math.min(1, sum / data.length / 85))
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)

    const ws = new WebSocket(liveSocketUrl(roundId))
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws
    ws.onopen = () => ws.send(new Float32Array(FRAME).buffer)
    ws.onclose = () => { if (!endedRef.current) setPhase('ended') }
    ws.onerror = () => setPhase('error')
    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) return playPCM(ev.data)
      const m = JSON.parse(ev.data)
      if (m.type === 'opening') { pushTurn({ role: 'interviewer', text: m.text, kind: 'opening' }); setPhase('speaking') }
      else if (m.type === 'partial') setPartial(m.text)
      else if (m.type === 'transcript') { setPartial(''); if (m.text) pushTurn({ role: 'candidate', text: m.text }); setPhase('thinking') }
      else if (m.type === 'speaking_start') { speakingDoneRef.current = false; playHeadRef.current = 0; setPhase('speaking') }
      else if (m.type === 'speaking_end') { speakingDoneRef.current = true; toListening() }
      else if (m.type === 'interviewer') pushTurn({ role: 'interviewer', text: m.text, kind: m.kind })
      else if (m.type === 'end') { endedRef.current = true; setPhase('ended') }
    }
  }, [roundId, stream, playPCM, toListening])

  const end = useCallback(() => {
    endedRef.current = true
    const ws = wsRef.current
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'end_round' }))
    setPhase('ended')
    cleanup()
  }, [cleanup])

  useEffect(() => cleanup, [cleanup])

  return { phase, turns, partial, amplitude, begin, end }
}
