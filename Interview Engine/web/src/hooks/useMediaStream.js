import { useState, useEffect, useRef } from 'react'

// Camera + mic. AEC/noise-suppression on the audio track so the
// interviewer's own voice doesn't leak into the mic (requirements §11).
export function useMediaStream() {
  const [stream, setStream] = useState(null)
  const [error, setError] = useState(null)
  const [hasPermission, setHasPermission] = useState(false)
  const videoRef = useRef(null)

  useEffect(() => {
    let active = null
    ;(async () => {
      try {
        const s = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        })
        active = s
        setStream(s)
        setHasPermission(true)
        if (videoRef.current) videoRef.current.srcObject = s
      } catch (err) {
        setError(err.message || 'Permission denied')
        setHasPermission(false)
      }
    })()
    return () => active?.getTracks().forEach((t) => t.stop())
  }, [])

  useEffect(() => {
    if (videoRef.current && stream && videoRef.current.srcObject !== stream) {
      videoRef.current.srcObject = stream
    }
  }, [stream])

  return { videoRef, stream, hasPermission, error }
}
