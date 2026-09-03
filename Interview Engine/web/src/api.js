// Same-origin: Vite proxies /v1 -> the engine on :8000 (see vite.config.js).
const base = '/v1'

async function json(res) {
  if (!res.ok) throw new Error((await res.text()) || res.statusText)
  return res.json()
}

export const api = {
  createSession: (body) =>
    fetch(`${base}/sessions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    }).then(json),

  session: (id) => fetch(`${base}/sessions/${id}`).then(json),

  sessionStatus: (id) => fetch(`${base}/sessions/${id}/status`).then(json),

  preflight: (result) =>
    fetch(`${base}/preflight`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(result),
    }).then(json),

  transcript: (roundId) => fetch(`${base}/rounds/${roundId}/transcript`).then(json),

  summary: (roundId) => fetch(`${base}/rounds/${roundId}/summary`, { method: 'POST' }).then(json),
}

export const liveSocketUrl = (roundId) => {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${location.host}${base}/rounds/${roundId}/live`
}
