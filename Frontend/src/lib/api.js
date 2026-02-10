// Get API base, removing trailing slash if present
const API_BASE_RAW = import.meta.env.VITE_API_BASE || ''
const API_BASE = API_BASE_RAW.endsWith('/') ? API_BASE_RAW.slice(0, -1) : API_BASE_RAW

function toWsBase() {
  const env = import.meta.env.VITE_WS_BASE
  if (env) {
    // Remove trailing slash if present
    return env.endsWith('/') ? env.slice(0, -1) : env
  }
  // Default: same-origin WS (works with Vite proxy)
  const { protocol, host } = window.location
  const wsProto = protocol === 'https:' ? 'wss:' : 'ws:'
  return `${wsProto}//${host}`
}

export async function fetchHistory(username) {
  // Use relative URL if API_BASE is empty (goes through Vite proxy)
  const url = API_BASE ? `${API_BASE}/chat/history/${encodeURIComponent(username)}` : `/chat/history/${encodeURIComponent(username)}`
  const r = await fetch(url)
  if (!r.ok) throw new Error(`History fetch failed: ${r.status}`)
  return await r.json()
}

export async function fetchState(username) {
  // Use relative URL if API_BASE is empty (goes through Vite proxy)
  const url = API_BASE ? `${API_BASE}/chat/state/${encodeURIComponent(username)}` : `/chat/state/${encodeURIComponent(username)}`
  const r = await fetch(url)
  if (!r.ok) throw new Error(`State fetch failed: ${r.status}`)
  return await r.json()
}

export function openChatWs(username) {
  const wsBase = toWsBase()
  const url = `${wsBase}/ws/${encodeURIComponent(username)}`
  return new WebSocket(url)
}

