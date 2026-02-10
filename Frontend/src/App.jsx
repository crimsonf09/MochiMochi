import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { fetchHistory, fetchState, openChatWs } from './lib/api'

function App() {
  const [username, setUsername] = useState(() => localStorage.getItem('username') || '')
  const [draftUsername, setDraftUsername] = useState(() => localStorage.getItem('username') || '')
  const [messages, setMessages] = useState([])
  const [composer, setComposer] = useState('')
  const [connStatus, setConnStatus] = useState('disconnected') // disconnected | connecting | connected
  const [affection, setAffection] = useState(0)
  const [stage, setStage] = useState('Cold / Defensive')
  const [error, setError] = useState('')

  const wsRef = useRef(null)
  const listRef = useRef(null)

  const isLoggedIn = useMemo(() => username.trim().length > 0, [username])

  function scrollToBottom() {
    const el = listRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages.length])

  async function loadUserData(u) {
    setError('')
    const [h, s] = await Promise.all([fetchHistory(u), fetchState(u)])
    setMessages(
      h.map((m) => ({
        id: m.id,
        role: m.role,
        text: m.message,
        emotionScore: m.emotion_score,
        emotionLabel: m.emotion_label,
        timestamp: m.timestamp,
      })),
    )
    setAffection(s.affection_score)
    setStage(s.persona_stage)
  }

  function disconnectWs() {
    if (wsRef.current) {
      try {
        wsRef.current.close()
      } catch {
        // ignore
      }
      wsRef.current = null
    }
    setConnStatus('disconnected')
  }

  function connectWs(u) {
    disconnectWs()
    setConnStatus('connecting')
    setError('')

    const ws = openChatWs(u)
    wsRef.current = ws

    ws.onopen = () => setConnStatus('connected')
    ws.onclose = () => setConnStatus('disconnected')
    ws.onerror = () => setError('WebSocket error (backend down or proxy misconfigured).')
    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data)
        if (data?.error) {
          setError(String(data.error))
          return
        }
        setMessages((prev) => [
          ...prev,
          {
            id: `ai-${Date.now()}`,
            role: 'ai',
            text: data.message,
            emotionScore: data.emotion_score,
            emotionLabel: data.emotion_label,
            timestamp: data.timestamp,
          },
        ])
        setAffection(data.emotion_score)
        setStage(data.emotion_label)
      } catch {
        setError('Bad message from server.')
      }
    }
  }

  useEffect(() => {
    if (!isLoggedIn) return
    let cancelled = false
    ;(async () => {
      try {
        await loadUserData(username)
        if (!cancelled) connectWs(username)
      } catch (e) {
        setError(e?.message || 'Failed to load history/state.')
      }
    })()
    return () => {
      cancelled = true
      disconnectWs()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [username])

  async function onLogin(e) {
    e.preventDefault()
    const u = draftUsername.trim()
    if (!u) return
    localStorage.setItem('username', u)
    setUsername(u)
  }

  function onLogout() {
    disconnectWs()
    localStorage.removeItem('username')
    setUsername('')
    setDraftUsername('')
    setMessages([])
    setComposer('')
    setError('')
    setAffection(0)
    setStage('Cold / Defensive')
  }

  function sendMessage() {
    const text = composer.trim()
    if (!text) return
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Not connected. Start the backend and refresh.')
      return
    }

    setError('')
    setMessages((prev) => [
      ...prev,
      {
        id: `user-${Date.now()}`,
        role: 'user',
        text,
        emotionScore: affection,
        emotionLabel: stage,
        timestamp: new Date().toISOString(),
      },
    ])
    setComposer('')
    wsRef.current.send(JSON.stringify({ message: text }))
  }

  return (
    <div className="app">
      <div className="panel">
        <div className="topbar">
          <div className="brand">
            <div className="brandTitle">Tsundere Chat (LangGraph + MongoDB)</div>
            <div className="brandMeta">
              {isLoggedIn ? (
                <>
                  User: <b>{username}</b> · Connection: <b>{connStatus}</b>
                </>
              ) : (
                'Enter a username to reconnect to your history.'
              )}
            </div>
          </div>
          {isLoggedIn && (
            <div className="statusPills">
              <span className="pill pillStrong">Affection: {affection}</span>
              <span className="pill">Stage: {stage}</span>
              <button className="btnGhost" onClick={onLogout}>
                Switch user
              </button>
            </div>
          )}
        </div>

        <div className="content">
          {!isLoggedIn ? (
            <div className="login">
              <form className="loginCard" onSubmit={onLogin}>
                <h1>Login (no password)</h1>
                <p>
                  Your <b>username</b> is your identity. Use the same one to restore chat history + affection score.
                </p>
                <div className="row">
                  <input
                    className="input"
                    value={draftUsername}
                    onChange={(e) => setDraftUsername(e.target.value)}
                    placeholder="e.g. natou"
                    autoFocus
                  />
                  <button className="btnPrimary" type="submit">
                    Start chat
                  </button>
                </div>
                {error && <div className="hint">Error: {error}</div>}
                <div className="hint">
                  Dev tip: start backend on <b>localhost:8000</b>. Frontend proxies <b>/chat</b> and <b>/ws</b>.
                </div>
              </form>
            </div>
          ) : (
            <div className="chat">
              <div className="messages" ref={listRef}>
                {messages.map((m) => (
                  <div
                    key={m.id}
                    className={`msgRow ${m.role === 'user' ? 'msgRowUser' : ''}`}
                    title={m.timestamp ? String(m.timestamp) : ''}
                  >
                    <div className={`bubble ${m.role === 'user' ? 'bubbleUser' : 'bubbleAi'}`}>
                      {m.text}
                      {m.role === 'ai' && (
                        <div className="bubbleMeta">
                          <span className="badge">{m.emotionLabel}</span>
                          <span className="badge">Affection: {m.emotionScore}</span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {error && <div className="hint">Error: {error}</div>}

              <div className="composer">
                <input
                  className="input"
                  value={composer}
                  onChange={(e) => setComposer(e.target.value)}
                  placeholder="Type a message…"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      sendMessage()
                    }
                  }}
                />
                <button className="btnPrimary sendBtn" onClick={sendMessage}>
                  Send
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
