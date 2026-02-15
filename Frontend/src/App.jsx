import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { fetchHistory, fetchMemory, fetchState, openChatWs } from './lib/api'

function App() {
  const [username, setUsername] = useState(() => localStorage.getItem('username') || '')
  const [draftUsername, setDraftUsername] = useState(() => localStorage.getItem('username') || '')
  const [messages, setMessages] = useState([])
  const [composer, setComposer] = useState('')
  const [connStatus, setConnStatus] = useState('disconnected') // disconnected | connecting | connected
  const [affection, setAffection] = useState(0)
  const [stage, setStage] = useState('Cold / Defensive')
  const [error, setError] = useState('')
  const [memory, setMemory] = useState(null)
  const [memoryLoading, setMemoryLoading] = useState(false)

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
    const [h, s, m] = await Promise.all([
      fetchHistory(u), 
      fetchState(u), 
      fetchMemory(u).catch((e) => {
        console.error('Failed to load memory:', e)
        return {
          identity_facts: [],
          episodic_memories: [],
          semantic_profile: null,
          working_memory_count: 0
        }
      })
    ])
    setMessages(
      h.map((m) => ({
        id: m.id,
        role: m.role,
        text: m.message,
        emotionScore: m.emotion_score,
        emotionLabel: m.emotion_label,
        emotion3d: m.emotion_3d || null,
        timestamp: m.timestamp,
      })),
    )
    setAffection(s.affection_score)
    setStage(s.persona_stage)
    setMemory(m || {
      identity_facts: [],
      episodic_memories: [],
      semantic_profile: null,
      working_memory_count: 0
    })
  }

  async function refreshMemory() {
    if (!username) return
    setMemoryLoading(true)
    try {
      console.log('[Frontend] Fetching memory for:', username)
      const m = await fetchMemory(username)
      console.log('[Frontend] Memory received:', {
        facts: m?.identity_facts?.length || 0,
        episodic: m?.episodic_memories?.length || 0,
        working: m?.working_memory_count || 0,
        hasProfile: !!m?.semantic_profile
      })
      setMemory(m || {
        identity_facts: [],
        episodic_memories: [],
        semantic_profile: null,
        working_memory_count: 0
      })
    } catch (e) {
      console.error('[Frontend] Failed to refresh memory:', e)
      setMemory({
        identity_facts: [],
        episodic_memories: [],
        semantic_profile: null,
        working_memory_count: 0
      })
    } finally {
      setMemoryLoading(false)
    }
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
            emotion3d: data.emotion_3d || null,
            timestamp: data.timestamp,
          },
        ])
        setAffection(data.emotion_score)
        setStage(data.emotion_label)
        // Refresh memory after AI response
        refreshMemory()
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
        emotion3d: null, // Will be updated when message is processed
        timestamp: new Date().toISOString(),
      },
    ])
    setComposer('')
    wsRef.current.send(JSON.stringify({ message: text }))
    // Refresh memory after sending message
    setTimeout(() => refreshMemory(), 1000)
  }

  return (
    <div className="app">
      <div className="panel chatPanel">
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
                          {m.emotion3d && (
                            <div className="emotion3d">
                              <span className="emotion3dLabel">V: {m.emotion3d.valence.toFixed(2)}</span>
                              <span className="emotion3dLabel">A: {m.emotion3d.arousal.toFixed(2)}</span>
                              <span className="emotion3dLabel">D: {m.emotion3d.dominance.toFixed(2)}</span>
                            </div>
                          )}
                        </div>
                      )}
                      {m.role === 'user' && m.emotion3d && (
                        <div className="bubbleMeta">
                          <div className="emotion3d">
                            <span className="emotion3dLabel">V: {m.emotion3d.valence.toFixed(2)}</span>
                            <span className="emotion3dLabel">A: {m.emotion3d.arousal.toFixed(2)}</span>
                            <span className="emotion3dLabel">D: {m.emotion3d.dominance.toFixed(2)}</span>
                          </div>
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

      {/* Memory Sidebar */}
      {isLoggedIn && (
        <div className="memorySidebar">
          <div className="memoryHeader">
            <h3>Memory System</h3>
            <button className="btnGhost btnSmall" onClick={refreshMemory} disabled={memoryLoading}>
              {memoryLoading ? '...' : '↻'}
            </button>
          </div>

          <div className="memoryContent">
            {/* Identity Memory */}
            <div className="memorySection">
              <h4>Identity Facts</h4>
              {memory && memory.identity_facts && memory.identity_facts.length > 0 ? (
                <div className="memoryList">
                  {memory.identity_facts.map((fact, idx) => (
                    <div key={idx} className="memoryItem">
                      <div className="memoryItemKey">{fact.key}:</div>
                      <div className="memoryItemValue">{fact.value}</div>
                      <div className="memoryItemMeta">Confidence: {(fact.confidence * 100).toFixed(0)}%</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="memoryEmpty">No facts stored yet</div>
              )}
            </div>

            {/* Episodic Memory */}
            <div className="memorySection">
              <h4>Episodic Memories ({memory?.episodic_memories?.length || 0})</h4>
              {memory && memory.episodic_memories && memory.episodic_memories.length > 0 ? (
                <div className="memoryList">
                  {memory.episodic_memories.slice(0, 10).map((mem, idx) => (
                    <div key={idx} className="memoryItem memoryItemEpisodic">
                      <div className="memoryItemValue">{mem.event_summary}</div>
                      <div className="memoryItemMeta">
                        Importance: {(mem.importance_score * 100).toFixed(0)}% · 
                        Accessed: {mem.access_count}× · 
                        {new Date(mem.timestamp).toLocaleDateString()}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="memoryEmpty">No significant events yet</div>
              )}
            </div>

            {/* Semantic Profile */}
            <div className="memorySection">
              <h4>Semantic Profile</h4>
              {memory && memory.semantic_profile ? (
                <div className="memoryItem">
                  <div className="memoryItemValue">{memory.semantic_profile.personality_summary}</div>
                  {memory.semantic_profile.behavior_patterns && memory.semantic_profile.behavior_patterns.length > 0 && (
                    <div className="memoryItemMeta">
                      Patterns: {memory.semantic_profile.behavior_patterns.slice(0, 3).join(', ')}
                    </div>
                  )}
                </div>
              ) : (
                <div className="memoryEmpty">No profile generated yet</div>
              )}
            </div>

            {/* Working Memory */}
            <div className="memorySection">
              <h4>Working Memory</h4>
              <div className="memoryItem">
                <div className="memoryItemValue">Last {memory?.working_memory_count || 0} conversation turns</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
