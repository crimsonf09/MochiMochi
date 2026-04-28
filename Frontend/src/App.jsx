import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { fetchHistory, fetchMemory, fetchState, openChatWs } from './lib/api'

function App() {
  const [username, setUsername] = useState(() => localStorage.getItem('username') || '')
  const [draftUsername, setDraftUsername] = useState(() => localStorage.getItem('username') || '')
  const [messages, setMessages] = useState([])
  const [composer, setComposer] = useState('')
  const [connStatus, setConnStatus] = useState('disconnected') // disconnected | connecting | connected
  const [affection, setAffection] = useState(0)
  const [error, setError] = useState('')
  const [syncingHistory, setSyncingHistory] = useState(false)
  const [lastHistorySyncAt, setLastHistorySyncAt] = useState(null)
  const [memory, setMemory] = useState(null)
  const [memoryLoading, setMemoryLoading] = useState(false)
  const [memoryError, setMemoryError] = useState('')
  const [memoryNextRefreshIn, setMemoryNextRefreshIn] = useState(3)
  const [memoryLastRefreshedAt, setMemoryLastRefreshedAt] = useState(null)
  /** Poll /api memory while chatting (Start / Stop). */
  const [memoryMonitorLive, setMemoryMonitorLive] = useState(false)
  /** When false, WebSocket sends memory_enabled: false (no retrieval writes / long-term updates). */
  const [longTermMemoryEnabled, setLongTermMemoryEnabled] = useState(true)

  const wsRef = useRef(null)
  const listRef = useRef(null)
  const memoryMonitorLiveRef = useRef(false)
  const reconnectTimerRef = useRef(null)
  const historySyncLockRef = useRef(false)

  const isLoggedIn = useMemo(() => username.trim().length > 0, [username])

  function normalizeHistory(rawHistory) {
    return [...rawHistory]
      .sort((a, b) => {
        const ta = a?.timestamp ? new Date(a.timestamp).getTime() : 0
        const tb = b?.timestamp ? new Date(b.timestamp).getTime() : 0
        if (ta !== tb) return ta - tb
        return String(a?.id ?? '').localeCompare(String(b?.id ?? ''))
      })
      .map((m) => ({
        id: m.id,
        role: m.role,
        text: m.message,
        emotionScore: m.emotion_score != null ? Number(m.emotion_score) : 0,
        weightedScore: m.weighted_score || null,
        emotionLabel: m.emotion_label || '',
        emotion3d: m.emotion_3d || null,
        timestamp: m.timestamp,
      }))
  }

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
    // emotion_score from API is [-10, 10] everywhere
    setMessages(normalizeHistory(h))
    // State returns emotion_score [-10, 10]
    setAffection(s.emotion_score != null ? s.emotion_score : 0)
    setMemory(m || {
      identity_facts: [],
      episodic_memories: [],
      semantic_profile: null,
      working_memory_count: 0
    })
  }

  const refreshMemory = useCallback(async () => {
    if (!username) return
    setMemoryLoading(true)
    setMemoryError('')
    try {
      const m = await fetchMemory(username)
      setMemory(m || {
        identity_facts: [],
        episodic_memories: [],
        semantic_profile: null,
        working_memory_count: 0
      })
      setMemoryLastRefreshedAt(new Date())
      setMemoryNextRefreshIn(3)
    } catch (e) {
      console.error('[Frontend] Failed to refresh memory:', e)
      setMemoryError(e?.message || 'Memory refresh failed')
    } finally {
      setMemoryLoading(false)
    }
  }, [username])

  useEffect(() => {
    memoryMonitorLiveRef.current = memoryMonitorLive
  }, [memoryMonitorLive])

  useEffect(() => {
    if (!isLoggedIn || !username.trim() || !memoryMonitorLive) return undefined
    setMemoryNextRefreshIn(3)
    refreshMemory()
    const id = setInterval(() => refreshMemory(), 3000)
    return () => clearInterval(id)
  }, [isLoggedIn, username, memoryMonitorLive, refreshMemory])

  useEffect(() => {
    if (!isLoggedIn || !memoryMonitorLive) return undefined
    const id = setInterval(() => {
      setMemoryNextRefreshIn((prev) => (prev <= 1 ? 3 : prev - 1))
    }, 1000)
    return () => clearInterval(id)
  }, [isLoggedIn, memoryMonitorLive])

  useEffect(() => {
    if (!isLoggedIn || !username.trim()) return undefined
    syncHistoryFromBackend()
    const id = setInterval(() => syncHistoryFromBackend(), 3000)
    return () => clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoggedIn, username])

  function startMemoryMonitor() {
    setMemoryMonitorLive(true)
    setMemoryNextRefreshIn(3)
  }

  function stopMemoryMonitor() {
    setMemoryMonitorLive(false)
  }

  const recentChatForMemoryPanel = useMemo(
    () => messages.slice(-40),
    [messages],
  )

  async function syncHistoryFromBackend() {
    if (!username || historySyncLockRef.current) return
    historySyncLockRef.current = true
    setSyncingHistory(true)
    try {
      const [h, s] = await Promise.all([fetchHistory(username), fetchState(username)])
      setMessages(normalizeHistory(h))
      setAffection(s.emotion_score != null ? s.emotion_score : 0)
      setLastHistorySyncAt(new Date())
    } catch (e) {
      console.error('[Frontend] Failed to sync history/state:', e)
    } finally {
      setSyncingHistory(false)
      historySyncLockRef.current = false
    }
  }

  function disconnectWs() {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
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

    ws.onopen = () => {
      setConnStatus('connected')
      setError('')
    }
    ws.onclose = () => {
      setConnStatus('disconnected')
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current)
      }
      reconnectTimerRef.current = setTimeout(() => {
        connectWs(u)
      }, 1500)
    }
    ws.onerror = () => setError('WebSocket error (auto reconnecting)...')
    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data)
        if (data?.error) {
          setError(String(data.error))
          return
        }
        // Keep backend history as source-of-truth, so Mochi matches Dmind history exactly.
        const score = data.emotion_score != null ? Number(data.emotion_score) : 0
        setAffection(score)
        syncHistoryFromBackend()
        if (memoryMonitorLiveRef.current) refreshMemory()
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
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }
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
    setMemoryMonitorLive(false)
    memoryMonitorLiveRef.current = false
  }

  function sendMessage() {
    const text = composer.trim()
    if (!text) return
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Not connected. Start the backend and refresh.')
      return
    }

    setError('')
    setComposer('')
    wsRef.current.send(
      JSON.stringify({ message: text, memory_enabled: longTermMemoryEnabled }),
    )
    setTimeout(() => syncHistoryFromBackend(), 300)
    if (memoryMonitorLive) {
      setTimeout(() => refreshMemory(), 1000)
    }
  }

  return (
    <div className="app">
      <div className="panel chatPanel">
        <div className="topbar">
          <div className="brand">
            <div className="brandTitle">Mochi Chat (LangGraph + MongoDB)</div>
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
              <button className="btnGhost" onClick={syncHistoryFromBackend} disabled={syncingHistory}>
                {syncingHistory ? 'Syncing...' : 'Sync from Dmind'}
              </button>
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
                    </div>
                  </div>
                ))}
              </div>

              {error && <div className="hint">Error: {error}</div>}
              {isLoggedIn && lastHistorySyncAt && (
                <div className="hint">Last synced: {lastHistorySyncAt.toLocaleTimeString()}</div>
              )}

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
            <button type="button" className="btnGhost btnSmall" onClick={refreshMemory} disabled={memoryLoading}>
              {memoryLoading ? '...' : '↻'}
            </button>
          </div>

          <div className="memoryToolbar">
            <div className="memoryToolbarLabel">Live memory panel</div>
            <div className="memoryToolbarRow">
              <button
                type="button"
                className="btnStart"
                onClick={startMemoryMonitor}
                disabled={memoryMonitorLive}
              >
                Start
              </button>
              <button
                type="button"
                className="btnStop"
                onClick={stopMemoryMonitor}
                disabled={!memoryMonitorLive}
              >
                Stop
              </button>
              <span className={`memoryLiveBadge ${memoryMonitorLive ? 'on' : 'off'}`}>
                {memoryMonitorLive ? `Live · refresh in ${memoryNextRefreshIn}s` : 'Paused'}
              </span>
            </div>
            {memoryMonitorLive && (
              <div className="hint" style={{ padding: 0, margin: 0 }}>
                {memoryLastRefreshedAt
                  ? `Last memory sync: ${memoryLastRefreshedAt.toLocaleTimeString()}`
                  : 'Waiting first sync...'}
              </div>
            )}
            {memoryError && (
              <div className="hint" style={{ padding: 0, margin: 0, color: '#ff6b6b' }}>
                Memory sync error: {memoryError}
              </div>
            )}
            <label className="memoryToggle">
              <input
                type="checkbox"
                checked={longTermMemoryEnabled}
                onChange={(e) => setLongTermMemoryEnabled(e.target.checked)}
              />
              <span>Long-term memory (retrieve + learn from chat)</span>
            </label>
            {!longTermMemoryEnabled && (
              <div className="hint" style={{ padding: 0, margin: 0 }}>
                Off: replies use only recent chat in DB; identity / episodic / profile are skipped for this turn.
              </div>
            )}
          </div>

          <div className="memoryBody">
            <div className="memorySection">
              <h4>Current chat (UI)</h4>
              {recentChatForMemoryPanel.length === 0 ? (
                <div className="memoryEmpty">No messages in this session yet</div>
              ) : (
                <div className="memoryCurrentChat">
                  {recentChatForMemoryPanel.map((m) => (
                    <div
                      key={m.id}
                      className={`memoryTurn ${m.role === 'user' ? 'memoryTurnUser' : 'memoryTurnAi'}`}
                    >
                      <div className="memoryTurnRole">{m.role === 'user' ? 'You' : 'Mochi'}</div>
                      {m.text}
                    </div>
                  ))}
                </div>
              )}
            </div>

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
              <h4>Working Memory (backend)</h4>
              <div className="memoryItem">
                <div className="memoryItemValue">Last {memory?.working_memory_count || 0} conversation turns in graph context</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
