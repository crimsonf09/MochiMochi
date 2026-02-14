# LangGraph Flow Diagram

## Complete Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    WebSocket Message Received                    │
│              { "message": "Hello, how are you?" }               │
└────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              graph.ainvoke() called with:                        │
│  {                                                               │
│    "username": "touch",                                          │
│    "user_message": "Hello, how are you?"                        │
│  }                                                               │
└────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ ENTRY POINT     │
                    │   "load"        │  ← Line 204: g.set_entry_point("load")
                    └────────┬────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  NODE 1: load                                                   │
│  ─────────────────────────────────────────────────────────────  │
│  Function: _load_history_and_state()                           │
│                                                                  │
│  Actions:                                                       │
│  1. Query MongoDB for last 10 messages (username)               │
│  2. Get latest AI message to find prev_score                   │
│                                                                  │
│  Returns to State:                                              │
│  {                                                               │
│    "history": [                                                 │
│      {"role": "user", "message": "..."},                       │
│      {"role": "ai", "message": "..."},                         │
│      ... (last 10 messages)                                     │
│    ],                                                            │
│    "prev_score": 2  // Current affection score                  │
│  }                                                               │
└────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   EDGE          │
                    │ load → sentiment│  ← Line 205: g.add_edge("load", "sentiment")
                    └────────┬────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  NODE 2: sentiment                                              │
│  ─────────────────────────────────────────────────────────────  │
│  Function: _analyze_sentiment()                                │
│                                                                  │
│  Actions:                                                       │
│  1. Analyze user_message for sentiment (deterministic)          │
│     - Kind words (thank, please) → +1                          │
│     - Rude words (stupid, hate) → -1                           │
│  2. Calculate new_score = prev_score + delta                   │
│  3. Determine emotion_label based on new_score:                │
│     ≤ -3: "Hostile Tsundere"                                    │
│     -2 to 1: "Cold / Defensive"                                 │
│     2 to 4: "Soft Tsundere"                                    │
│     ≥ 5: "Dere Mode"                                            │
│                                                                  │
│  Returns to State:                                              │
│  {                                                               │
│    "delta": 1,          // +1 for kind message                 │
│    "new_score": 3,      // 2 + 1 = 3                           │
│    "emotion_label": "Soft Tsundere"                            │
│  }                                                               │
└────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   EDGE          │
                    │ sentiment → respond│  ← Line 206: g.add_edge("sentiment", "respond")
                    └────────┬────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  NODE 3: respond                                               │
│  ─────────────────────────────────────────────────────────────  │
│  Function: _generate_response()                                │
│                                                                  │
│  Actions:                                                       │
│                                                                  │
│  IF OPENAI_API_KEY is set:                                      │
│    ┌──────────────────────────────────────────────────────┐   │
│    │ 1. Call _messages_for_llm() to format messages:     │   │
│    │    [                                                   │   │
│    │      {                                                 │   │
│    │        "role": "system",                              │   │
│    │        "content": "You are an AI chatbot with..."     │   │
│    │            + "Current persona stage: Soft Tsundere"   │   │
│    │            + "Current affection score: 3"             │   │
│    │      },                                                │   │
│    │      ... (history messages),                          │   │
│    │      {                                                 │   │
│    │        "role": "user",                                │   │
│    │        "content": "Hello, how are you?"               │   │
│    │      }                                                 │   │
│    │    ]                                                   │   │
│    │                                                        │   │
│    │ 2. Call OpenAI API:                                   │   │
│    │    client.chat.completions.create(                    │   │
│    │      model="gpt-4o-mini",                             │   │
│    │      messages=[...],                                   │   │
│    │      temperature=0.8                                  │   │
│    │    )                                                   │   │
│    │                                                        │   │
│    │ 3. Extract AI response from completion                │   │
│    └──────────────────────────────────────────────────────┘   │
│                                                                  │
│  ELSE (no API key):                                             │
│    ┌──────────────────────────────────────────────────────┐   │
│    │ Use fallback_tsundere_response()                     │   │
│    │ Returns deterministic response based on stage        │   │
│    └──────────────────────────────────────────────────────┘   │
│                                                                  │
│  Returns to State:                                              │
│  {                                                               │
│    "ai_message": "O-okay... I can help. Don't misunderstand..." │
│  }                                                               │
└────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   EDGE          │
                    │ respond → persist│  ← Line 207: g.add_edge("respond", "persist")
                    └────────┬────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  NODE 4: persist                                               │
│  ─────────────────────────────────────────────────────────────  │
│  Function: _persist()                                         │
│                                                                  │
│  Actions:                                                       │
│  1. Insert user message to MongoDB:                           │
│     {                                                           │
│       "username": "touch",                                     │
│       "role": "user",                                          │
│       "message": "Hello, how are you?",                       │
│       "emotion_score": 2,  // prev_score                        │
│       "emotion_label": "Cold / Defensive",                    │
│       "timestamp": "2024-03-05T..."                            │
│     }                                                           │
│                                                                  │
│  2. Insert AI message to MongoDB:                              │
│     {                                                           │
│       "username": "touch",                                     │
│       "role": "ai",                                            │
│       "message": "O-okay... I can help...",                   │
│       "emotion_score": 3,  // new_score                        │
│       "emotion_label": "Soft Tsundere",                        │
│       "timestamp": "2024-03-05T..."                           │
│     }                                                           │
│                                                                  │
│  Returns to State:                                              │
│  {                                                               │
│    "timestamp": datetime(...)                                  │
│  }                                                               │
└────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ FINISH POINT    │
                    │   "persist"     │  ← Line 208: g.set_finish_point("persist")
                    └────────┬────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Return Final State                            │
│  {                                                               │
│    "username": "touch",                                         │
│    "user_message": "Hello, how are you?",                      │
│    "history": [...],                                            │
│    "prev_score": 2,                                             │
│    "delta": 1,                                                  │
│    "new_score": 3,                                              │
│    "emotion_label": "Soft Tsundere",                           │
│    "ai_message": "O-okay... I can help...",                    │
│    "timestamp": datetime(...)                                  │
│  }                                                               │
└────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              WebSocket Response Sent                             │
│  {                                                               │
│    "message": "O-okay... I can help...",                       │
│    "emotion_score": 3,                                          │
│    "emotion_label": "Soft Tsundere",                           │
│    "timestamp": "2024-03-05T..."                                │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

## State Flow Through Nodes

### Initial State (Input)
```python
{
    "username": "touch",
    "user_message": "Hello, how are you?"
}
```

### After Node 1: load
```python
{
    "username": "touch",
    "user_message": "Hello, how are you?",
    "history": [...],  # Last 10 messages
    "prev_score": 2    # From latest AI message
}
```

### After Node 2: sentiment
```python
{
    "username": "touch",
    "user_message": "Hello, how are you?",
    "history": [...],
    "prev_score": 2,
    "delta": 1,              # +1 for kind message
    "new_score": 3,           # 2 + 1
    "emotion_label": "Soft Tsundere"
}
```

### After Node 3: respond
```python
{
    "username": "touch",
    "user_message": "Hello, how are you?",
    "history": [...],
    "prev_score": 2,
    "delta": 1,
    "new_score": 3,
    "emotion_label": "Soft Tsundere",
    "ai_message": "O-okay... I can help. Don't misunderstand, though."
}
```

### After Node 4: persist (Final)
```python
{
    "username": "touch",
    "user_message": "Hello, how are you?",
    "history": [...],
    "prev_score": 2,
    "delta": 1,
    "new_score": 3,
    "emotion_label": "Soft Tsundere",
    "ai_message": "O-okay... I can help. Don't misunderstand, though.",
    "timestamp": datetime(2024, 3, 5, ...)
}
```

## Key Points

1. **Entry Point (Line 204):** `g.set_entry_point("load")`
   - Graph execution always starts at the "load" node

2. **Sequential Edges (Lines 205-207):**
   - `load → sentiment → respond → persist`
   - Each node runs in order, passing state forward

3. **Finish Point (Line 208):** `g.set_finish_point("persist")`
   - Graph execution completes after "persist" node
   - Final state is returned to caller

4. **GPT Integration:**
   - Happens in the "respond" node (line 193-194)
   - Only if `OPENAI_API_KEY` is configured
   - Falls back to deterministic responder if GPT fails

5. **State Persistence:**
   - State flows through all nodes
   - Each node adds/updates state fields
   - Final state includes all accumulated data

## Execution Context

The graph is invoked from `main.py` WebSocket handler:

```python
# Line 109 in main.py
result = await graph.ainvoke({
    "username": username,
    "user_message": client_msg.message
})
```

This triggers the entire LangGraph flow from entry point to finish point.
