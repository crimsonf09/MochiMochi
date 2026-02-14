# Memory System Implementation

## Overview

The chatbot now implements a comprehensive memory system that maintains long-term continuity, remembers user facts, recalls meaningful past events, and understands user behavior patterns **without sending full conversation history to the LLM**.

## System Architecture

### 1. Identity Memory (Structured Facts)

**Purpose:** Store stable user information (name, birthday, job, goals, etc.)

**Storage:** MongoDB collection `identity_memory`
- Key-value pairs with confidence scoring
- Updates when factual statements are detected
- Only relevant fields retrieved during prompt construction

**Example:**
```json
{
  "username": "touch",
  "key": "name",
  "value": "John",
  "confidence": 0.9,
  "updated_at": "2024-03-05T..."
}
```

**Detection:** Pattern matching for statements like:
- "My name is John"
- "I work as a developer"
- "My birthday is March 15"

### 2. Working Memory (Short-Term Context)

**Purpose:** Maintain last 6-10 conversation turns for immediate context

**Storage:** MongoDB collection `chat_messages` (existing)
- Automatically trimmed based on token limits
- If overflow occurs, older messages are summarized and moved to episodic memory
- Always injected into the prompt

**Token Management:**
- Default: Last 10 messages
- If exceeds 2000 tokens: Keep last 6, summarize older ones

### 3. Episodic Memory (Event-Based Long-Term)

**Purpose:** Store summarized important events (emotional or meaningful interactions)

**Storage:** MongoDB collection `episodic_memory`

**Each record contains:**
- `event_summary`: Text summary of the event
- `importance_score`: 0.0 to 1.0 (decays over time)
- `timestamp`: When the event occurred
- `access_count`: How many times retrieved
- `embedding`: Vector embedding for similarity search
- `metadata`: Additional context (emotion score, etc.)

**Retrieval Process:**
1. Generate embedding for current user message
2. Perform cosine similarity search against all episodic memories
3. Rank by: `(similarity * 0.6) + (importance * 0.4)`
4. Retrieve top K memories (default: 5)
5. Boost importance slightly when retrieved
6. Apply time-based decay periodically

**Time Decay:**
- Importance reduces by 1% per day
- Prevents old memories from dominating

### 4. Semantic Profile (Behavior Pattern Layer)

**Purpose:** Extract and store user behavior patterns and personality summary

**Storage:** MongoDB collection `semantic_profile`

**Contains:**
- `personality_summary`: High-level personality description
- `preferences`: User preferences and likes/dislikes
- `behavior_patterns`: Common interaction patterns
- `updated_at`: Last update timestamp

**Generation:**
- Periodically analyzes top episodic memories (by importance)
- Extracts patterns and themes
- Updates profile automatically

## Runtime Flow

```
1. Receive user message
   ↓
2. Load all memory systems:
   - Identity facts
   - Working memory (last 6-10 turns)
   - Semantic profile
   ↓
3. Classify message:
   - Fact → Extract and update Identity Memory
   - Significant event → Create Episodic Memory
   - Regular → Continue
   ↓
4. Generate embedding for current message
   ↓
5. Retrieve relevant episodic memories (similarity search)
   ↓
6. Build prompt with:
   - Character rules (tsundere persona)
   - Identity facts
   - Top episodic memories (3-5)
   - Semantic profile
   - Working memory (last 6 turns)
   - Current user message
   ↓
7. Generate response via GPT
   ↓
8. Update memory systems:
   - Store conversation in chat_messages
   - Update identity memory if facts detected
   - Create episodic memory if significant event
   - Update semantic profile periodically
```

## Database Collections

### `identity_memory`
```json
{
  "username": "string",
  "key": "string",  // "name", "birthday", "job", etc.
  "value": "string",
  "confidence": 0.0-1.0,
  "source_message": "string",
  "updated_at": "ISODate"
}
```

**Indexes:**
- `(username, key)` - Unique

### `episodic_memory`
```json
{
  "username": "string",
  "event_summary": "string",
  "importance_score": 0.0-1.0,
  "timestamp": "ISODate",
  "access_count": 0,
  "embedding": [float, ...],
  "metadata": {}
}
```

**Indexes:**
- `(username, timestamp)` - Descending
- `(username, importance_score)` - Descending

### `semantic_profile`
```json
{
  "username": "string",
  "profile": {
    "personality_summary": "string",
    "preferences": {},
    "behavior_patterns": [],
    "updated_at": "ISODate"
  },
  "updated_at": "ISODate"
}
```

**Indexes:**
- `(username)` - Unique

## Key Principles

✅ **Never send full history to the model**
- Only last 6-10 turns in working memory
- Episodic memories are summaries, not full conversations

✅ **Always summarize before storing long-term**
- Events are stored as summaries (200 chars)
- Older working memory is summarized before moving to episodic

✅ **Use vector similarity + importance scoring**
- Cosine similarity for relevance
- Importance score for significance
- Combined ranking: `(similarity * 0.6) + (importance * 0.4)`

✅ **Apply time decay to long-term memories**
- Importance reduces by 1% per day
- Prevents stale memories from dominating

✅ **Keep working memory small and clean**
- Maximum 10 turns
- Token-based trimming
- Automatic summarization of overflow

## Integration with LangGraph

The memory system is integrated into the LangGraph flow:

```
load → classify → sentiment → respond → persist
  │        │          │          │         │
  │        │          │          │         └─ Update all memory systems
  │        │          │          └─ Generate response with memory context
  │        │          └─ Calculate emotion update
  │        └─ Classify message & retrieve episodic memories
  └─ Load all memory systems
```

## Benefits

1. **Scalability:** Works across long conversations without token explosion
2. **Continuity:** Remembers user facts and past events
3. **Relevance:** Retrieves only relevant memories via similarity search
4. **Efficiency:** Summarized memories reduce token usage
5. **Adaptability:** Importance scoring and time decay keep memories fresh

## Usage Examples

### Identity Memory
User: "My name is Alice"
→ Stored: `{"key": "name", "value": "Alice", "confidence": 0.9}`
→ Future prompts include: "Known facts: name: Alice"

### Episodic Memory
User: "Thank you so much! That really helped me."
→ Classified as "event"
→ Stored as episodic memory with embedding
→ Retrieved when user mentions similar gratitude

### Semantic Profile
After multiple interactions:
→ Profile: "User is polite, asks technical questions, prefers detailed explanations"
→ Injected into system prompt for consistent personality understanding

## Configuration

No additional configuration needed. The system works automatically:
- Uses existing `OPENAI_API_KEY` for embeddings
- Stores in MongoDB (same database)
- Integrates seamlessly with existing emotion system

## Future Enhancements

- LLM-based summarization for episodic memories
- Automatic semantic profile regeneration
- Memory importance learning from user feedback
- Cross-user pattern recognition (if needed)
