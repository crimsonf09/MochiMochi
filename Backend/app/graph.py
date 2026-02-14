from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from langgraph.graph import StateGraph
from motor.motor_asyncio import AsyncIOMotorDatabase

from .classifier import classify_message, extract_facts_from_message
from .emotion import apply_emotion_update, emotion_label_for_score
from .memory import (
    create_episodic_memory,
    generate_embedding,
    get_identity_memory,
    get_semantic_profile,
    get_working_memory,
    search_episodic_memory,
    update_identity_memory,
)
from .tsundere import fallback_tsundere_response, persona_system_prompt


class GraphState(TypedDict, total=False):
    username: str
    user_message: str

    # Loaded from DB
    history: list[dict[str, Any]]  # last 10 messages, ascending
    prev_score: int

    # Memory System
    message_type: str  # "fact", "event", "regular"
    identity_facts: dict[str, Any]  # Retrieved identity memory
    working_memory: list[dict[str, Any]]  # Short-term context
    episodic_memories: list[Any]  # Retrieved episodic memories
    semantic_profile: dict[str, Any]  # Behavior patterns

    # Derived
    delta: int
    new_score: int
    emotion_label: str
    ai_message: str

    # Output
    timestamp: datetime


@dataclass(frozen=True)
class GraphDeps:
    db: AsyncIOMotorDatabase
    openai_api_key: str | None
    openai_model: str


async def _load_history_and_state(state: GraphState, deps: GraphDeps) -> GraphState:
    """Load all memory systems and conversation state"""
    username = state["username"]
    coll = deps.db["chat_messages"]

    # Working memory: last 6-10 turns
    working_memory = await get_working_memory(deps.db, username, max_turns=10)
    
    # Legacy history (for backward compatibility)
    history = working_memory

    # Current emotion state: latest AI message score, else 0
    latest_ai = await coll.find_one({"username": username, "role": "ai"}, sort=[("timestamp", -1)])
    prev_score = int(latest_ai["emotion_score"]) if latest_ai and "emotion_score" in latest_ai else 0

    # Load identity memory (all facts)
    identity_facts = await get_identity_memory(deps.db, username)
    identity_dict = {k: v.value for k, v in identity_facts.items()}

    # Load semantic profile
    semantic_profile = await get_semantic_profile(deps.db, username)

    return {
        "history": history,
        "working_memory": working_memory,
        "prev_score": prev_score,
        "identity_facts": identity_dict,
        "semantic_profile": semantic_profile,
    }


async def _classify_and_retrieve_memory(state: GraphState, deps: GraphDeps) -> GraphState:
    """Classify message and retrieve relevant episodic memories"""
    username = state["username"]
    user_message = state["user_message"]
    
    # Classify message
    msg_type, metadata = classify_message(user_message)
    print(f"[Memory] Classified message as: {msg_type}")
    
    # Generate embedding for episodic memory search
    query_embedding = None
    if deps.openai_api_key:
        try:
            query_embedding = await generate_embedding(
                user_message,
                deps.openai_api_key,
                "text-embedding-3-small"
            )
        except Exception as e:
            print(f"[Memory] Failed to generate query embedding: {e}")
    
    # Retrieve episodic memories if embedding available
    episodic_memories = []
    if query_embedding:
        episodic_memories = await search_episodic_memory(
            deps.db,
            username,
            query_embedding,
            top_k=5
        )
        print(f"[Memory] Retrieved {len(episodic_memories)} relevant episodic memories")
    else:
        print(f"[Memory] No embedding available, skipping episodic memory search")
    
    # Store episodic memory summaries
    episodic_summaries = [m.event_summary for m in episodic_memories] if episodic_memories else []
    
    return {
        "message_type": msg_type,
        "episodic_memories": episodic_summaries,
    }


async def _analyze_sentiment(state: GraphState, deps: GraphDeps) -> GraphState:
    upd = apply_emotion_update(int(state.get("prev_score", 0)), state["user_message"])
    return {"delta": upd.delta, "new_score": upd.new_score, "emotion_label": upd.label}


def _messages_for_llm(state: GraphState) -> list[dict[str, str]]:
    """
    Format messages for GPT API call with memory system integration.
    Includes: system prompt, identity facts, episodic memories, semantic profile,
    working memory, and current user message.
    """
    stage = state["emotion_label"]
    affection = int(state["new_score"])
    
    # Build enhanced system prompt with memory context
    system_parts = [persona_system_prompt(stage, affection)]
    
    # Add identity facts
    identity_facts = state.get("identity_facts", {})
    if identity_facts:
        facts_text = "Known facts about the user:\n"
        for key, value in identity_facts.items():
            facts_text += f"- {key}: {value}\n"
        system_parts.append(facts_text.strip())
    
    # Add episodic memories
    episodic_memories = state.get("episodic_memories", [])
    if episodic_memories:
        memories_text = "Relevant past events:\n"
        for i, memory in enumerate(episodic_memories[:3], 1):  # Top 3
            memories_text += f"{i}. {memory}\n"
        system_parts.append(memories_text.strip())
    
    # Add semantic profile
    semantic_profile = state.get("semantic_profile", {})
    if semantic_profile:
        profile_text = f"User behavior patterns: {semantic_profile.get('personality_summary', '')}"
        system_parts.append(profile_text)
    
    # Combine system prompt
    full_system_prompt = "\n\n".join(system_parts)
    
    msgs: list[dict[str, str]] = [
        {"role": "system", "content": full_system_prompt}
    ]
    
    # Add working memory (last 6-10 turns)
    working_memory = state.get("working_memory", state.get("history", []))
    for m in working_memory[-6:]:  # Last 6 turns
        role = "assistant" if m.get("role") == "ai" else "user"
        msgs.append({"role": role, "content": m.get("message", "")})
    
    # Add current user message
    msgs.append({"role": "user", "content": state["user_message"]})
    
    return msgs


async def _generate_response(state: GraphState, deps: GraphDeps) -> GraphState:
    """
    Generate AI response using GPT via LangGraph node.
    Falls back to deterministic responder if GPT is unavailable.
    Includes retry logic with exponential backoff for rate limits.
    """
    stage = state["emotion_label"]
    affection = int(state["new_score"])
    user_text = state["user_message"]

    # Try GPT first if API key is configured
    if deps.openai_api_key:
        try:
            from openai import OpenAI
            from openai import APIError, AuthenticationError, RateLimitError

            # Create OpenAI client with timeout
            client = OpenAI(
                api_key=deps.openai_api_key,
                timeout=30.0,  # 30 second timeout
                max_retries=3,  # Built-in retry support
            )
            
            # Prepare messages for GPT (includes system prompt, history, and current message)
            messages = _messages_for_llm(state)
            
            # Validate API key format
            if not deps.openai_api_key.startswith("sk-"):
                print(f"[LangGraph] WARNING: API key format looks invalid (should start with 'sk-')")
            
            # Retry logic with exponential backoff for rate limits
            max_retries = 3
            base_delay = 2  # Start with 2 seconds
            
            for attempt in range(max_retries):
                try:
                    # Call GPT through OpenAI API
                    completion = client.chat.completions.create(
                        model=deps.openai_model,
                        messages=messages,
                        temperature=0.8,
                        max_tokens=5000,
                    )
                    
                    # Extract response
                    ai_text = (completion.choices[0].message.content or "").strip()
                    
                    if ai_text:
                        if attempt > 0:
                            print(f"[LangGraph] ✓ GPT response generated (after {attempt} retries)")
                        else:
                            print(f"[LangGraph] ✓ GPT response generated (stage: {stage}, affection: {affection})")
                        return {"ai_message": ai_text}
                    else:
                        print(f"[LangGraph] ⚠ GPT returned empty response, using fallback")
                        break
                        
                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        # Calculate wait time: exponential backoff (2s, 4s, 8s)
                        wait_time = base_delay * (2 ** attempt)
                        print(f"[LangGraph] ⚠ Rate limit hit (attempt {attempt + 1}/{max_retries})")
                        print(f"[LangGraph]   → Waiting {wait_time} seconds before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # Final attempt failed
                        print(f"[LangGraph] ✗ GPT Rate Limit Error (after {max_retries} attempts): {e}")
                        print(f"[LangGraph]   → Rate limit exceeded. Please wait a few minutes.")
                        print(f"[LangGraph]   → Check your usage: https://platform.openai.com/usage")
                        print(f"[LangGraph]   → Using fallback responder")
                        break
                        
        except AuthenticationError as e:
            # Invalid API key
            print(f"[LangGraph] ✗ GPT Authentication Error: {e}")
            print(f"[LangGraph]   → Check your OPENAI_API_KEY in .env file")
            print(f"[LangGraph]   → Using fallback responder")
        except APIError as e:
            # Other API errors
            print(f"[LangGraph] ✗ GPT API Error: {e}")
            print(f"[LangGraph]   → Error type: {type(e).__name__}")
            print(f"[LangGraph]   → Using fallback responder")
        except Exception as e:
            # Other errors (network, timeout, etc.)
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"[LangGraph] ✗ GPT Error ({error_type}): {error_msg}")
            print(f"[LangGraph]   → Possible causes:")
            print(f"[LangGraph]     - Network connection issue")
            print(f"[LangGraph]     - Invalid model name: {deps.openai_model}")
            print(f"[LangGraph]     - API key not set correctly")
            print(f"[LangGraph]   → Using fallback responder")

    # Fallback to deterministic responder
    print(f"[LangGraph] Using fallback responder (stage: {stage}, affection: {affection})")
    return {"ai_message": fallback_tsundere_response(stage, user_text, affection)}


async def _update_memory_systems(state: GraphState, deps: GraphDeps) -> GraphState:
    """Update all memory systems based on message classification"""
    username = state["username"]
    user_message = state["user_message"]
    msg_type = state.get("message_type", "regular")
    
    # Update identity memory if facts detected
    if msg_type == "fact":
        facts = extract_facts_from_message(user_message)
        if facts:
            print(f"[Memory] Detected facts: {list(facts.keys())}")
            await update_identity_memory(
                deps.db,
                username,
                facts,
                source_message=user_message
            )
            print(f"[Memory] Updated identity memory with {len(facts)} facts")
    
    # Create episodic memory for events OR regular messages (with different importance)
    should_create_episodic = False
    importance = 0.3  # Default for regular messages
    
    if msg_type == "event":
        importance = 0.7  # Higher importance for significant events
        should_create_episodic = True
        print(f"[Memory] Detected significant event, creating episodic memory (importance: {importance})")
    elif msg_type == "regular":
        # Create episodic memory for regular messages too, but with lower importance
        # This ensures we have memories to work with even if classification is strict
        # Only create if message is substantial (more than 10 chars)
        if len(user_message.strip()) > 10:
            should_create_episodic = True
            print(f"[Memory] Creating episodic memory for regular message (importance: {importance})")
    
    if should_create_episodic:
        # Generate embedding for the memory
        event_embedding = None
        if deps.openai_api_key:
            try:
                event_embedding = await generate_embedding(
                    user_message,
                    deps.openai_api_key,
                    "text-embedding-3-small"
                )
            except Exception as e:
                print(f"[Memory] Failed to generate embedding: {e}")
        
        # Extract event summary
        summary = user_message[:200]  # First 200 chars as summary
        
        await create_episodic_memory(
            deps.db,
            username,
            summary,
            importance,
            event_embedding,
            metadata={"emotion_score": state.get("new_score", 0), "message_type": msg_type}
        )
        print(f"[Memory] Created episodic memory: {summary[:50]}...")
        
        # Regenerate semantic profile periodically (every 5 new memories)
        # Check how many episodic memories exist
        episodic_coll = deps.db["episodic_memory"]
        memory_count = await episodic_coll.count_documents({"username": username})
        
        if memory_count % 5 == 0:  # Every 5th memory
            print(f"[Memory] Regenerating semantic profile (total memories: {memory_count})")
            from .memory import generate_semantic_profile
            await generate_semantic_profile(deps.db, username)
    
    return {}


async def _persist(state: GraphState, deps: GraphDeps) -> GraphState:
    """Persist conversation and update memory systems"""
    username = state["username"]
    coll = deps.db["chat_messages"]
    ts = datetime.now(timezone.utc)
    score = int(state["new_score"])
    label = state["emotion_label"]

    # Store user message
    prev_score = int(state.get("prev_score", 0))
    prev_label = emotion_label_for_score(prev_score)
    await coll.insert_one(
        {
            "username": username,
            "role": "user",
            "message": state["user_message"],
            "emotion_score": prev_score,
            "emotion_label": prev_label,
            "timestamp": ts,
        }
    )
    
    # Store AI message
    await coll.insert_one(
        {
            "username": username,
            "role": "ai",
            "message": state["ai_message"],
            "emotion_score": score,
            "emotion_label": label,
            "timestamp": ts,
        }
    )
    
    # Update memory systems
    await _update_memory_systems(state, deps)

    return {"timestamp": ts}


def build_chat_graph(deps: GraphDeps):
    """
    Build LangGraph pipeline for chat processing.
    
    Flow:
    1. load -> Load conversation history and emotion state from MongoDB
    2. sentiment -> Calculate emotion score update (deterministic)
    3. respond -> Generate AI response using GPT (if API key set) or fallback
    4. persist -> Save user message and AI response to MongoDB
    
    GPT Integration:
    - If OPENAI_API_KEY is set, the 'respond' node calls GPT directly
    - Messages include system prompt (tsundere persona), history, and current message
    - GPT responses are generated through LangGraph node execution
    """
    g = StateGraph(GraphState)

    async def load(state: GraphState) -> GraphState:
        return await _load_history_and_state(state, deps)

    async def sentiment(state: GraphState) -> GraphState:
        return await _analyze_sentiment(state, deps)

    async def respond(state: GraphState) -> GraphState:
        return await _generate_response(state, deps)

    async def persist(state: GraphState) -> GraphState:
        return await _persist(state, deps)

    async def classify(state: GraphState) -> GraphState:
        return await _classify_and_retrieve_memory(state, deps)

    g.add_node("load", load)
    g.add_node("classify", classify)
    g.add_node("sentiment", sentiment)
    g.add_node("respond", respond)
    g.add_node("persist", persist)

    # LangGraph Flow with Memory System:
    # ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐    ┌─────────┐
    # │  load   │───▶│ classify │───▶│sentiment │───▶│ respond │───▶│ persist │
    # │ (entry) │    │          │    │          │    │  (GPT)  │    │(finish) │
    # └─────────┘    └──────────┘    └──────────┘    └─────────┘    └─────────┘
    #    │              │                │                │              │
    #    │ Load all     │ Classify msg   │ Calculate      │ Generate     │ Save & update
    #    │ memories     │ & retrieve     │ emotion        │ AI response  │ memory systems
    #    │              │ episodic       │ update         │ with memory  │
    #
    g.set_entry_point("load")
    g.add_edge("load", "classify")
    g.add_edge("classify", "sentiment")
    g.add_edge("sentiment", "respond")
    g.add_edge("respond", "persist")
    g.set_finish_point("persist")
    return g.compile()

