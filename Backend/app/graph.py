from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from langgraph.graph import StateGraph
from motor.motor_asyncio import AsyncIOMotorDatabase

from .classifier import classify_message, extract_facts_from_message
from .emotion_judge import get_default_emotion_3d, judge_emotion_3d
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
    
    # 3D Emotion Scores (LLM-judged)
    user_emotion_3d: dict[str, float]  # {"valence": float, "arousal": float, "dominance": float, "impact": float}
    ai_emotion_3d: dict[str, float]  # {"valence": float, "arousal": float, "dominance": float, "impact": float}

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

    # Current emotion state: derive from latest AI message's 3D emotion, else default
    latest_ai = await coll.find_one({"username": username, "role": "ai"}, sort=[("timestamp", -1)])
    if latest_ai and "emotion_3d" in latest_ai:
        prev_score, _ = _derive_persona_from_emotion_3d(latest_ai.get("emotion_3d"))
    else:
        prev_score = 0

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


def calculate_affection_score_from_3d(emotion_3d: dict[str, float] | None) -> float:
    """
    Calculate affection score (0-10) from 3D emotion dimensions.
    
    Equation:
    - Base: Map valence (-1.0 to 1.0) to (0 to 10) linearly
    - Arousal modulation: Higher arousal amplifies the effect (0.7 to 1.3 multiplier)
    - Dominance modulation: Lower dominance (more submissive) slightly increases affection (0.9 to 1.1 multiplier)
    
    Formula:
    base = (valence + 1.0) * 5.0  # Maps -1.0 to 1.0 -> 0.0 to 10.0
    arousal_mult = 0.7 + (arousal * 0.6)  # Maps 0.0 to 1.0 -> 0.7 to 1.3
    dominance_mult = 1.1 - (dominance * 0.2)  # Maps 0.0 to 1.0 -> 1.1 to 0.9 (inverse)
    affection_score = base * arousal_mult * dominance_mult
    
    Returns:
        Affection score in range 0.0 to 10.0
    """
    if not emotion_3d:
        return 5.0  # Neutral default
    
    valence = emotion_3d.get("valence", 0.0)
    arousal = emotion_3d.get("arousal", 0.5)
    dominance = emotion_3d.get("dominance", 0.5)
    
    # Clamp inputs to valid ranges
    valence = max(-1.0, min(1.0, valence))
    arousal = max(0.0, min(1.0, arousal))
    dominance = max(0.0, min(1.0, dominance))
    
    # Base score: Map valence from -1.0 to 1.0 -> 0.0 to 10.0
    base = (valence + 1.0) * 5.0
    
    # Arousal modulation: Higher arousal amplifies the effect
    # Maps arousal 0.0 to 1.0 -> multiplier 0.7 to 1.3
    arousal_mult = 0.7 + (arousal * 0.6)
    
    # Dominance modulation: Lower dominance (more submissive) slightly increases affection
    # In tsundere context, being more submissive can indicate hidden affection
    # Maps dominance 0.0 to 1.0 -> multiplier 1.1 to 0.9 (inverse relationship)
    dominance_mult = 1.1 - (dominance * 0.2)
    
    # Calculate final affection score
    affection_score = base * arousal_mult * dominance_mult
    
    # Clamp to 0-10 range
    affection_score = max(0.0, min(10.0, affection_score))
    
    return affection_score


def _calculate_emotion_score_from_3d(emotion_3d: dict[str, float] | None) -> float:
    """
    Calculate emotion score from 3D emotion.
    Uses valence as base, scaled by impact.
    Returns score in range -5.0 to 5.0.
    """
    if not emotion_3d:
        return 0.0
    
    valence = emotion_3d.get("valence", 0.0)
    impact = emotion_3d.get("impact", 0.5)
    
    # Base score from valence (-5 to 5)
    base_score = valence * 5.0
    
    # Scale by impact (high impact = stronger effect)
    score = base_score * impact
    
    return score


def _derive_persona_from_emotion_3d(emotion_3d: dict[str, float] | None) -> tuple[int, str]:
    """
    Derive persona stage and affection score from 3D emotion.
    Uses the new affection_score calculation (0-10) based on all three dimensions.
    """
    if not emotion_3d:
        return 0, "Cold / Defensive"
    
    # Calculate affection score (0-10) using all three dimensions
    affection_score = calculate_affection_score_from_3d(emotion_3d)
    
    # Convert to integer and map to persona stage
    # Map 0-10 affection score to tsundere stages
    if affection_score <= 2.0:
        stage = "Hostile Tsundere"
        affection = -3
    elif affection_score <= 4.0:
        stage = "Cold / Defensive"
        affection = 0
    elif affection_score <= 7.0:
        stage = "Soft Tsundere"
        affection = 3
    else:
        stage = "Dere Mode"
        affection = 5
    
    return affection, stage


def _calculate_weighted_emotion_score(
    prev_score: float,
    current_score: float,
    current_impact: float,
    prev_weight: float = 0.7,
    current_weight: float = 0.3,
    max_impact_delta: float = 2.0
) -> float:
    """
    Calculate weighted emotion score with maximum impact cap.
    
    Args:
        prev_score: Previous overall emotion score
        current_score: Current message emotion score
        current_impact: Impact of current message (0.0 to 1.0)
        prev_weight: Weight for previous score (default 0.7)
        current_weight: Weight for current score (default 0.3)
        max_impact_delta: Maximum change per message (default 2.0)
    
    Returns:
        New weighted emotion score
    """
    # Calculate weighted combination
    weighted_score = (prev_score * prev_weight) + (current_score * current_weight)
    
    # Calculate raw delta
    delta = weighted_score - prev_score
    
    # Apply maximum impact cap
    if abs(delta) > max_impact_delta:
        delta = max_impact_delta if delta > 0 else -max_impact_delta
    
    # Apply capped delta to previous score
    new_score = prev_score + delta
    
    # Clamp to reasonable range (-10 to 10)
    new_score = max(-10.0, min(10.0, new_score))
    
    return new_score


async def _judge_emotions(state: GraphState, deps: GraphDeps) -> GraphState:
    """Judge 3D emotions for user message using LLM and derive persona stage"""
    user_message = state["user_message"]
    prev_score = state.get("prev_score", 0)
    
    # Build memory context for emotion judge
    memory_context = {
        "identity_facts": state.get("identity_facts", {}),
        "episodic_memories": state.get("episodic_memories", []),
        "semantic_profile": state.get("semantic_profile", {}),
        "working_memory": state.get("working_memory", []),
    }
    
    # Judge user emotion
    user_emotion = await judge_emotion_3d(
        message=user_message,
        role="user",
        memory_context=memory_context,
        openai_api_key=deps.openai_api_key,
        openai_model=deps.openai_model
    )
    
    if user_emotion:
        print(f"[EmotionJudge] User emotion: V={user_emotion.valence:.2f}, A={user_emotion.arousal:.2f}, D={user_emotion.dominance:.2f}, Impact={user_emotion.impact:.2f}")
        user_emotion_dict = {
            "valence": user_emotion.valence,
            "arousal": user_emotion.arousal,
            "dominance": user_emotion.dominance,
            "impact": user_emotion.impact
        }
    else:
        print(f"[EmotionJudge] Using default emotion for user (LLM unavailable)")
        default = get_default_emotion_3d()
        user_emotion_dict = {
            "valence": default.valence,
            "arousal": default.arousal,
            "dominance": default.dominance,
            "impact": default.impact
        }
    
    # Calculate emotion score from 3D emotion
    current_score = _calculate_emotion_score_from_3d(user_emotion_dict)
    current_impact = user_emotion_dict.get("impact", 0.5)
    
    # Calculate weighted score with maximum impact cap
    new_score = _calculate_weighted_emotion_score(
        prev_score=float(prev_score),
        current_score=current_score,
        current_impact=current_impact,
        prev_weight=0.7,  # 70% previous score
        current_weight=0.3,  # 30% current message
        max_impact_delta=2.0  # Maximum change of ±2.0 per message
    )
    
    # Derive persona stage from new weighted score
    # Convert score to valence-like value for stage mapping
    score_valence = new_score / 5.0  # Normalize to -1.0 to 1.0 range
    score_emotion_3d = {"valence": max(-1.0, min(1.0, score_valence))}
    _, emotion_label = _derive_persona_from_emotion_3d(score_emotion_3d)
    
    delta = new_score - prev_score
    print(f"[EmotionJudge] Score: prev={prev_score:.2f}, current={current_score:.2f}, new={new_score:.2f}, delta={delta:.2f} (capped at ±2.0)")
    
    # Calculate user's affection score from their 3D emotion
    user_affection_score = calculate_affection_score_from_3d(user_emotion_dict)
    
    return {
        "user_emotion_3d": user_emotion_dict,
        "delta": delta,
        "new_score": new_score,
        "emotion_label": emotion_label,
        "user_affection_score": user_affection_score  # Store user's affection score (0-10)
    }


def _messages_for_llm(state: GraphState) -> list[dict[str, str]]:
    """
    Format messages for GPT API call with memory system integration.
    Includes: system prompt, identity facts, episodic memories, semantic profile,
    working memory, and current user message.
    """
    stage = state["emotion_label"]
    
    # Calculate affection score (0-10) from weighted emotion score
    # Use the new weighted score to derive affection
    new_score = state.get("new_score", 0)
    user_emotion_3d = state.get("user_emotion_3d", {})
    
    # Calculate AI's affection score from weighted score (convert to 0-10 range)
    # The new_score is in range -10 to 10, map to 0-10
    ai_affection_score = max(0.0, min(10.0, (new_score + 10.0) / 2.0))
    
    # Calculate user's affection score from their 3D emotion
    user_affection_score = calculate_affection_score_from_3d(user_emotion_3d)
    
    # Build enhanced system prompt with memory context and emotion scores
    system_parts = [persona_system_prompt(stage, int(ai_affection_score), user_affection_score)]
    
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
    new_score = state.get("new_score", 0)
    user_emotion_3d = state.get("user_emotion_3d", {})
    
    # Calculate affection scores (0-10) from weighted scores
    ai_affection_score = max(0.0, min(10.0, (new_score + 10.0) / 2.0))
    
    # Get user's affection score from state (calculated in _judge_emotions) or calculate it
    user_affection_score = state.get("user_affection_score")
    if user_affection_score is None:
        user_affection_score = calculate_affection_score_from_3d(user_emotion_3d)
    
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
                            print(f"[LangGraph] ✓ GPT response generated (stage: {stage}, AI affection: {ai_affection_score:.1f}/10, User affection: {user_affection_score:.1f}/10)")
                        
                        # Judge AI emotion after response is generated
                        memory_context = {
                            "identity_facts": state.get("identity_facts", {}),
                            "episodic_memories": state.get("episodic_memories", []),
                            "semantic_profile": state.get("semantic_profile", {}),
                            "working_memory": state.get("working_memory", []),
                        }
                        
                        ai_emotion = await judge_emotion_3d(
                            message=ai_text,
                            role="ai",
                            memory_context=memory_context,
                            openai_api_key=deps.openai_api_key,
                            openai_model=deps.openai_model
                        )
                        
                        if ai_emotion:
                            print(f"[EmotionJudge] AI emotion: V={ai_emotion.valence:.2f}, A={ai_emotion.arousal:.2f}, D={ai_emotion.dominance:.2f}, Impact={ai_emotion.impact:.2f}")
                            ai_emotion_dict = {
                                "valence": ai_emotion.valence,
                                "arousal": ai_emotion.arousal,
                                "dominance": ai_emotion.dominance,
                                "impact": ai_emotion.impact
                            }
                        else:
                            print(f"[EmotionJudge] Using default emotion for AI (LLM unavailable)")
                            default = get_default_emotion_3d()
                            ai_emotion_dict = {
                                "valence": default.valence,
                                "arousal": default.arousal,
                                "dominance": default.dominance,
                                "impact": default.impact
                            }
                        
                        return {
                            "ai_message": ai_text,
                            "ai_emotion_3d": ai_emotion_dict
                        }
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
    print(f"[LangGraph] Using fallback responder (stage: {stage}, AI affection: {ai_affection_score:.1f}/10, User affection: {user_affection_score:.1f}/10)")
    ai_text = fallback_tsundere_response(stage, user_text, int(ai_affection_score))
    
    # Judge AI emotion for fallback response too
    memory_context = {
        "identity_facts": state.get("identity_facts", {}),
        "episodic_memories": state.get("episodic_memories", []),
        "semantic_profile": state.get("semantic_profile", {}),
        "working_memory": state.get("working_memory", []),
    }
    
    ai_emotion = await judge_emotion_3d(
        message=ai_text,
        role="ai",
        memory_context=memory_context,
        openai_api_key=deps.openai_api_key,
        openai_model=deps.openai_model
    )
    
    if ai_emotion:
        ai_emotion_dict = {
            "valence": ai_emotion.valence,
            "arousal": ai_emotion.arousal,
            "dominance": ai_emotion.dominance,
            "impact": ai_emotion.impact
        }
    else:
        default = get_default_emotion_3d()
        ai_emotion_dict = {
            "valence": default.valence,
            "arousal": default.arousal,
            "dominance": default.dominance,
            "impact": default.impact
        }
    
    return {
        "ai_message": ai_text,
        "ai_emotion_3d": ai_emotion_dict
    }


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

    # Get 3D emotions (with defaults if not set)
    user_emotion_3d = state.get("user_emotion_3d", {"valence": 0.0, "arousal": 0.5, "dominance": 0.5, "impact": 0.3})
    ai_emotion_3d = state.get("ai_emotion_3d", {"valence": 0.0, "arousal": 0.5, "dominance": 0.5, "impact": 0.3})

    # Store user message - derive emotion_label from user's 3D emotion
    prev_score = int(state.get("prev_score", 0))
    _, prev_label = _derive_persona_from_emotion_3d(user_emotion_3d)
    await coll.insert_one(
        {
            "username": username,
            "role": "user",
            "message": state["user_message"],
            "emotion_score": prev_score,
            "emotion_label": prev_label,
            "emotion_3d": user_emotion_3d,
            "timestamp": ts,
        }
    )
    
    # Store AI message - derive emotion_label from AI's 3D emotion
    _, ai_label = _derive_persona_from_emotion_3d(ai_emotion_3d)
    await coll.insert_one(
        {
            "username": username,
            "role": "ai",
            "message": state["ai_message"],
            "emotion_score": score,
            "emotion_label": ai_label,
            "emotion_3d": ai_emotion_3d,
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
    1. load -> Load conversation history and memory state from MongoDB
    2. classify -> Classify message and retrieve relevant episodic memories
    3. judge_emotion -> Judge user emotion using LLM (3D: valence, arousal, dominance) and derive persona
    4. respond -> Generate AI response using GPT (if API key set) or fallback, then judge AI emotion
    5. persist -> Save user message and AI response to MongoDB with 3D emotions
    
    Emotion System:
    - Uses LLM-based 3D emotion judge (valence, arousal, dominance)
    - Derives persona stage and affection score from 3D emotions
    - No word-based sentiment analysis
    
    GPT Integration:
    - If OPENAI_API_KEY is set, the 'respond' node calls GPT directly
    - Messages include system prompt (tsundere persona), history, and current message
    - GPT responses are generated through LangGraph node execution
    """
    g = StateGraph(GraphState)

    async def load(state: GraphState) -> GraphState:
        return await _load_history_and_state(state, deps)

    async def respond(state: GraphState) -> GraphState:
        return await _generate_response(state, deps)

    async def persist(state: GraphState) -> GraphState:
        return await _persist(state, deps)

    async def classify(state: GraphState) -> GraphState:
        return await _classify_and_retrieve_memory(state, deps)
    
    async def judge_emotion(state: GraphState) -> GraphState:
        return await _judge_emotions(state, deps)

    g.add_node("load", load)
    g.add_node("classify", classify)
    g.add_node("judge_emotion", judge_emotion)
    g.add_node("respond", respond)
    g.add_node("persist", persist)

    # LangGraph Flow with Memory System and Emotion Judge:
    # ┌─────────┐    ┌──────────┐    ┌──────────────┐    ┌─────────┐    ┌─────────┐
    # │  load   │───▶│ classify │───▶│judge_emotion │───▶│ respond │───▶│ persist │
    # │ (entry) │    │          │    │  (LLM 3D)   │    │  (GPT)  │    │(finish) │
    # └─────────┘    └──────────┘    └──────────────┘    └─────────┘    └─────────┘
    #    │              │                  │                  │              │
    #    │ Load all     │ Classify msg     │ Judge user        │ Generate     │ Save & update
    #    │ memories     │ & retrieve       │ emotion (3D)      │ AI response  │ memory systems
    #    │              │ episodic         │ & derive persona  │ & judge AI   │ & 3D emotions
    #    │              │                  │                   │ emotion (3D) │
    #
    g.set_entry_point("load")
    g.add_edge("load", "classify")
    g.add_edge("classify", "judge_emotion")
    g.add_edge("judge_emotion", "respond")
    g.add_edge("respond", "persist")
    g.set_finish_point("persist")
    return g.compile()

