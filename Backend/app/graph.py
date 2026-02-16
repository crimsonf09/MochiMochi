from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from langgraph.graph import StateGraph
from motor.motor_asyncio import AsyncIOMotorDatabase

from .emotion_judge import get_default_emotion_3d, judge_emotion_3d
from .guardrail import check_guardrail
from .memory import (
    create_episodic_memory,
    generate_embedding,
    get_identity_memory,
    get_semantic_profile,
    get_working_memory,
    search_episodic_memory,
    trim_working_memory_if_needed,
    update_identity_memory_from_conversation,
)
from .security_agent import handle_dangerous_message
from .tsundere import fallback_tsundere_response, persona_system_prompt


class GraphState(TypedDict, total=False):
    username: str
    user_message: str
    history: list[dict[str, Any]]
    prev_score: int
    identity_facts: dict[str, Any]
    working_memory: list[dict[str, Any]]
    episodic_memories: list[Any]
    semantic_profile: dict[str, Any]
    prev_ai_emotion_3d: dict[str, float] | None
    delta: int
    new_score: int
    weighted_score: float
    ai_message: str
    user_emotion_3d: dict[str, float]
    ai_emotion_3d: dict[str, float]
    guardrail_passed: bool
    guardrail_reason: str
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
    working_memory = await trim_working_memory_if_needed(
        deps.db, username, max_tokens=2000,
        openai_api_key=deps.openai_api_key, openai_model=deps.openai_model
    )
    if not working_memory:
        working_memory = await get_working_memory(deps.db, username, max_turns=10)
    history = working_memory
    latest_ai = await coll.find_one({"username": username, "role": "ai"}, sort=[("timestamp", -1)])
    if latest_ai is not None and "emotion_score" in latest_ai:
        prev_score = int(latest_ai.get("emotion_score", 0))
        prev_score = max(-10, min(10, prev_score))
    else:
        prev_score = 0
    identity_facts = await get_identity_memory(deps.db, username)
    identity_dict = {k: v.value for k, v in identity_facts.items()}
    semantic_profile = await get_semantic_profile(
        deps.db, username, deps.openai_api_key, deps.openai_model
    )
    prev_ai_emotion_3d = None
    if latest_ai and "emotion_3d" in latest_ai:
        prev_ai_emotion_3d = latest_ai.get("emotion_3d")

    return {
        "history": history,
        "working_memory": working_memory,
        "prev_score": prev_score,
        "identity_facts": identity_dict,
        "semantic_profile": semantic_profile,
        "prev_ai_emotion_3d": prev_ai_emotion_3d,  # Store for use in LLM prompt
    }


async def _classify_and_retrieve_memory(state: GraphState, deps: GraphDeps) -> GraphState:
    """Retrieve relevant episodic memories by embedding similarity (no classifier)."""
    username = state["username"]
    user_message = state["user_message"]

    query_embedding = None
    if deps.openai_api_key:
        try:
            query_embedding = await generate_embedding(
                user_message,
                deps.openai_api_key,
                "text-embedding-3-small"
            )
        except Exception:
            pass
    episodic_memories = []
    if query_embedding:
        episodic_memories = await search_episodic_memory(
            deps.db,
            username,
            query_embedding,
            top_k=5
        )
    episodic_summaries = [m.event_summary for m in episodic_memories] if episodic_memories else []
    return {"episodic_memories": episodic_summaries}


def calculate_affection_score_from_3d(emotion_3d: dict[str, float] | None) -> float:
    """
    Affection score (0-10) from 3D emotion. Valence dominates so aggressive (negative valence) => low affection.
    - Valence: -1..1 -> 0 to 8 (main driver; negative valence pulls score down)
    - Arousal: 0..1 -> 0 to 1
    - Dominance: 0..1 -> (1-d) -> 0 to 1 (submissive = slightly higher)
    Sum 0-10, clamped. All inputs clamped so no wrong results.
    """
    if not emotion_3d:
        return 5.0
    v = max(-1.0, min(1.0, float(emotion_3d.get("valence", 0.0))))
    a = max(0.0, min(1.0, float(emotion_3d.get("arousal", 0.5))))
    d = max(0.0, min(1.0, float(emotion_3d.get("dominance", 0.5))))
    valence_contrib = (v + 1.0) * 4.0
    arousal_contrib = a * 1.0
    dominance_contrib = (1.0 - d) * 1.0
    return max(0.0, min(10.0, valence_contrib + arousal_contrib + dominance_contrib))


def _calculate_emotion_score_from_3d(emotion_3d: dict[str, float] | None) -> float:
    """
    Emotion score from 3D: valence as base, scaled by impact. Returns roughly -3.5 to 4.0.
    All inputs clamped so bad LLM output cannot produce wrong results.
    """
    if not emotion_3d:
        return 0.0
    valence = max(-1.0, min(1.0, float(emotion_3d.get("valence", 0.0))))
    impact = max(0.0, min(1.0, float(emotion_3d.get("impact", 0.5))))
    arousal = max(0.0, min(1.0, float(emotion_3d.get("arousal", 0.5))))
    dominance = max(0.0, min(1.0, float(emotion_3d.get("dominance", 0.5))))
    if valence > 0:
        base_score = valence * 4.0
    else:
        base_score = valence * 3.5
    if valence > 0:
        impact_multiplier = 0.8 + (impact * 0.4)
    else:
        impact_multiplier = 0.7 + (impact * 0.3)
    score = base_score * impact_multiplier
    return score


def _derive_affection_from_emotion_3d(emotion_3d: dict[str, float] | None) -> float:
    """
    Derive affection score from 3D emotion.
    Uses the new affection_score calculation (0-10) based on all three dimensions.
    """
    if not emotion_3d:
        return 5.0
    affection_score = calculate_affection_score_from_3d(emotion_3d)
    
    return affection_score


def _apply_weight_to_emotion_3d(
    current_emotion_3d: dict[str, float],
    current_impact: float
) -> tuple[dict[str, float], float]:
    """
    Apply impact to each dimension (V, A, D). Weight and impact are the same value.
    Impact (0–1) = how much this message matters; we use it directly as the multiplier.
    
    Flow:
    1. LLM judges impact (0.0 to 1.0)
    2. Use impact as weight (same value), clamped to [0.1, 1.0] so always positive
    3. Multiply each dimension (V, A, D) with impact
    4. Calculate emotion score from weighted 3D emotion
    """
    impact_clamped = max(0.0, min(1.0, current_impact))
    weight = max(0.1, min(1.0, impact_clamped))
    curr_v = current_emotion_3d.get("valence", 0.0)
    curr_a = current_emotion_3d.get("arousal", 0.5)
    curr_d = current_emotion_3d.get("dominance", 0.5)
    weighted_v = curr_v * weight
    weighted_a = curr_a * weight
    weighted_d = curr_d * weight
    weighted_emotion_3d = {
        "valence": weighted_v,
        "arousal": weighted_a,
        "dominance": weighted_d,
        "impact": weight
    }
    weighted_score = _calculate_emotion_score_from_3d(weighted_emotion_3d)
    return weighted_emotion_3d, weighted_score


async def _check_guardrail(state: GraphState, deps: GraphDeps) -> GraphState:
    """
    Check user message for prompt injection and system manipulation attempts.
    Uses both word-based and LLM-based detection.
    Routes dangerous messages to security agent for analysis and logging.
    """
    user_message = state["user_message"]
    username = state["username"]
    guardrail_result = await check_guardrail(
        message=user_message,
        openai_api_key=deps.openai_api_key,
        openai_model=deps.openai_model,
        use_llm=True
    )
    if not guardrail_result.is_safe:
        guardrail_dict = {
            "risk_level": guardrail_result.risk_level,
            "reason": guardrail_result.reason,
            "sanitized_message": guardrail_result.sanitized_message,
            "detection_method": guardrail_result.detection_method
        }
        try:
            security_result = await handle_dangerous_message(
                message=user_message,
                username=username,
                guardrail_result=guardrail_dict,
                db=deps.db,
                openai_api_key=deps.openai_api_key,
                openai_model=deps.openai_model
            )
            security_response = security_result.get("response", "")
        except Exception:
            security_response = "Hmph... I'm not going to follow strange instructions like that. What do you actually want to talk about?"
        if guardrail_result.risk_level == "high":
            return {
                "user_message": security_response,  # Replace with security agent response
                "guardrail_passed": False,
                "guardrail_reason": f"High-risk prompt injection detected: {guardrail_result.reason}. Sent to security agent for analysis."
            }
        else:
            sanitized = guardrail_result.sanitized_message or user_message
            return {
                "user_message": sanitized,
                "guardrail_passed": True,
                "guardrail_reason": f"Message sanitized: {guardrail_result.reason}. Logged to security agent."
            }
    return {
        "guardrail_passed": True,
        "guardrail_reason": "Message passed all guardrail checks"
    }


async def _judge_emotions(state: GraphState, deps: GraphDeps) -> GraphState:
    """
    Affection = previous + (valence * weight*2). No word lists.
    valence from LLM (-1 to 1), weight = impact (0 to 1), prev_score (-10 to 10).
    """
    user_message = state["user_message"]
    prev_score_float = float(state.get("prev_score", 0))
    max_step = 2.0  # max change per message when valence=±1 and impact=1

    memory_context = {
        "identity_facts": state.get("identity_facts", {}),
        "episodic_memories": state.get("episodic_memories", []),
        "semantic_profile": state.get("semantic_profile", {}),
        "working_memory": state.get("working_memory", []),
    }

    user_emotion = await judge_emotion_3d(
        message=user_message,
        role="user",
        memory_context=memory_context,
        openai_api_key=deps.openai_api_key,
        openai_model=deps.openai_model,
    )

    if user_emotion:
        valence = max(-1.0, min(1.0, user_emotion.valence))
        weight = max(0.0, min(1.0, user_emotion.impact))
        user_emotion_dict = {
            "valence": user_emotion.valence,
            "arousal": user_emotion.arousal,
            "dominance": user_emotion.dominance,
            "impact": user_emotion.impact,
        }
    else:
        valence = 0.0
        weight = 0.5
        default = get_default_emotion_3d()
        user_emotion_dict = {
            "valence": default.valence,
            "arousal": default.arousal,
            "dominance": default.dominance,
            "impact": default.impact,
        }
    delta = valence * weight * max_step
    new_score = prev_score_float + delta
    new_score = max(-10.0, min(10.0, new_score))
    user_affection_score = (new_score + 10.0) / 2.0
    delta_actual = new_score - prev_score_float

    return {
        "user_emotion_3d": user_emotion_dict,
        "delta": delta_actual,
        "new_score": new_score,
        "weighted_score": new_score,
        "user_affection_score": user_affection_score,
    }


def _messages_for_llm(state: GraphState) -> list[dict[str, str]]:
    """
    Format messages for GPT API call with memory system integration.
    Includes: system prompt, identity facts, episodic memories, semantic profile,
    working memory, and current user message.
    """
    new_score = state.get("new_score", 0)
    user_emotion_3d = state.get("user_emotion_3d", {})
    ai_affection_score = max(0.0, min(10.0, (new_score + 10.0) / 2.0))
    user_affection_score = state.get("user_affection_score")
    if user_affection_score is None:
        user_affection_score = calculate_affection_score_from_3d(user_emotion_3d)
    prev_ai_emotion_3d = state.get("prev_ai_emotion_3d", {})
    if prev_ai_emotion_3d:
        ai_valence = prev_ai_emotion_3d.get("valence", 0.0)
        ai_arousal = prev_ai_emotion_3d.get("arousal", 0.5)
        ai_dominance = prev_ai_emotion_3d.get("dominance", 0.5)
    else:
        ai_valence = (ai_affection_score / 10.0) * 2.0 - 1.0
        ai_arousal = 0.5
        ai_dominance = 0.5
    system_parts = [persona_system_prompt(
        ai_affection=ai_affection_score,
        user_affection=user_affection_score,
        ai_valence=ai_valence,
        ai_arousal=ai_arousal,
        ai_dominance=ai_dominance,
        character_name="Mochi"
    )]
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
    semantic_profile = state.get("semantic_profile", {})
    if semantic_profile:
        profile_text = f"User behavior patterns: {semantic_profile.get('personality_summary', '')}"
        system_parts.append(profile_text)
    
    full_system_prompt = "\n\n".join(system_parts)
    
    msgs: list[dict[str, str]] = [
        {"role": "system", "content": full_system_prompt}
    ]
    working_memory = state.get("working_memory", state.get("history", []))
    for m in working_memory[-6:]:
        role = "assistant" if m.get("role") == "ai" else "user"
        msgs.append({"role": role, "content": m.get("message", "")})
    msgs.append({"role": "user", "content": state["user_message"]})
    
    return msgs


async def _generate_response(state: GraphState, deps: GraphDeps) -> GraphState:
    """
    Generate AI response using GPT via LangGraph node.
    Falls back to deterministic responder if GPT is unavailable.
    Includes retry logic with exponential backoff for rate limits.
    """
    new_score = state.get("new_score", 0)
    user_emotion_3d = state.get("user_emotion_3d", {})
    ai_affection_score = max(0.0, min(10.0, (new_score + 10.0) / 2.0))
    user_affection_score = state.get("user_affection_score")
    if user_affection_score is None:
        user_affection_score = calculate_affection_score_from_3d(user_emotion_3d)
    
    user_text = state["user_message"]
    if deps.openai_api_key:
        try:
            from openai import OpenAI
            from openai import APIError, AuthenticationError, RateLimitError

            client = OpenAI(
                api_key=deps.openai_api_key,
                timeout=30.0,
                max_retries=3,
            )
            messages = _messages_for_llm(state)
            max_retries = 3
            base_delay = 2
            for attempt in range(max_retries):
                try:
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
                            print(f"[LangGraph] ✓ GPT response generated (AI affection: {ai_affection_score:.1f}/10, User affection: {user_affection_score:.1f}/10)")
                        
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
                    else:
                        break
                except RateLimitError:
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt)
                        await asyncio.sleep(wait_time)
                        continue
                    break
        except (AuthenticationError, APIError, Exception):
            pass
    ai_text = fallback_tsundere_response(user_text, ai_affection_score)
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
    """Update episodic memory and semantic profile. Identity facts are handled every 5 turns via LLM."""
    username = state["username"]
    user_message = state["user_message"]
    if len(user_message.strip()) > 10:
        importance = 0.4
        event_embedding = None
        if deps.openai_api_key:
            try:
                event_embedding = await generate_embedding(
                    user_message,
                    deps.openai_api_key,
                    "text-embedding-3-small"
                )
            except Exception:
                pass
        summary = user_message[:200]
        await create_episodic_memory(
            deps.db,
            username,
            summary,
            importance,
            event_embedding,
            metadata={"emotion_score": state.get("new_score", 0)}
        )
        episodic_coll = deps.db["episodic_memory"]
        memory_count = await episodic_coll.count_documents({"username": username})
        if memory_count % 5 == 0:
            from .memory import generate_semantic_profile
            await generate_semantic_profile(
                deps.db, username, deps.openai_api_key, deps.openai_model
            )
    return {}


async def _persist(state: GraphState, deps: GraphDeps) -> GraphState:
    """Persist conversation and update memory systems"""
    username = state["username"]
    coll = deps.db["chat_messages"]
    ts = datetime.now(timezone.utc)
    score = int(state["new_score"])
    user_emotion_3d = state.get("user_emotion_3d", {"valence": 0.0, "arousal": 0.5, "dominance": 0.5, "impact": 0.3})
    ai_emotion_3d = state.get("ai_emotion_3d", {"valence": 0.0, "arousal": 0.5, "dominance": 0.5, "impact": 0.3})
    prev_score = int(state.get("prev_score", 0))
    await coll.insert_one(
        {
            "username": username,
            "role": "user",
            "message": state["user_message"],
            "emotion_score": prev_score,
            "emotion_3d": user_emotion_3d,
            "timestamp": ts,
        }
    )
    await coll.insert_one(
        {
            "username": username,
            "role": "ai",
            "message": state["ai_message"],
            "emotion_score": score,
            "emotion_3d": ai_emotion_3d,
            "timestamp": ts,
        }
    )
    message_count = await coll.count_documents({"username": username})
    if message_count >= 5 and message_count % 5 == 0 and deps.openai_api_key:
        await update_identity_memory_from_conversation(
            deps.db, username, deps.openai_api_key, deps.openai_model, last_n_turns=10
        )
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
    - Calculates weighted emotion score and affection score from 3D emotions
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

    async def guardrail(state: GraphState) -> GraphState:
        return await _check_guardrail(state, deps)
    g.add_node("guardrail", guardrail)
    g.set_entry_point("load")
    g.add_edge("load", "guardrail")
    g.add_edge("guardrail", "classify")
    g.add_edge("classify", "judge_emotion")
    g.add_edge("judge_emotion", "respond")
    g.add_edge("respond", "persist")
    g.set_finish_point("persist")
    return g.compile()

