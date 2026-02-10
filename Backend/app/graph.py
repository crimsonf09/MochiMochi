from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from langgraph.graph import StateGraph
from motor.motor_asyncio import AsyncIOMotorDatabase

from .emotion import apply_emotion_update, emotion_label_for_score
from .tsundere import fallback_tsundere_response, persona_system_prompt


class GraphState(TypedDict, total=False):
    username: str
    user_message: str

    # Loaded from DB
    history: list[dict[str, Any]]  # last 10 messages, ascending
    prev_score: int

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
    username = state["username"]
    coll = deps.db["chat_messages"]

    # Short-term memory: last 10 messages (ascending)
    cursor = (
        coll.find({"username": username}, projection={"_id": 0, "role": 1, "message": 1})
        .sort("timestamp", -1)
        .limit(10)
    )
    recent = await cursor.to_list(length=10)
    history = list(reversed(recent))

    # Current emotion state: latest AI message score, else 0
    latest_ai = await coll.find_one({"username": username, "role": "ai"}, sort=[("timestamp", -1)])
    prev_score = int(latest_ai["emotion_score"]) if latest_ai and "emotion_score" in latest_ai else 0

    return {"history": history, "prev_score": prev_score}


async def _analyze_sentiment(state: GraphState, deps: GraphDeps) -> GraphState:
    upd = apply_emotion_update(int(state.get("prev_score", 0)), state["user_message"])
    return {"delta": upd.delta, "new_score": upd.new_score, "emotion_label": upd.label}


def _messages_for_llm(state: GraphState) -> list[dict[str, str]]:
    stage = state["emotion_label"]
    affection = int(state["new_score"])
    msgs: list[dict[str, str]] = [{"role": "system", "content": persona_system_prompt(stage, affection)}]
    for m in state.get("history", []):
        role = "assistant" if m.get("role") == "ai" else "user"
        msgs.append({"role": role, "content": m.get("message", "")})
    msgs.append({"role": "user", "content": state["user_message"]})
    return msgs


async def _generate_response(state: GraphState, deps: GraphDeps) -> GraphState:
    stage = state["emotion_label"]
    affection = int(state["new_score"])
    user_text = state["user_message"]

    if deps.openai_api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=deps.openai_api_key)
            completion = client.chat.completions.create(
                model=deps.openai_model,
                messages=_messages_for_llm(state),
                temperature=0.8,
            )
            ai_text = (completion.choices[0].message.content or "").strip()
            if ai_text:
                return {"ai_message": ai_text}
        except Exception:
            # Fall back to deterministic responder
            pass

    return {"ai_message": fallback_tsundere_response(stage, user_text, affection)}


async def _persist(state: GraphState, deps: GraphDeps) -> GraphState:
    username = state["username"]
    coll = deps.db["chat_messages"]
    ts = datetime.now(timezone.utc)
    score = int(state["new_score"])
    label = state["emotion_label"]

    # Store user message (with the previous score label for visibility; not required but helpful)
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
    # Store AI message (authoritative for emotion history)
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

    return {"timestamp": ts}


def build_chat_graph(deps: GraphDeps):
    """
    LangGraph pipeline:
    load DB state -> deterministic emotion update -> response generation -> persist.
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

    g.add_node("load", load)
    g.add_node("sentiment", sentiment)
    g.add_node("respond", respond)
    g.add_node("persist", persist)

    g.set_entry_point("load")
    g.add_edge("load", "sentiment")
    g.add_edge("sentiment", "respond")
    g.add_edge("respond", "persist")
    g.set_finish_point("persist")
    return g.compile()

