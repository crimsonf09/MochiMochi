from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient

from .db import ensure_indexes, get_db
from .graph import GraphDeps, build_chat_graph
from .memory import (
    get_identity_memory,
    get_semantic_profile,
    get_working_memory,
)
from .schemas import (
    ChatMessage,
    ChatState,
    MemoryData,
    MirrorMessageIn,
    WsClientMessage,
    WsServerMessage,
)
from .security_agent import get_security_events
from .settings import load_settings

CHAT_NAMESPACE = "mochi"


def _oid_to_str(v: Any) -> str:
    if isinstance(v, ObjectId):
        return str(v)
    return str(v)


def _serialize_doc(doc: dict[str, Any]) -> dict[str, Any]:
    result = {
        "id": _oid_to_str(doc.get("_id")),
        "username": doc.get("username", ""),
        "role": doc.get("role", ""),
        "message": doc.get("message", ""),
        "emotion_score": int(doc.get("emotion_score", 0)),
        "emotion_label": doc.get("emotion_label", ""),
        "timestamp": doc.get("timestamp"),
    }
    emotion_3d = doc.get("emotion_3d")
    if emotion_3d:
        emotion_3d_dict = {
            "valence": emotion_3d.get("valence", 0.0),
            "arousal": emotion_3d.get("arousal", 0.5),
            "dominance": emotion_3d.get("dominance", 0.5),
            "impact": emotion_3d.get("impact", 0.3)
        }
        emotion_3d_dict["valence"] = max(-1.0, min(1.0, emotion_3d_dict["valence"]))
        emotion_3d_dict["arousal"] = max(0.0, min(1.0, emotion_3d_dict["arousal"]))
        emotion_3d_dict["dominance"] = max(0.0, min(1.0, emotion_3d_dict["dominance"]))
        emotion_3d_dict["impact"] = max(0.0, min(1.0, emotion_3d_dict["impact"]))
        result["emotion_3d"] = emotion_3d_dict
    return result


def _mirror_role_to_chat_role(role: str, name: str) -> str:
    role_l = (role or "").strip().lower()
    name_l = (name or "").strip().lower()
    if role_l == "user" or name_l == "user":
        return "user"
    return "ai"


def _dmind_display_text(raw: str) -> str:
    """If stored mirror text is still a DMind JSON payload, show inner content only."""
    s = (raw or "").strip()
    if not s or not s.startswith("{"):
        return s
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and isinstance(obj.get("content"), str):
            inner = obj["content"].strip()
            if inner.startswith("{"):
                try:
                    nested = json.loads(inner)
                    if isinstance(nested, dict) and isinstance(nested.get("content"), str):
                        return nested["content"]
                except json.JSONDecodeError:
                    pass
            return obj["content"]
    except json.JSONDecodeError:
        pass
    return s


def _scoped_username(username: str) -> str:
    u = (username or "").strip()
    if not u:
        return f"{CHAT_NAMESPACE}:anonymous"
    # Keep Dmind session usernames shared so Mochi can read that stream.
    if u.startswith("dmind-"):
        return u
    return f"{CHAT_NAMESPACE}:{u}"


load_dotenv()  # supports local env file usage without committing dotfiles
settings = load_settings()

app = FastAPI(title="Mochi Chat Backend")

cors_origins = settings.cors_origins.copy()
# Local Vite dev may use any port (5173, 5174, …) or 127.0.0.1 — match full Origin for CORSMiddleware.
allow_origin_regex = (
    r"(https://.*\.vercel\.app"
    r"|http://localhost(?::\d+)?"
    r"|http://127\.0\.0\.1(?::\d+)?)"
)
exact_origins = [o for o in cors_origins if "*" not in o and "vercel.app" not in o]
vercel_urls = [o for o in cors_origins if "vercel.app" in o and "*" not in o]
exact_origins.extend(vercel_urls)

app.add_middleware(
    CORSMiddleware,
    allow_origins=exact_origins if exact_origins else ["*"],
    allow_origin_regex=allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    try:
        client = AsyncIOMotorClient(settings.mongo_uri)
        await client.admin.command("ping")
        db = get_db(client, settings.mongo_db)
        await ensure_indexes(db)

        app.state.mongo_client = client
        app.state.db = db
        app.state.graph = build_chat_graph(
            GraphDeps(db=db, openai_api_key=settings.openai_api_key, openai_model=settings.openai_model)
        )
    except Exception:
        raise


@app.on_event("shutdown")
async def _shutdown() -> None:
    client: AsyncIOMotorClient | None = getattr(app.state, "mongo_client", None)
    if client is not None:
        client.close()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "mochi-chat-backend"}


@app.get("/chat/history/{username}", response_model=list[ChatMessage])
async def get_chat_history(username: str):
    try:
        scoped_username = _scoped_username(username)
        db = app.state.db
        # For DMind bridge sessions, prefer exact mirrored DMind messages.
        if scoped_username.startswith("dmind-"):
            mirror_coll = db["dmind_mirror_messages"]
            mirror_cursor = mirror_coll.find({"username": scoped_username}).sort("timestamp", 1)
            mirror_docs = await mirror_cursor.to_list(length=20_000)
            if mirror_docs:
                serialized = []
                for doc in mirror_docs:
                    try:
                        serialized.append(
                            {
                                "id": _oid_to_str(doc.get("_id")),
                                "username": scoped_username,
                                "role": _mirror_role_to_chat_role(doc.get("dmind_role", ""), doc.get("dmind_name", "")),
                                "message": _dmind_display_text(str(doc.get("message", ""))),
                                "emotion_score": 0,
                                "emotion_label": "",
                                "timestamp": doc.get("timestamp"),
                            }
                        )
                    except Exception:
                        continue
                return serialized

        coll = db["chat_messages"]
        cursor = coll.find({"username": scoped_username}).sort("timestamp", 1)
        docs = await cursor.to_list(length=10_000)
        serialized = []
        for doc in docs:
            try:
                serialized.append(_serialize_doc(doc))
            except Exception:
                continue
        return serialized
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/chat/mirror/{username}")
async def mirror_chat_history(username: str, messages: list[MirrorMessageIn]):
    """
    Replace mirrored DMind snapshot for the given username.
    This enables Mochi frontend to render an exact DMind chat stream.
    """
    try:
        scoped_username = _scoped_username(username)
        db = app.state.db
        coll = db["dmind_mirror_messages"]

        await coll.delete_many({"username": scoped_username})
        if not messages:
            return {"ok": True, "username": scoped_username, "count": 0}

        docs = [
            {
                "username": scoped_username,
                "dmind_id": m.id,
                "dmind_role": m.role,
                "dmind_name": m.name,
                "message": m.content,
                "timestamp": m.timestamp,
            }
            for m in messages
        ]
        await coll.insert_many(docs)
        return {"ok": True, "username": scoped_username, "count": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mirror sync error: {str(e)}")


@app.get("/chat/state/{username}", response_model=ChatState)
async def get_chat_state(username: str):
    """Return current user state. emotion_score is in [-10, 10] (from latest AI message)."""
    try:
        scoped_username = _scoped_username(username)
        db = app.state.db
        coll = db["chat_messages"]
        latest_ai = await coll.find_one({"username": scoped_username, "role": "ai"}, sort=[("timestamp", -1)])
        if latest_ai and "emotion_score" in latest_ai:
            emotion_score = max(-10, min(10, int(latest_ai.get("emotion_score", 0))))
        elif latest_ai and "emotion_3d" in latest_ai:
            from .graph import _derive_affection_from_emotion_3d
            aff = _derive_affection_from_emotion_3d(latest_ai.get("emotion_3d"))
            emotion_score = max(-10, min(10, int(round(aff * 2 - 10))))
        else:
            emotion_score = 0
        return {"username": username, "emotion_score": emotion_score, "persona_stage": ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/security/events/{username}", response_model=list[dict])
async def get_user_security_events(username: str, limit: int = 50):
    """
    Get security events for a specific user.
    Useful for monitoring and analysis.
    """
    try:
        db = app.state.db
        events = await get_security_events(db, username=_scoped_username(username), limit=limit)
        return events
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/security/events", response_model=list[dict])
async def get_all_security_events(limit: int = 100):
    """
    Get all security events (admin endpoint).
    Useful for monitoring system-wide security threats.
    """
    try:
        db = app.state.db
        events = await get_security_events(db, username=None, limit=limit)
        return events
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/chat/memory/{username}", response_model=MemoryData)
async def get_chat_memory(username: str):
    """Get all memory data for a user"""
    try:
        scoped_username = _scoped_username(username)
        db = app.state.db
        identity_facts_dict = await get_identity_memory(db, scoped_username)
        identity_facts = [
            {"key": k, "value": v.value, "confidence": v.confidence}
            for k, v in identity_facts_dict.items()
        ]
        episodic_coll = db["episodic_memory"]
        episodic_cursor = (
            episodic_coll.find({"username": scoped_username})
            .sort("importance_score", -1)
            .limit(20)
        )
        episodic_docs = await episodic_cursor.to_list(length=20)
        episodic_memories = [
            {
                "event_summary": doc["event_summary"],
                "importance_score": doc.get("importance_score", 0.5),
                "timestamp": doc.get("timestamp", datetime.now(timezone.utc)),
                "access_count": doc.get("access_count", 0),
            }
            for doc in episodic_docs
        ]
        semantic_profile = await get_semantic_profile(
            db, scoped_username, settings.openai_api_key, settings.openai_model
        )
        working_memory = await get_working_memory(db, scoped_username, max_turns=10)
        result = {
            "identity_facts": identity_facts,
            "episodic_memories": episodic_memories,
            "semantic_profile": {
                "personality_summary": semantic_profile.get("personality_summary", "No patterns detected yet."),
                "preferences": semantic_profile.get("preferences", {}),
                "behavior_patterns": semantic_profile.get("behavior_patterns", []),
            },
            "working_memory_count": len(working_memory),
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory retrieval error: {str(e)}")


@app.websocket("/ws/{username}")
async def ws_chat(websocket: WebSocket, username: str):
    await websocket.accept()
    graph = app.state.graph
    scoped_username = _scoped_username(username)

    try:
        while True:
            payload = await websocket.receive_json()
            try:
                client_msg = WsClientMessage.model_validate(payload)
            except Exception:
                await websocket.send_json(
                    {"error": "Invalid message format. Expected: { message: string, memory_enabled?: boolean }"}
                )
                continue

            try:
                result = await graph.ainvoke(
                    {
                        "username": scoped_username,
                        "user_message": client_msg.message,
                        "memory_enabled": client_msg.memory_enabled,
                    }
                )
            except Exception as e:
                await websocket.send_json({"error": f"LLM unavailable: {str(e)}"})
                continue
            ai_emotion_3d = result.get("ai_emotion_3d")
            from .schemas import Emotion3D
            emotion_3d_obj = None
            if ai_emotion_3d:
                emotion_3d_obj = Emotion3D(**ai_emotion_3d)
            new_score = result.get("new_score", 0)
            raw_score = max(-10, min(10, int(round(float(new_score)))))
            server_msg = WsServerMessage(
                message=result["ai_message"],
                emotion_score=raw_score,
                weighted_score=float(raw_score),
                emotion_3d=emotion_3d_obj,
                timestamp=result.get("timestamp") or datetime.utcnow(),
            )
            await websocket.send_json(server_msg.model_dump(mode="json"))
    except WebSocketDisconnect:
        return

