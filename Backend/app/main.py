from __future__ import annotations

from datetime import datetime
from typing import Any

from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient

from .db import ensure_indexes, get_db
from .emotion import emotion_label_for_score
from .graph import GraphDeps, build_chat_graph
from .schemas import ChatMessage, ChatState, WsClientMessage, WsServerMessage
from .settings import load_settings


def _oid_to_str(v: Any) -> str:
    if isinstance(v, ObjectId):
        return str(v)
    return str(v)


def _serialize_doc(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _oid_to_str(doc.get("_id")),
        "username": doc.get("username", ""),
        "role": doc.get("role", ""),
        "message": doc.get("message", ""),
        "emotion_score": int(doc.get("emotion_score", 0)),
        "emotion_label": doc.get("emotion_label", ""),
        "timestamp": doc.get("timestamp"),
    }


load_dotenv()  # supports local env file usage without committing dotfiles
settings = load_settings()

app = FastAPI(title="Tsundere Chat Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    try:
        client = AsyncIOMotorClient(settings.mongo_uri)
        # Test connection
        await client.admin.command("ping")
        db = get_db(client, settings.mongo_db)
        await ensure_indexes(db)

        app.state.mongo_client = client
        app.state.db = db
        app.state.graph = build_chat_graph(
            GraphDeps(db=db, openai_api_key=settings.openai_api_key, openai_model=settings.openai_model)
        )
        print(f"[OK] Backend started: MongoDB connected, CORS origins: {settings.cors_origins}")
    except Exception as e:
        print(f"[ERROR] Startup error: {e}")
        print(f"  MongoDB URI: {settings.mongo_uri}")
        print(f"  Make sure MongoDB is running!")
        raise


@app.on_event("shutdown")
async def _shutdown() -> None:
    client: AsyncIOMotorClient | None = getattr(app.state, "mongo_client", None)
    if client is not None:
        client.close()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "tsundere-chat-backend"}


@app.get("/chat/history/{username}", response_model=list[ChatMessage])
async def get_chat_history(username: str):
    try:
        db = app.state.db
        coll = db["chat_messages"]
        cursor = coll.find({"username": username}).sort("timestamp", 1)
        docs = await cursor.to_list(length=10_000)
        return [_serialize_doc(d) for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/chat/state/{username}", response_model=ChatState)
async def get_chat_state(username: str):
    try:
        db = app.state.db
        coll = db["chat_messages"]
        latest_ai = await coll.find_one({"username": username, "role": "ai"}, sort=[("timestamp", -1)])
        score = int(latest_ai["emotion_score"]) if latest_ai and "emotion_score" in latest_ai else 0
        stage = latest_ai.get("emotion_label") if latest_ai else emotion_label_for_score(score)
        return {"username": username, "affection_score": score, "persona_stage": stage}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.websocket("/ws/{username}")
async def ws_chat(websocket: WebSocket, username: str):
    await websocket.accept()
    graph = app.state.graph

    try:
        while True:
            payload = await websocket.receive_json()
            try:
                client_msg = WsClientMessage.model_validate(payload)
            except Exception:
                await websocket.send_json({"error": "Invalid message format. Expected: { message: string }"})
                continue

            result = await graph.ainvoke({"username": username, "user_message": client_msg.message})
            server_msg = WsServerMessage(
                message=result["ai_message"],
                emotion_score=int(result["new_score"]),
                emotion_label=result["emotion_label"],
                timestamp=result.get("timestamp") or datetime.utcnow(),
            )
            await websocket.send_json(server_msg.model_dump(mode="json"))
    except WebSocketDisconnect:
        return

