"""
Memory System for Chatbot
Implements Identity Memory, Working Memory, Episodic Memory, and Semantic Profile.
Uses LLM for semantic profile generation and working-memory summarization.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase


async def _llm_chat(
    prompt: str,
    openai_api_key: str | None,
    openai_model: str = "gpt-4o-mini",
    timeout: float = 20.0
) -> str | None:
    """Single LLM call; returns content string or None if disabled/failed."""
    if not openai_api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key, timeout=timeout)
        response = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return None


@dataclass
class IdentityFact:
    """Structured fact about the user"""
    key: str  # e.g., "name", "birthday", "job"
    value: str
    confidence: float = 1.0  # 0.0 to 1.0
    source_message: str = ""  # Message that provided this fact
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


async def update_identity_memory(
    db: AsyncIOMotorDatabase,
    username: str,
    facts: dict[str, tuple[str, float]],
    source_message: str = ""
) -> None:
    """
    Update identity memory with new facts.
    facts: {key: (value, confidence)}
    """
    coll = db["identity_memory"]
    now = datetime.now(timezone.utc)
    
    for key, (value, confidence) in facts.items():
        # Check if fact already exists
        existing = await coll.find_one({"username": username, "key": key})
        
        if existing:
            # Update if new confidence is higher
            if confidence > existing.get("confidence", 0.0):
                await coll.update_one(
                    {"username": username, "key": key},
                    {
                        "$set": {
                            "value": value,
                            "confidence": confidence,
                            "source_message": source_message,
                            "updated_at": now,
                        }
                    }
                )
        else:
            await coll.insert_one({
                "username": username,
                "key": key,
                "value": value,
                "confidence": confidence,
                "source_message": source_message,
                "updated_at": now,
            })


async def extract_identity_facts_with_llm(
    messages: list[dict[str, Any]],
    openai_api_key: str | None,
    openai_model: str = "gpt-4o-mini"
) -> dict[str, tuple[str, float]]:
    """
    Use LLM to extract identity facts about the user from recent conversation turns.
    messages: list of {role, message} in chronological order.
    Returns dict key -> (value, confidence) for update_identity_memory.
    """
    if not openai_api_key or not messages:
        return {}
    conv_text = "\n".join(
        f"{m.get('role', 'unknown')}: {m.get('message', '')}"
        for m in messages[-20:]  # last 20 turns
    )
    if len(conv_text) > 3500:
        conv_text = conv_text[-3500:]
    prompt = f"""From this conversation, extract only explicit facts about the USER (the human). Include only what the user clearly stated about themselves.

Possible fact types: name (or nickname), birthday, age, job/work, preferences (likes/dislikes), goal, location, relationship status. Use short keys: name, birthday, age, job, preference_1, preference_2, goal, etc.

Conversation:
{conv_text}

Return a JSON object with keys as fact names and values as the exact fact (string). Only include facts the user clearly stated. If nothing clear, return {{}}.
Example: {{"name": "Alice", "job": "developer"}}
JSON only, no other text:"""
    raw = await _llm_chat(prompt, openai_api_key, openai_model, timeout=25.0)
    if not raw:
        return {}
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        facts = {}
        for k, v in data.items():
            if not isinstance(k, str) or not isinstance(v, (str, int, float)):
                continue
            val = str(v).strip()
            if not val:
                continue
            conf = 0.9 if k in ("name", "birthday", "age", "job") else 0.8
            facts[k] = (val[:500], conf)
        return facts
    except json.JSONDecodeError:
        return {}


async def update_identity_memory_from_conversation(
    db: AsyncIOMotorDatabase,
    username: str,
    openai_api_key: str | None,
    openai_model: str = "gpt-4o-mini",
    last_n_turns: int = 10
) -> int:
    """
    Get last N conversation turns, extract identity facts with LLM, and update identity memory.
    Call every 5 turns to create/arrange identity facts. Returns number of facts written.
    """
    coll = db["chat_messages"]
    cursor = (
        coll.find({"username": username}, projection={"role": 1, "message": 1})
        .sort("timestamp", -1)
        .limit(last_n_turns * 2)  # user + ai turns
    )
    docs = await cursor.to_list(length=last_n_turns * 2)
    messages = list(reversed(docs))  # chronological
    if not messages:
        return 0
    facts = await extract_identity_facts_with_llm(messages, openai_api_key, openai_model)
    if not facts:
        return 0
    source = messages[-1].get("message", "")[:200] if messages else ""
    await update_identity_memory(db, username, facts, source_message=source)
    return len(facts)


async def get_identity_memory(
    db: AsyncIOMotorDatabase,
    username: str,
    keys: list[str] | None = None
) -> dict[str, IdentityFact]:
    """Retrieve identity memory facts"""
    coll = db["identity_memory"]
    query = {"username": username}
    if keys:
        query["key"] = {"$in": keys}
    
    cursor = coll.find(query)
    docs = await cursor.to_list(length=100)
    
    return {
        doc["key"]: IdentityFact(
            key=doc["key"],
            value=doc["value"],
            confidence=doc.get("confidence", 1.0),
            source_message=doc.get("source_message", ""),
            updated_at=doc.get("updated_at", datetime.now(timezone.utc))
        )
        for doc in docs
    }


async def get_working_memory(
    db: AsyncIOMotorDatabase,
    username: str,
    max_turns: int = 10
) -> list[dict[str, Any]]:
    """Get last N conversation turns"""
    coll = db["chat_messages"]
    cursor = (
        coll.find({"username": username}, projection={"_id": 0, "role": 1, "message": 1, "timestamp": 1})
        .sort("timestamp", -1)
        .limit(max_turns)
    )
    recent = await cursor.to_list(length=max_turns)
    return list(reversed(recent))  # Return in chronological order


async def trim_working_memory_if_needed(
    db: AsyncIOMotorDatabase,
    username: str,
    max_tokens: int = 2000,
    openai_api_key: str | None = None,
    openai_model: str = "gpt-4o-mini"
) -> list[dict[str, Any]]:
    """
    Get working memory, trimming if token count exceeds limit.
    If overflow, use LLM to summarize older messages and store as episodic memory.
    """
    messages = await get_working_memory(db, username, max_turns=20)
    total_chars = sum(len(m.get("message", "")) for m in messages)
    estimated_tokens = total_chars // 4

    if estimated_tokens <= max_tokens:
        return messages[-10:]  # last 10 (most recent)

    keep_messages = messages[-6:]
    older_messages = messages[:-6]
    if not older_messages:
        return keep_messages
    conversation_text = "\n".join(
        f"{m.get('role', 'unknown')}: {m.get('message', '')}"
        for m in older_messages
    )
    prompt = f"""Summarize this conversation in 2-4 short sentences. Capture the main topics, decisions, or feelings. Be concise.

Conversation:
{conversation_text[:4000]}

Summary:"""
    summary = await _llm_chat(prompt, openai_api_key, openai_model, timeout=15.0)
    if not summary:
        summary = f"Previous conversation: {len(older_messages)} messages."
    event_embedding = None
    if openai_api_key:
        event_embedding = await generate_embedding(summary, openai_api_key, "text-embedding-3-small")
    await create_episodic_memory(
        db, username, summary, importance_score=0.4,
        embedding=event_embedding,
        metadata={"source": "working_memory_trim", "message_count": len(older_messages)}
    )
    return keep_messages


@dataclass
class EpisodicMemory:
    """Important event memory"""
    event_summary: str
    importance_score: float  # 0.0 to 1.0
    timestamp: datetime
    access_count: int = 0
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


async def create_episodic_memory(
    db: AsyncIOMotorDatabase,
    username: str,
    event_summary: str,
    importance_score: float,
    embedding: list[float] | None = None,
    metadata: dict[str, Any] | None = None
) -> str:
    """Create a new episodic memory record"""
    coll = db["episodic_memory"]
    now = datetime.now(timezone.utc)
    
    doc = {
        "username": username,
        "event_summary": event_summary,
        "importance_score": importance_score,
        "timestamp": now,
        "access_count": 0,
        "embedding": embedding,
        "metadata": metadata or {},
    }
    
    result = await coll.insert_one(doc)
    return str(result.inserted_id)


async def search_episodic_memory(
    db: AsyncIOMotorDatabase,
    username: str,
    query_embedding: list[float],
    top_k: int = 5
) -> list[EpisodicMemory]:
    """
    Search episodic memories by similarity.
    Returns top K memories ranked by similarity + importance.
    """
    coll = db["episodic_memory"]
    cursor = coll.find({"username": username, "embedding": {"$ne": None}})
    all_memories = await cursor.to_list(length=1000)
    
    if not all_memories:
        return []
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    scored = []
    for doc in all_memories:
        embedding = doc.get("embedding")
        if not embedding:
            continue
        
        similarity = cosine_similarity(query_embedding, embedding)
        importance = doc.get("importance_score", 0.5)
        score = (similarity * 0.6) + (importance * 0.4)
        
        scored.append((score, doc))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, doc in scored[:top_k]:
        memory = EpisodicMemory(
            event_summary=doc["event_summary"],
            importance_score=doc.get("importance_score", 0.5),
            timestamp=doc.get("timestamp", datetime.now(timezone.utc)),
            access_count=doc.get("access_count", 0),
            embedding=doc.get("embedding"),
            metadata=doc.get("metadata", {})
        )
        results.append(memory)
        await coll.update_one(
            {"_id": doc["_id"]},
            {
                "$inc": {"access_count": 1},
                "$set": {
                    "importance_score": min(1.0, doc.get("importance_score", 0.5) + 0.05)
                }
            }
        )
    
    return results


async def apply_time_decay_to_episodic_memory(
    db: AsyncIOMotorDatabase,
    username: str,
    decay_rate: float = 0.01  # Reduce importance by 1% per day
) -> None:
    """Apply time-based decay to episodic memory importance scores"""
    coll = db["episodic_memory"]
    now = datetime.now(timezone.utc)
    
    cursor = coll.find({"username": username})
    async for doc in cursor:
        timestamp = doc.get("timestamp", now)
        days_old = (now - timestamp).days
        
        if days_old > 0:
            new_importance = max(0.0, doc.get("importance_score", 0.5) - (decay_rate * days_old))
            await coll.update_one(
                {"_id": doc["_id"]},
                {"$set": {"importance_score": new_importance}}
            )


async def get_semantic_profile(
    db: AsyncIOMotorDatabase,
    username: str,
    openai_api_key: str | None = None,
    openai_model: str = "gpt-4o-mini"
) -> dict[str, Any]:
    """Get or generate semantic profile from episodic memory"""
    coll = db["semantic_profile"]
    profile = await coll.find_one({"username": username})
    if profile:
        return profile.get("profile", {})
    return await generate_semantic_profile(db, username, openai_api_key, openai_model)


async def generate_semantic_profile(
    db: AsyncIOMotorDatabase,
    username: str,
    openai_api_key: str | None = None,
    openai_model: str = "gpt-4o-mini"
) -> dict[str, Any]:
    """Generate semantic profile by analyzing episodic memory with LLM."""
    coll_episodic = db["episodic_memory"]
    coll_profile = db["semantic_profile"]
    cursor = coll_episodic.find({"username": username}).sort("importance_score", -1).limit(20)
    memories = await cursor.to_list(length=20)

    if not memories:
        profile = {
            "personality_summary": "No significant patterns detected yet.",
            "preferences": {},
            "behavior_patterns": [],
            "updated_at": datetime.now(timezone.utc)
        }
    else:
        summaries = [m.get("event_summary", "") for m in memories[:15]]
        memories_text = "\n".join(f"- {s}" for s in summaries if s)

        if openai_api_key and memories_text:
            prompt = f"""Based on these remembered events about the user, produce a short profile in JSON only.

Events:
{memories_text[:3500]}

Return a single JSON object with exactly these keys (no other text):
- "personality_summary": 2-4 sentences describing the user's style, recurring topics, and how they tend to interact.
- "preferences": object with a few key preferences if evident (e.g. "communication_style": "...", "topics": ["..."]), or empty {{}}.
- "behavior_patterns": array of 3-6 short strings (e.g. "Often shares personal updates", "Uses casual language").

Example format:
{{"personality_summary": "...", "preferences": {{}}, "behavior_patterns": ["...", "..."]}}
"""
            raw = await _llm_chat(prompt, openai_api_key, openai_model, timeout=25.0)
            profile = _parse_semantic_profile_json(raw, len(memories), summaries)
        else:
            profile = _fallback_semantic_profile(memories, summaries)

        profile["updated_at"] = datetime.now(timezone.utc)

    await coll_profile.update_one(
        {"username": username},
        {
            "$set": {
                "username": username,
                "profile": profile,
                "updated_at": datetime.now(timezone.utc)
            }
        },
        upsert=True
    )
    return profile


def _parse_semantic_profile_json(raw: str | None, memory_count: int, summaries: list[str]) -> dict[str, Any]:
    """Parse LLM JSON response into semantic profile; fallback on invalid/missing."""
    if not raw:
        return _fallback_semantic_profile(
            [{"event_summary": s} for s in summaries[:10]], summaries
        )
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return _fallback_semantic_profile(
            [{"event_summary": s} for s in summaries[:10]], summaries
        )
    try:
        data = json.loads(match.group(0))
        personality = (data.get("personality_summary") or "").strip()
        if not personality:
            raise ValueError("missing personality_summary")
        prefs = data.get("preferences")
        if not isinstance(prefs, dict):
            prefs = {}
        patterns = data.get("behavior_patterns")
        if not isinstance(patterns, list):
            patterns = [p for p in (patterns or "").split(",") if p.strip()][:6]
        return {
            "personality_summary": personality[:800],
            "preferences": {str(k): str(v) for k, v in list(prefs.items())[:10]},
            "behavior_patterns": [str(p).strip()[:200] for p in patterns[:8]],
            "updated_at": datetime.now(timezone.utc)
        }
    except (json.JSONDecodeError, ValueError):
        return _fallback_semantic_profile(
            [{"event_summary": s} for s in summaries[:10]], summaries
        )


def _fallback_semantic_profile(memories: list[dict], summaries: list[str]) -> dict[str, Any]:
    """Non-LLM fallback when API is off or parsing fails."""
    return {
        "personality_summary": f"User has {len(memories)} significant interactions. Recent themes: {', '.join(summaries[:3])}.",
        "preferences": {},
        "behavior_patterns": summaries[:5],
        "updated_at": datetime.now(timezone.utc)
    }


async def generate_embedding(
    text: str,
    openai_api_key: str | None = None,
    openai_model: str = "text-embedding-3-small"
) -> list[float] | None:
    """Generate embedding for text using OpenAI"""
    if not openai_api_key:
        return None
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)
        
        response = client.embeddings.create(
            model=openai_model,
            input=text
        )
        return response.data[0].embedding
    except Exception:
        return None


async def ensure_memory_indexes(db: AsyncIOMotorDatabase) -> None:
    """Create indexes for memory collections"""
    await db["identity_memory"].create_index([("username", 1), ("key", 1)], unique=True)
    await db["episodic_memory"].create_index([("username", 1), ("timestamp", -1)])
    await db["episodic_memory"].create_index([("username", 1), ("importance_score", -1)])
    await db["semantic_profile"].create_index([("username", 1)], unique=True)
