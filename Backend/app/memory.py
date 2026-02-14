"""
Memory System for Chatbot
Implements Identity Memory, Working Memory, Episodic Memory, and Semantic Profile
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase


# ============================================================================
# Identity Memory (Structured Facts)
# ============================================================================

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
            # Insert new fact
            await coll.insert_one({
                "username": username,
                "key": key,
                "value": value,
                "confidence": confidence,
                "source_message": source_message,
                "updated_at": now,
            })


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


# ============================================================================
# Working Memory (Short-Term Context)
# ============================================================================

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
    max_tokens: int = 2000
) -> list[dict[str, Any]]:
    """
    Get working memory, trimming if token count exceeds limit.
    If overflow, summarize older messages and move to episodic memory.
    """
    messages = await get_working_memory(db, username, max_turns=20)  # Get more to check
    
    # Simple token estimation (rough: 1 token ≈ 4 characters)
    total_chars = sum(len(m.get("message", "")) for m in messages)
    estimated_tokens = total_chars // 4
    
    if estimated_tokens <= max_tokens:
        return messages[:10]  # Return last 10
    
    # Need to trim - keep last 6, summarize older ones
    keep_messages = messages[-6:]
    older_messages = messages[:-6]
    
    if older_messages:
        # Summarize older messages (simple concatenation for now)
        # In production, you'd use LLM to summarize
        summary = f"Previous conversation: {len(older_messages)} messages about various topics."
        
        # Store summary in episodic memory (will be created by episodic memory system)
        # For now, just return the kept messages
        pass
    
    return keep_messages


# ============================================================================
# Episodic Memory (Event-Based Long-Term)
# ============================================================================

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
    
    # Get all memories for user
    cursor = coll.find({"username": username, "embedding": {"$ne": None}})
    all_memories = await cursor.to_list(length=1000)
    
    if not all_memories:
        return []
    
    # Calculate cosine similarity
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    
    # Score each memory: similarity * importance
    scored = []
    for doc in all_memories:
        embedding = doc.get("embedding")
        if not embedding:
            continue
        
        similarity = cosine_similarity(query_embedding, embedding)
        importance = doc.get("importance_score", 0.5)
        
        # Combined score: weighted average
        score = (similarity * 0.6) + (importance * 0.4)
        
        scored.append((score, doc))
    
    # Sort by score and return top K
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
        
        # Increment access count and slightly boost importance
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
    
    # Calculate days since last update
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


# ============================================================================
# Semantic Profile (Behavior Patterns)
# ============================================================================

async def get_semantic_profile(
    db: AsyncIOMotorDatabase,
    username: str
) -> dict[str, Any]:
    """Get or generate semantic profile from episodic memory"""
    coll = db["semantic_profile"]
    
    profile = await coll.find_one({"username": username})
    if profile:
        return profile.get("profile", {})
    
    # Generate from episodic memory if doesn't exist
    return await generate_semantic_profile(db, username)


async def generate_semantic_profile(
    db: AsyncIOMotorDatabase,
    username: str
) -> dict[str, Any]:
    """Generate semantic profile by analyzing episodic memory"""
    coll_episodic = db["episodic_memory"]
    coll_profile = db["semantic_profile"]
    
    # Get all episodic memories
    cursor = coll_episodic.find({"username": username}).sort("importance_score", -1).limit(20)
    memories = await cursor.to_list(length=20)
    
    if not memories:
        # Return empty profile
        profile = {
            "personality_summary": "No significant patterns detected yet.",
            "preferences": {},
            "behavior_patterns": [],
            "updated_at": datetime.now(timezone.utc)
        }
    else:
        # Simple extraction (in production, use LLM to analyze)
        summaries = [m.get("event_summary", "") for m in memories[:10]]
        profile = {
            "personality_summary": f"User has {len(memories)} significant interactions. Recent themes: {', '.join(summaries[:3])}.",
            "preferences": {},
            "behavior_patterns": summaries[:5],
            "updated_at": datetime.now(timezone.utc)
        }
    
    # Store profile
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


# ============================================================================
# Embedding Generation
# ============================================================================

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
    except Exception as e:
        print(f"[Memory] Failed to generate embedding: {e}")
        return None


# ============================================================================
# Memory Initialization
# ============================================================================

async def ensure_memory_indexes(db: AsyncIOMotorDatabase) -> None:
    """Create indexes for memory collections"""
    # Identity memory indexes
    await db["identity_memory"].create_index([("username", 1), ("key", 1)], unique=True)
    
    # Episodic memory indexes
    await db["episodic_memory"].create_index([("username", 1), ("timestamp", -1)])
    await db["episodic_memory"].create_index([("username", 1), ("importance_score", -1)])
    
    # Semantic profile indexes
    await db["semantic_profile"].create_index([("username", 1)], unique=True)
