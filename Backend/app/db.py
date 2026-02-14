from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase


def get_db(client: AsyncIOMotorClient, db_name: str) -> AsyncIOMotorDatabase:
    return client[db_name]


async def ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    """Create indexes for all collections"""
    # Chat messages indexes
    coll = db["chat_messages"]
    await coll.create_index([("username", 1), ("timestamp", 1)])
    await coll.create_index([("username", 1), ("timestamp", -1)])
    
    # Memory system indexes (imported from memory module)
    from .memory import ensure_memory_indexes
    await ensure_memory_indexes(db)

