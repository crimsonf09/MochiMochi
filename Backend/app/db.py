from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase


def get_db(client: AsyncIOMotorClient, db_name: str) -> AsyncIOMotorDatabase:
    return client[db_name]


async def ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    """Create indexes for all collections"""
    coll = db["chat_messages"]
    await coll.create_index([("username", 1), ("timestamp", 1)])
    await coll.create_index([("username", 1), ("timestamp", -1)])
    from .memory import ensure_memory_indexes
    await ensure_memory_indexes(db)
    security_coll = db["security_events"]
    await security_coll.create_index([("username", 1), ("timestamp", -1)])
    await security_coll.create_index([("risk_level", 1), ("timestamp", -1)])

