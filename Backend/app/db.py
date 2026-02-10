from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase


def get_db(client: AsyncIOMotorClient, db_name: str) -> AsyncIOMotorDatabase:
    return client[db_name]


async def ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    coll = db["chat_messages"]
    # Query patterns:
    # - history by username sorted by timestamp
    # - latest state by username sorted by timestamp
    await coll.create_index([("username", 1), ("timestamp", 1)])
    await coll.create_index([("username", 1), ("timestamp", -1)])

