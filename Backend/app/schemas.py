from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


Role = Literal["user", "ai"]


class ChatMessage(BaseModel):
    id: str = Field(..., description="Mongo ObjectId as string")
    username: str
    role: Role
    message: str
    emotion_score: int
    emotion_label: str
    timestamp: datetime


class ChatState(BaseModel):
    username: str
    affection_score: int
    persona_stage: str


class WsClientMessage(BaseModel):
    message: str


class WsServerMessage(BaseModel):
    role: Literal["ai"] = "ai"
    message: str
    emotion_score: int
    emotion_label: str
    timestamp: datetime


class IdentityFact(BaseModel):
    key: str
    value: str
    confidence: float


class EpisodicMemory(BaseModel):
    event_summary: str
    importance_score: float
    timestamp: datetime
    access_count: int


class SemanticProfile(BaseModel):
    personality_summary: str
    preferences: dict
    behavior_patterns: list[str]


class MemoryData(BaseModel):
    identity_facts: list[IdentityFact]
    episodic_memories: list[EpisodicMemory]
    semantic_profile: SemanticProfile
    working_memory_count: int

