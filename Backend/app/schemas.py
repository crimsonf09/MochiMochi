from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


Role = Literal["user", "ai"]


class Emotion3D(BaseModel):
    valence: float = Field(..., ge=-1.0, le=1.0, description="Emotional positivity/negativity (-1.0 to 1.0)")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Emotional intensity (0.0 to 1.0)")
    dominance: float = Field(..., ge=0.0, le=1.0, description="Sense of control/power (0.0 to 1.0)")
    impact: float = Field(..., ge=0.0, le=1.0, description="Message impact/significance (0.0 to 1.0)")


class ChatMessage(BaseModel):
    id: str = Field(..., description="Mongo ObjectId as string")
    username: str
    role: Role
    message: str
    emotion_score: int
    emotion_label: Optional[str] = ""  # Deprecated, kept for backward compatibility
    emotion_3d: Optional[Emotion3D] = None
    timestamp: datetime


class ChatState(BaseModel):
    username: str
    affection_score: int  # 0-10
    persona_stage: str = ""  # Deprecated, kept for backward compatibility


class WsClientMessage(BaseModel):
    message: str


class WsServerMessage(BaseModel):
    role: Literal["ai"] = "ai"
    message: str
    emotion_score: int
    weighted_score: Optional[float] = None  # Weighted score before capping (for display)
    emotion_label: str = ""  # Deprecated, kept for backward compatibility
    emotion_3d: Optional[Emotion3D] = None
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

