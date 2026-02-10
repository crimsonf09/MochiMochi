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

