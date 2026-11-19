"""Public request and response schemas."""

from typing import Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    model: str | None = None


class ChatResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    model: str
    provider: str
    content: str
    finish_reason: Literal["stop"] = "stop"


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    provider: str
    model: str
