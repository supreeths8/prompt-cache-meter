from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt, PositiveInt


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: tuple[Message, ...] = Field(min_length=1)
    model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: PositiveInt = 1024


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    cached_input_tokens: NonNegativeInt = 0


class ChatResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    usage: Usage
    latency_seconds: NonNegativeFloat
