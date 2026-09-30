from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, JsonValue, NonNegativeFloat, NonNegativeInt

from prompt_cache_meter.rules import Generation
from prompt_cache_meter.simulator import Bill


class ProxyError(Exception):
    """Base class for proxy failures."""


class TranslationError(ProxyError):
    """The request body is not a Chat Completions request the simulator can read."""


class UpstreamUnavailableError(ProxyError):
    """The upstream provider could not be reached."""


class ChatMessageIn(BaseModel):
    # Unknown fields are allowed: the proxy forwards the raw body, this model only reads what the simulator needs.
    model_config = ConfigDict(frozen=True, extra="allow")

    role: str
    content: str | list[dict[str, JsonValue]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, JsonValue]] | None = None


class ChatCompletionBody(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    model: str
    messages: list[ChatMessageIn] = Field(min_length=1)
    stream: bool = False
    tools: list[dict[str, JsonValue]] | None = None
    response_format: dict[str, JsonValue] | None = None
    reasoning_effort: str | None = None
    verbosity: str | None = None
    parallel_tool_calls: bool | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None


class UpstreamUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: NonNegativeInt
    completion_tokens: NonNegativeInt
    cached_tokens: NonNegativeInt = 0


class ReplySummary(BaseModel):
    """What the proxy collected from the upstream reply: the text it billed as output, and the upstream's usage."""

    model_config = ConfigDict(frozen=True)

    text: str
    usage: UpstreamUsage | None


class ProxyRecord(BaseModel):
    """One proxied call. Deliberately holds no message content."""

    model_config = ConfigDict(frozen=True)

    time: datetime
    requested_model: str | None
    priced_model: str | None
    generation: Generation
    prompt_cache_key: str | None
    is_stream: bool
    latency_seconds: NonNegativeFloat
    bill: Bill | None
    upstream_usage: UpstreamUsage | None
    warnings: tuple[str, ...]
