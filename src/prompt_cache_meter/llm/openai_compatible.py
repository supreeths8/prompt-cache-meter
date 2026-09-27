import time
from collections.abc import Callable

import openai
from openai import OpenAI
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from prompt_cache_meter.llm.errors import ProviderResponseError, ProviderUnavailableError
from prompt_cache_meter.llm.models import ChatRequest, ChatResponse, Message, Role, Usage


class OpenAICompatibleChatModel:
    """Chat model for any provider that speaks the OpenAI chat completions API (Ollama, OpenRouter)."""

    def __init__(self, client: OpenAI, default_model: str, clock: Callable[[], float] = time.perf_counter) -> None:
        self._client = client
        self._default_model = default_model
        self._clock = clock

    def complete(self, request: ChatRequest) -> ChatResponse:
        started = self._clock()
        try:
            completion = self._client.chat.completions.create(
                model=request.model or self._default_model,
                messages=[_to_openai_message(message) for message in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except openai.APIConnectionError as error:
            raise ProviderUnavailableError(str(error)) from error
        except openai.APIError as error:
            raise ProviderResponseError(str(error)) from error
        return _to_response(completion, latency_seconds=self._clock() - started)


def _to_openai_message(message: Message) -> ChatCompletionMessageParam:
    match message.role:
        case Role.SYSTEM:
            return ChatCompletionSystemMessageParam(role="system", content=message.content)
        case Role.USER:
            return ChatCompletionUserMessageParam(role="user", content=message.content)
        case Role.ASSISTANT:
            return ChatCompletionAssistantMessageParam(role="assistant", content=message.content)


def _to_response(completion: ChatCompletion, latency_seconds: float) -> ChatResponse:
    choice = next(iter(completion.choices), None)
    if choice is None:
        raise ProviderResponseError("response had no choices")
    if completion.usage is None:
        raise ProviderResponseError("response had no token usage")
    return ChatResponse(
        text=choice.message.content or "",
        model=completion.model,
        usage=_to_usage(completion.usage),
        latency_seconds=latency_seconds,
    )


def _to_usage(usage: CompletionUsage) -> Usage:
    details = usage.prompt_tokens_details
    cached = details.cached_tokens if details and details.cached_tokens else 0
    return Usage(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens, cached_input_tokens=cached)
