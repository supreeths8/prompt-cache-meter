from typing import Protocol

from prompt_cache_meter.llm.models import ChatRequest, ChatResponse


class ChatModel(Protocol):
    def complete(self, request: ChatRequest) -> ChatResponse:
        """Send one chat request and return the reply with usage and latency.

        Raises an `LLMError` subclass on failure, never a provider-specific exception.
        """
        ...


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        """Turn text into token ids the way the target provider would count them.

        Raises a `TokenizerError` on failure.
        """
        ...
