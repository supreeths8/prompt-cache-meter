"""Reads the reply text and usage out of upstream responses, streamed or not. Never changes the bytes."""

import json
from typing import Any

from pydantic import ValidationError

from prompt_cache_meter.proxy.models import ReplySummary, UpstreamUsage

_SSE_DATA = b"data:"
_SSE_DONE = b"[DONE]"


def summarise_completion(body: bytes) -> ReplySummary:
    payload = _load_json(body)
    choices = payload.get("choices") or []
    texts = [_message_text(choice.get("message") or {}) for choice in choices if isinstance(choice, dict)]
    return ReplySummary(text="".join(texts), usage=_usage(payload.get("usage")))


class StreamCollector:
    """Fed raw SSE bytes as they pass through; collects delta text, tool-call arguments and the usage chunk."""

    def __init__(self) -> None:
        self._buffer = b""
        self._parts: list[str] = []
        self._usage: UpstreamUsage | None = None

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        *lines, self._buffer = self._buffer.split(b"\n")
        for line in lines:
            self._read_line(line.strip())

    def summary(self) -> ReplySummary:
        self._read_line(self._buffer.strip())
        self._buffer = b""
        return ReplySummary(text="".join(self._parts), usage=self._usage)

    def _read_line(self, line: bytes) -> None:
        if not line.startswith(_SSE_DATA):
            return
        data = line.removeprefix(_SSE_DATA).strip()
        if not data or data == _SSE_DONE:
            return
        payload = _load_json(data)
        for choice in payload.get("choices") or []:
            if isinstance(choice, dict):
                self._parts.append(_message_text(choice.get("delta") or {}))
        self._usage = _usage(payload.get("usage")) or self._usage


def _load_json(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _message_text(message: dict[str, Any]) -> str:
    # Output tokens cover the reply text plus any tool-call names and arguments the model generated.
    parts = [message.get("content") or ""]
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {} if isinstance(call, dict) else {}
        parts.append(function.get("name") or "")
        parts.append(function.get("arguments") or "")
    return "".join(part for part in parts if isinstance(part, str))


def _usage(raw: object) -> UpstreamUsage | None:
    if not isinstance(raw, dict):
        return None
    details = raw.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    try:
        return UpstreamUsage(
            prompt_tokens=raw.get("prompt_tokens", 0),
            completion_tokens=raw.get("completion_tokens", 0),
            cached_tokens=cached or 0,
        )
    except ValidationError:
        return None
