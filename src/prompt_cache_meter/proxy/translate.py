"""Turns an OpenAI Chat Completions request body into a SimRequest. Pure: no I/O."""

import json
from collections.abc import Collection
from dataclasses import dataclass

from pydantic import JsonValue, ValidationError

from prompt_cache_meter.proxy.models import ChatCompletionBody, ChatMessageIn, TranslationError
from prompt_cache_meter.rules import Ttl
from prompt_cache_meter.simulator import SimMessage, SimRequest, SimRole

# "function" is the pre-tools name for a tool response.
_ROLES = {role.value: role for role in SimRole} | {"function": SimRole.TOOL}
_RETENTION_TTLS = {"in_memory": Ttl.IN_MEMORY, "24h": Ttl.HOURS_24}


@dataclass(frozen=True)
class Translation:
    body: ChatCompletionBody
    request: SimRequest
    warnings: tuple[str, ...]


def parse_body(raw: bytes) -> ChatCompletionBody:
    try:
        return ChatCompletionBody.model_validate_json(raw)
    except ValidationError as error:
        raise TranslationError(f"not a chat completions request: {error.error_count()} validation errors") from error


def translate(
    body: ChatCompletionBody, priced_models: Collection[str], price_as: str, allowed_ttls: Collection[Ttl]
) -> Translation:
    warnings: list[str] = []
    messages = tuple(_to_sim_message(message, warnings) for message in body.messages)
    request = SimRequest(
        model=body.model if body.model in priced_models else price_as,
        messages=messages,
        tools=tuple(body.tools or ()),
        response_schema=body.response_format,
        prefix_options=_prefix_options(body),
        prompt_cache_key=body.prompt_cache_key or "",
        ttl=_ttl(body.prompt_cache_retention, allowed_ttls, warnings),
    )
    return Translation(body=body, request=request, warnings=tuple(warnings))


def _to_sim_message(message: ChatMessageIn, warnings: list[str]) -> SimMessage:
    role = _ROLES.get(message.role)
    if role is None:
        raise TranslationError(f"unknown message role {message.role!r}")
    return SimMessage(role=role, content=_render_content(message, warnings))


def _render_content(message: ChatMessageIn, warnings: list[str]) -> str:
    text = _render_parts(message.content, warnings)
    # Tool call ids, names and calls are part of what OpenAI renders, so a change to any of them changes the prefix.
    extras = {
        key: value
        for key, value in (
            ("name", message.name),
            ("tool_call_id", message.tool_call_id),
            ("tool_calls", message.tool_calls),
        )
        if value is not None
    }
    return f"{text}\n{json.dumps(extras, sort_keys=True)}" if extras else text


def _render_parts(content: str | list[dict[str, JsonValue]] | None, warnings: list[str]) -> str:
    if content is None or isinstance(content, str):
        return content or ""
    rendered = []
    for part in content:
        text = part.get("text")
        if part.get("type") == "text" and isinstance(text, str):
            rendered.append(text)
            continue
        # Real image and audio token counts depend on the model; JSON keeps a changed part a cache miss.
        rendered.append(json.dumps(part, sort_keys=True))
        _warn_once(warnings, f"non-text content part {part.get('type')!r} counted approximately")
    return "\n".join(rendered)


def _prefix_options(body: ChatCompletionBody) -> dict[str, JsonValue]:
    options: dict[str, JsonValue] = {
        "reasoning_effort": body.reasoning_effort,
        "verbosity": body.verbosity,
        "parallel_tool_calls": body.parallel_tool_calls,
    }
    return {key: value for key, value in options.items() if value is not None}


def _ttl(retention: str | None, allowed_ttls: Collection[Ttl], warnings: list[str]) -> Ttl | None:
    if retention is None:
        return None
    ttl = _RETENTION_TTLS.get(retention)
    if ttl is None or ttl not in allowed_ttls:
        warnings.append(f"prompt_cache_retention {retention!r} not supported by these rules; using the default TTL")
        return None
    return ttl


def _warn_once(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)
