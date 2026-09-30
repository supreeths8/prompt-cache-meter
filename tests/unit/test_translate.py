from typing import Any

import pytest

from prompt_cache_meter.proxy.models import ChatCompletionBody, TranslationError
from prompt_cache_meter.proxy.translate import Translation, parse_body, translate
from prompt_cache_meter.rules import Ttl
from prompt_cache_meter.simulator import SimRole

PRICED = {"gpt-5", "gpt-6.1-terra"}
NEW_TTLS = (Ttl.MINUTES_30, Ttl.HOURS_12)
LEGACY_TTLS = (Ttl.IN_MEMORY, Ttl.HOURS_24)


def body(**overrides: Any) -> ChatCompletionBody:
    fields: dict[str, Any] = {"model": "qwen2.5:14b", "messages": [{"role": "user", "content": "hi"}]}
    return ChatCompletionBody.model_validate(fields | overrides)


def run(chat_body: ChatCompletionBody, allowed_ttls: tuple[Ttl, ...] = NEW_TTLS) -> Translation:
    return translate(chat_body, PRICED, price_as="gpt-6.1-terra", allowed_ttls=allowed_ttls)


def test_roles_map_including_developer_and_tool() -> None:
    messages = [
        {"role": "developer", "content": "rules"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "f"}}]},
        {"role": "tool", "content": "42", "tool_call_id": "c1"},
    ]

    roles = [message.role for message in run(body(messages=messages)).request.messages]

    assert roles == [SimRole.DEVELOPER, SimRole.USER, SimRole.ASSISTANT, SimRole.TOOL]


def test_unknown_role_raises_translation_error() -> None:
    with pytest.raises(TranslationError, match="narrator"):
        run(body(messages=[{"role": "narrator", "content": "x"}]))


def test_text_parts_are_joined_without_warning() -> None:
    parts = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]

    translation = run(body(messages=[{"role": "user", "content": parts}]))

    assert translation.request.messages[0].content == "hello\nworld"
    assert translation.warnings == ()


def test_image_part_is_rendered_and_warned_about() -> None:
    parts = [{"type": "image_url", "image_url": {"url": "data:a"}}, {"type": "image_url", "image_url": {"url": "b"}}]

    translation = run(body(messages=[{"role": "user", "content": parts}]))

    assert "data:a" in translation.request.messages[0].content
    assert translation.warnings == ("non-text content part 'image_url' counted approximately",)


def test_tool_calls_are_part_of_the_rendered_message() -> None:
    first = {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "lookup", "arguments": "{}"}}]}
    second = {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "search", "arguments": "{}"}}]}

    one = run(body(messages=[first])).request.messages[0].content
    two = run(body(messages=[second])).request.messages[0].content

    assert "lookup" in one
    assert one != two


def test_prefix_settings_are_carried_over() -> None:
    chat_body = body(
        tools=[{"type": "function", "function": {"name": "f"}}],
        response_format={"type": "json_object"},
        reasoning_effort="low",
        parallel_tool_calls=False,
    )

    request = run(chat_body).request

    assert request.tools == ({"type": "function", "function": {"name": "f"}},)
    assert request.response_schema == {"type": "json_object"}
    assert request.prefix_options == {"reasoning_effort": "low", "parallel_tool_calls": False}


def test_prompt_cache_key_passes_through() -> None:
    assert run(body(prompt_cache_key="tenant-a")).request.prompt_cache_key == "tenant-a"


def test_legacy_retention_maps_to_ttl() -> None:
    request = run(body(prompt_cache_retention="24h"), allowed_ttls=LEGACY_TTLS).request

    assert request.ttl is Ttl.HOURS_24


def test_retention_unsupported_by_rules_warns_and_uses_default() -> None:
    translation = run(body(prompt_cache_retention="24h"), allowed_ttls=NEW_TTLS)

    assert translation.request.ttl is None
    assert "prompt_cache_retention" in translation.warnings[0]


def test_unpriced_model_is_priced_as_fallback() -> None:
    assert run(body(model="qwen2.5:14b")).request.model == "gpt-6.1-terra"


def test_priced_model_keeps_its_own_price() -> None:
    assert run(body(model="gpt-5")).request.model == "gpt-5"


def test_body_without_messages_raises_translation_error() -> None:
    with pytest.raises(TranslationError):
        parse_body(b'{"model": "x"}')
