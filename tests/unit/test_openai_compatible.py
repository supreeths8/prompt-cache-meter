import json
from collections.abc import Callable
from typing import Any

import httpx2 as httpx
import pytest
from openai import OpenAI

from prompt_cache_meter.llm.errors import ProviderResponseError, ProviderUnavailableError
from prompt_cache_meter.llm.models import ChatRequest, Message, Role
from prompt_cache_meter.llm.openai_compatible import OpenAICompatibleChatModel

Handler = Callable[[httpx.Request], httpx.Response]


def make_model(handler: Handler, ticks: tuple[float, float] = (10.0, 12.5)) -> OpenAICompatibleChatModel:
    client = OpenAI(
        api_key="test-key",
        base_url="http://provider.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    clock = iter(ticks)
    return OpenAICompatibleChatModel(client, default_model="default-model", clock=lambda: next(clock))


def completion_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "default-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi there"}}],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 7,
            "total_tokens": 127,
            "prompt_tokens_details": {"cached_tokens": 100},
        },
    }
    body.update(overrides)
    return body


def hello_request(model: str | None = None) -> ChatRequest:
    return ChatRequest(
        messages=(Message(role=Role.SYSTEM, content="be brief"), Message(role=Role.USER, content="hello")), model=model
    )


def test_complete_returns_text_usage_and_latency() -> None:
    model = make_model(lambda request: httpx.Response(200, json=completion_body()))

    response = model.complete(hello_request())

    assert response.text == "hi there"
    assert response.usage.input_tokens == 120
    assert response.usage.output_tokens == 7
    assert response.usage.cached_input_tokens == 100
    assert response.latency_seconds == pytest.approx(2.5)


def test_complete_sends_messages_and_default_model() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=completion_body())

    make_model(handler).complete(hello_request())

    assert sent[0]["model"] == "default-model"
    assert sent[0]["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}]


def test_complete_uses_model_from_request_over_default() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=completion_body())

    make_model(handler).complete(hello_request(model="other-model"))

    assert sent[0]["model"] == "other-model"


def test_cached_tokens_default_to_zero_when_provider_omits_details() -> None:
    usage = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    model = make_model(lambda request: httpx.Response(200, json=completion_body(usage=usage)))

    response = model.complete(hello_request())

    assert response.usage.cached_input_tokens == 0


def test_missing_usage_raises_response_error() -> None:
    model = make_model(lambda request: httpx.Response(200, json=completion_body(usage=None)))

    with pytest.raises(ProviderResponseError, match="usage"):
        model.complete(hello_request())


def test_empty_choices_raises_response_error() -> None:
    model = make_model(lambda request: httpx.Response(200, json=completion_body(choices=[])))

    with pytest.raises(ProviderResponseError, match="choices"):
        model.complete(hello_request())


def test_server_error_raises_response_error() -> None:
    model = make_model(lambda request: httpx.Response(500, json={"error": {"message": "boom"}}))

    with pytest.raises(ProviderResponseError):
        model.complete(hello_request())


def test_connection_failure_raises_unavailable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ProviderUnavailableError):
        make_model(handler).complete(hello_request())
