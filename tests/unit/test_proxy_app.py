import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx2 as httpx

from prompt_cache_meter.proxy.app import create_app
from prompt_cache_meter.proxy.models import ProxyRecord
from prompt_cache_meter.proxy.service import SimulationService
from prompt_cache_meter.proxy.upstream import Upstream
from prompt_cache_meter.rules import Generation
from prompt_cache_meter.simulator import FakeClock
from tests.unit.simulator_fixtures import WordTokenizer, make_rules

Handler = Callable[[httpx.Request], httpx.Response]

SYSTEM_PROMPT = " ".join(f"rule{index}" for index in range(2000))
COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "model": "local",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "four words of reply"}}],
    "usage": {"prompt_tokens": 2100, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 2048}},
}


class ListRecorder:
    def __init__(self) -> None:
        self.records: list[ProxyRecord] = []

    def record(self, record: ProxyRecord) -> None:
        self.records.append(record)


def chat_body(**overrides: Any) -> dict[str, Any]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "what is rule 7?"}]
    return {"model": "local", "messages": messages} | overrides


def call(
    handler: Handler, *requests: tuple[str, str, dict[str, Any] | None]
) -> tuple[list[httpx.Response], ListRecorder]:
    """Send requests through a proxy whose upstream is `handler`; return the responses and what was recorded."""
    recorder = ListRecorder()
    service = SimulationService(make_rules(), Generation.NEW, WordTokenizer(), FakeClock(), price_as="new-model")
    upstream = Upstream("http://upstream.test/v1", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    app = create_app(upstream, service, recorder)

    async def send_all() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test") as client:
            headers = {"Authorization": "Bearer secret-key"}
            return [
                await client.request(method, path, json=payload, headers=headers) for method, path, payload in requests
            ]

    return asyncio.run(send_all()), recorder


def post(payload: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    return ("POST", "/v1/chat/completions", payload)


def test_response_body_is_returned_unchanged() -> None:
    raw = json.dumps(COMPLETION).encode()

    responses, _ = call(lambda request: httpx.Response(200, content=raw), post(chat_body()))

    assert responses[0].status_code == 200
    assert responses[0].content == raw


def test_second_identical_call_records_a_cache_hit() -> None:
    _, recorder = call(lambda request: httpx.Response(200, json=COMPLETION), post(chat_body()), post(chat_body()))

    first, second = (record.bill for record in recorder.records)
    assert first is not None and second is not None
    assert first.cached_tokens == 0
    assert second.cached_tokens == second.input_tokens
    assert recorder.records[1].upstream_usage is not None
    assert recorder.records[1].upstream_usage.cached_tokens == 2048


def test_unpriced_model_is_billed_at_fallback_price() -> None:
    _, recorder = call(lambda request: httpx.Response(200, json=COMPLETION), post(chat_body()))

    assert recorder.records[0].requested_model == "local"
    assert recorder.records[0].priced_model == "new-model"


def test_stream_is_passed_through_and_output_counted() -> None:
    events = [
        {"choices": [{"delta": {"content": "two words"}}]},
        {"choices": [{"delta": {"content": " more"}}]},
        {"choices": [], "usage": {"prompt_tokens": 2100, "completion_tokens": 3}},
    ]
    stream = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events) + b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})

    responses, recorder = call(handler, post(chat_body(stream=True)))

    record = recorder.records[0]
    assert responses[0].content == stream
    assert record.is_stream
    assert record.bill is not None and record.bill.output_tokens == 3
    assert record.upstream_usage is not None and record.upstream_usage.completion_tokens == 3


def test_upstream_error_passes_through_without_simulating() -> None:
    error = {"error": {"message": "bad model"}}

    responses, recorder = call(lambda request: httpx.Response(400, json=error), post(chat_body()))

    assert responses[0].status_code == 400
    assert responses[0].json() == error
    assert recorder.records == []


def test_unreachable_upstream_returns_502() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    responses, _ = call(handler, post(chat_body()))

    assert responses[0].status_code == 502
    assert responses[0].json()["error"]["code"] == "bad_gateway"


def test_unreadable_request_still_gets_upstream_response_with_warning() -> None:
    payload = chat_body(messages=[{"role": "narrator", "content": "x"}])

    responses, recorder = call(lambda request: httpx.Response(200, json=COMPLETION), post(payload))

    assert responses[0].status_code == 200
    assert recorder.records[0].bill is None
    assert "not simulated" in recorder.records[0].warnings[0]


def test_authorization_is_forwarded_but_never_recorded() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=COMPLETION)

    _, recorder = call(handler, post(chat_body()))

    assert seen[0].headers["authorization"] == "Bearer secret-key"
    assert "secret-key" not in recorder.records[0].model_dump_json()


def test_other_endpoints_pass_through_with_query() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": []})

    responses, recorder = call(handler, ("GET", "/v1/models?limit=5", None))

    assert responses[0].json() == {"data": []}
    assert str(seen[0].url) == "http://upstream.test/v1/models?limit=5"
    assert recorder.records == []
