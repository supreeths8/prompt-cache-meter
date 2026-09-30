"""The HTTP surface: forwards every call unchanged and records a simulated bill for successful chat completions."""

import json
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from prompt_cache_meter.proxy.models import UpstreamUnavailableError
from prompt_cache_meter.proxy.ports import Recorder
from prompt_cache_meter.proxy.replies import StreamCollector, summarise_completion
from prompt_cache_meter.proxy.service import SimulationService
from prompt_cache_meter.proxy.upstream import Upstream, UpstreamStream

_ALL_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


def create_app(
    upstream: Upstream,
    service: SimulationService,
    recorder: Recorder,
    timer: Callable[[], float] = time.perf_counter,
) -> Starlette:
    async def chat_completions(request: Request) -> Response:
        body = await request.body()
        started = timer()
        try:
            if _is_stream(body):
                stream = await upstream.stream("POST", request.url.path, request.headers, body)
                return _stream_response(stream, _record_stream(body, started))
            response = await upstream.send("POST", request.url.path, request.headers, body)
        except UpstreamUnavailableError as error:
            return _bad_gateway(error)
        if _is_success(response.status_code):
            recorder.record(service.observe(body, summarise_completion(response.body), timer() - started, False))
        return Response(response.body, response.status_code, headers=response.headers)

    def _record_stream(body: bytes, started: float) -> Callable[[StreamCollector], None]:
        def record(collector: StreamCollector) -> None:
            recorder.record(service.observe(body, collector.summary(), timer() - started, True))

        return record

    async def passthrough(request: Request) -> Response:
        try:
            response = await upstream.send(request.method, _target(request), request.headers, await request.body())
        except UpstreamUnavailableError as error:
            return _bad_gateway(error)
        return Response(response.body, response.status_code, headers=response.headers)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        yield
        await upstream.close()

    routes = [
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/{path:path}", passthrough, methods=_ALL_METHODS),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


def _stream_response(stream: UpstreamStream, on_complete: Callable[[StreamCollector], None]) -> Response:
    if not _is_success(stream.status_code):
        return StreamingResponse(stream.chunks, stream.status_code, headers=stream.headers)
    return StreamingResponse(_tee(stream.chunks, on_complete), stream.status_code, headers=stream.headers)


async def _tee(chunks: AsyncIterator[bytes], on_complete: Callable[[StreamCollector], None]) -> AsyncIterator[bytes]:
    # Bytes go to the app untouched; the collector reads a copy. Record only once the stream finished cleanly.
    collector = StreamCollector()
    async for chunk in chunks:
        collector.feed(chunk)
        yield chunk
    on_complete(collector)


def _target(request: Request) -> str:
    query = request.url.query
    return f"{request.url.path}?{query}" if query else request.url.path


def _is_stream(body: bytes) -> bool:
    try:
        payload = json.loads(body)
    except ValueError:
        return False
    return isinstance(payload, dict) and payload.get("stream") is True


def _is_success(status_code: int) -> bool:
    return 200 <= status_code < 300


def _bad_gateway(error: UpstreamUnavailableError) -> JSONResponse:
    error_body = {"error": {"message": str(error), "type": "upstream_unavailable", "code": "bad_gateway"}}
    return JSONResponse(error_body, status_code=502)
