from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass

import httpx2 as httpx

from prompt_cache_meter.proxy.models import UpstreamUnavailableError

# Hop-by-hop and length headers belong to one connection; the next hop sets its own.
_DROPPED_REQUEST_HEADERS = frozenset(
    {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "upgrade", "accept-encoding"}
)
_DROPPED_RESPONSE_HEADERS = frozenset(
    {"content-length", "content-encoding", "connection", "keep-alive", "transfer-encoding"}
)


@dataclass(frozen=True)
class UpstreamResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes


@dataclass(frozen=True)
class UpstreamStream:
    status_code: int
    headers: dict[str, str]
    chunks: AsyncIterator[bytes]


class Upstream:
    """Forwards requests to one OpenAI-compatible provider. The only proxy code that touches the network."""

    def __init__(self, base_url: str, client: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client

    async def send(self, method: str, path: str, headers: Mapping[str, str], body: bytes) -> UpstreamResponse:
        try:
            response = await self._client.send(self._build(method, path, headers, body))
        except httpx.TransportError as error:
            raise UpstreamUnavailableError(f"{self._base_url}: {error}") from error
        return UpstreamResponse(response.status_code, _response_headers(response), response.content)

    async def stream(self, method: str, path: str, headers: Mapping[str, str], body: bytes) -> UpstreamStream:
        try:
            response = await self._client.send(self._build(method, path, headers, body), stream=True)
        except httpx.TransportError as error:
            raise UpstreamUnavailableError(f"{self._base_url}: {error}") from error
        return UpstreamStream(response.status_code, _response_headers(response), _chunks(response))

    async def close(self) -> None:
        await self._client.aclose()

    def _build(self, method: str, path: str, headers: Mapping[str, str], body: bytes) -> httpx.Request:
        # The app calls the proxy at <proxy>/v1/...; the upstream base URL already ends in its own version prefix.
        url = self._base_url + path.removeprefix("/v1")
        forwarded = {key: value for key, value in headers.items() if key.lower() not in _DROPPED_REQUEST_HEADERS}
        # Ask for an uncompressed reply so streamed bytes can be parsed and passed on unchanged.
        forwarded["accept-encoding"] = "identity"
        return self._client.build_request(method, url, headers=forwarded, content=body)


async def _chunks(response: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {key: value for key, value in response.headers.items() if key.lower() not in _DROPPED_RESPONSE_HEADERS}
