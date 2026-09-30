from typing import Protocol

from prompt_cache_meter.proxy.models import ProxyRecord


class Recorder(Protocol):
    def record(self, record: ProxyRecord) -> None:
        """Show or store one proxied call."""
        ...
