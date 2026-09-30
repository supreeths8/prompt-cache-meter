from pathlib import Path

from rich.console import Console
from rich.markup import escape

from prompt_cache_meter.proxy.models import ProxyRecord


class ConsoleJsonlRecorder:
    """Prints one line per call and appends the full record to a daily JSONL file."""

    def __init__(self, console: Console, directory: Path) -> None:
        self._console = console
        self._directory = directory

    def record(self, record: ProxyRecord) -> None:
        self._console.print(format_line(record))
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{record.time:%Y%m%d}.jsonl"
        with path.open("a", encoding="utf-8") as file:
            file.write(record.model_dump_json() + "\n")


def format_line(record: ProxyRecord) -> str:
    head = f"{record.time:%H:%M:%S} {record.requested_model or '?'} as {record.priced_model or '-'}"
    head += f" key={record.prompt_cache_key or '-'}"
    parts = [escape(head), _bill_text(record), _upstream_text(record), f"{record.latency_seconds:.2f}s"]
    line = " | ".join(parts)
    if record.warnings:
        line += f" [yellow]{escape('; '.join(record.warnings))}[/yellow]"
    return line


def _bill_text(record: ProxyRecord) -> str:
    bill = record.bill
    if bill is None:
        return "not simulated"
    return (
        f"in {bill.input_tokens:,} cached {bill.cached_tokens:,} written {bill.written_tokens:,}"
        f" 1x {bill.uncached_tokens:,} hit {bill.hit_rate:.0%} ${bill.cost_usd:.6f}"
    )


def _upstream_text(record: ProxyRecord) -> str:
    usage = record.upstream_usage
    if usage is None:
        return "upstream usage -"
    return f"upstream in {usage.prompt_tokens:,} cached {usage.cached_tokens:,} out {usage.completion_tokens:,}"
