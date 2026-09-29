import json
from collections.abc import Iterable
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.table import Table

from prompt_cache_meter.scenarios import Row, ScenarioResult


def print_result(console: Console, result: ScenarioResult, is_summary_only: bool) -> None:
    console.rule(f"[bold]{result.name}[/bold]: {result.title}")
    for note in result.notes:
        console.print(f"[dim]{note}[/dim]")
    if not is_summary_only:
        console.print(_request_table(result.rows))
    console.print(_summary_table(result.rows))


def _request_table(rows: Iterable[Row]) -> Table:
    table = Table(title="Per request: simulated OpenAI bill | what Ollama reported", title_justify="left")
    for name in ("Variant", "Request", "Input", "Cached", "Written", "1x", "Hit", "Sim $"):
        table.add_column(name, justify="left" if name in ("Variant", "Request") else "right")
    for name in ("Ollama in", "Ollama cached", "Latency s", "Warnings"):
        table.add_column(name, justify="left" if name == "Warnings" else "right", style="cyan")
    for row in rows:
        table.add_row(*_bill_cells(row), *_ollama_cells(row))
    return table


def _bill_cells(row: Row) -> tuple[str, ...]:
    bill = row.bill
    return (
        row.variant,
        row.label,
        f"{bill.input_tokens:,}",
        f"{bill.cached_tokens:,}",
        f"{bill.written_tokens:,}",
        f"{bill.uncached_tokens:,}",
        f"{bill.hit_rate:.0%}",
        f"{bill.cost_usd:.6f}",
    )


def _ollama_cells(row: Row) -> tuple[str, ...]:
    warnings = "; ".join(row.warnings)
    if row.ollama is None:
        return ("-", "-", "-", warnings)
    return (
        f"{row.ollama.prompt_tokens:,}",
        f"{row.ollama.cached_tokens:,}",
        f"{row.ollama.latency_seconds:.2f}",
        warnings,
    )


def _summary_table(rows: tuple[Row, ...]) -> Table:
    table = Table(title="Summary by variant", title_justify="left")
    for name in ("Variant", "Requests", "Input", "Cached", "Written", "1x", "Hit", "Sim $", "Mean latency s"):
        table.add_column(name, justify="left" if name == "Variant" else "right")
    for variant in dict.fromkeys(row.variant for row in rows):
        table.add_row(*_summary_cells(variant, [row for row in rows if row.variant == variant]))
    return table


def _summary_cells(variant: str, rows: list[Row]) -> tuple[str, ...]:
    input_tokens = sum(row.bill.input_tokens for row in rows)
    cached = sum(row.bill.cached_tokens for row in rows)
    latencies = [row.ollama.latency_seconds for row in rows if row.ollama is not None]
    return (
        variant,
        str(len(rows)),
        f"{input_tokens:,}",
        f"{cached:,}",
        f"{sum(row.bill.written_tokens for row in rows):,}",
        f"{sum(row.bill.uncached_tokens for row in rows):,}",
        f"{cached / input_tokens:.0%}" if input_tokens else "-",
        f"{sum(row.bill.cost_usd for row in rows):.6f}",
        f"{mean(latencies):.2f}" if latencies else "-",
    )


def save_result(result: ScenarioResult, directory: Path) -> Path:
    """Write one JSON line per call: tokens (including cached), latency and model for both views."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.name}.jsonl"
    with path.open("w", encoding="utf-8") as file:
        for row in result.rows:
            file.write(json.dumps(_record(result.name, row)) + "\n")
    return path


def _record(scenario: str, row: Row) -> dict[str, object]:
    return {
        "scenario": scenario,
        "variant": row.variant,
        "label": row.label,
        "simulated": row.bill.model_dump(mode="json") | {"hit_rate": row.bill.hit_rate},
        "ollama": row.ollama.model_dump(mode="json") if row.ollama else None,
        "warnings": list(row.warnings),
    }
