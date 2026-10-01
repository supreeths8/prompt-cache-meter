from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich.console import Console

from prompt_cache_meter.bootstrap import build_chat_model, build_proxy_app, build_scenario_context
from prompt_cache_meter.config import Provider, Settings
from prompt_cache_meter.llm.errors import LLMError
from prompt_cache_meter.llm.models import ChatRequest, Message, Role
from prompt_cache_meter.report import print_result, save_result
from prompt_cache_meter.rules import DEFAULT_RULES_PATH, Generation, RulesError
from prompt_cache_meter.scenarios import SCENARIOS
from prompt_cache_meter.simulator import SimulatorError

app = typer.Typer(no_args_is_help=True)
console = Console()

ALL_SCENARIOS = "all"
DEFAULT_UPSTREAM = "http://127.0.0.1:11434/v1"


@app.callback()
def main() -> None:
    """Simulate OpenAI prompt caching and billing next to real Ollama runs."""


@app.command()
def ping(
    provider: Annotated[Provider, typer.Option(help="Which provider to call.")] = Provider.OLLAMA,
    prompt: Annotated[str, typer.Option(help="Prompt to send.")] = "Say hello in five words.",
    model: Annotated[str | None, typer.Option(help="Override the provider's default model.")] = None,
) -> None:
    """Send one prompt and print the reply, token usage and latency."""
    try:
        chat_model = build_chat_model(provider, Settings())
        response = chat_model.complete(ChatRequest(messages=(Message(role=Role.USER, content=prompt),), model=model))
    except LLMError as error:
        console.print(f"[red]{type(error).__name__}:[/red] {error}")
        raise typer.Exit(code=1) from error
    console.print(response.text)
    usage = response.usage
    console.print(
        f"[dim]{response.model} · in {usage.input_tokens} (cached {usage.cached_input_tokens})"
        f" · out {usage.output_tokens} · {response.latency_seconds:.2f}s[/dim]"
    )


@app.command()
def scenarios(
    name: Annotated[str, typer.Argument(help=f"One of: {', '.join(SCENARIOS)}, or {ALL_SCENARIOS}.")] = ALL_SCENARIOS,
    generation: Annotated[Generation, typer.Option(help="Which OpenAI caching rules to simulate.")] = Generation.NEW,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Run the simulator only; do not call Ollama.")] = False,
    model: Annotated[str | None, typer.Option(help="Ollama model; defaults to OLLAMA__DEFAULT_MODEL.")] = None,
    rules: Annotated[Path, typer.Option(help="Cache rules TOML file.")] = DEFAULT_RULES_PATH,
    summary_only: Annotated[bool, typer.Option("--summary-only", help="Skip the per-request tables.")] = False,
) -> None:
    """Run caching scenarios and print the simulated bill next to what Ollama reported."""
    selected = list(SCENARIOS) if name == ALL_SCENARIOS else [name]
    unknown = [scenario for scenario in selected if scenario not in SCENARIOS]
    if unknown:
        raise typer.BadParameter(f"unknown scenario {unknown[0]!r}; choose from {', '.join(SCENARIOS)}")
    run_directory = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        context = build_scenario_context(generation, is_dry_run=dry_run, model=model, rules_path=rules)
        for scenario in selected:
            result = SCENARIOS[scenario](context)
            print_result(console, result, is_summary_only=summary_only)
            save_result(result, run_directory)
    except (LLMError, RulesError, SimulatorError) as error:
        console.print(f"[red]{type(error).__name__}:[/red] {error}")
        raise typer.Exit(code=1) from error
    console.print(f"[dim]Per-call logs written to {run_directory}/[/dim]")


@app.command()
def proxy(
    upstream: Annotated[str, typer.Option(help="OpenAI-compatible base URL to forward to.")] = DEFAULT_UPSTREAM,
    generation: Annotated[Generation, typer.Option(help="Which OpenAI caching rules to simulate.")] = Generation.NEW,
    price_as: Annotated[
        str | None, typer.Option(help="Price models missing from the rules file as this one; default per generation.")
    ] = None,
    host: Annotated[str, typer.Option(help="Interface to listen on.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8787,
    rules: Annotated[Path, typer.Option(help="Cache rules TOML file.")] = DEFAULT_RULES_PATH,
) -> None:
    """Run an OpenAI-compatible proxy that logs a simulated cache bill for every chat completion."""
    try:
        proxy_app = build_proxy_app(upstream, generation, console, Path("runs") / "proxy", price_as, rules)
    except (LLMError, RulesError) as error:
        console.print(f"[red]{type(error).__name__}:[/red] {error}")
        raise typer.Exit(code=1) from error
    console.print(f"Proxy on http://{host}:{port}/v1 -> {upstream} ({generation.value} rules)")
    uvicorn.run(proxy_app, host=host, port=port, log_level="warning")
