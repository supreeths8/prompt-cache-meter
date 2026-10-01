from pathlib import Path

import httpx2 as httpx
from openai import OpenAI
from rich.console import Console
from starlette.applications import Starlette

from prompt_cache_meter.config import OllamaServerSettings, Provider, Settings
from prompt_cache_meter.llm.errors import MissingApiKeyError
from prompt_cache_meter.llm.openai_compatible import OpenAICompatibleChatModel
from prompt_cache_meter.llm.ports import ChatModel
from prompt_cache_meter.llm.tiktoken_tokenizer import TiktokenTokenizer
from prompt_cache_meter.proxy.app import create_app
from prompt_cache_meter.proxy.recorder import ConsoleJsonlRecorder
from prompt_cache_meter.proxy.service import SimulationService
from prompt_cache_meter.proxy.upstream import Upstream
from prompt_cache_meter.rules import DEFAULT_RULES_PATH, Generation, load_rules
from prompt_cache_meter.scenarios import ScenarioContext
from prompt_cache_meter.simulator import SystemClock

# Local models can take minutes on a long prompt; only the connect step should fail fast.
_UPSTREAM_TIMEOUT = httpx.Timeout(600.0, connect=10.0)

# Ollama ignores the key, but the OpenAI client refuses to start without one.
_UNUSED_API_KEY = "not-needed"


def build_chat_model(provider: Provider, settings: Settings) -> ChatModel:
    provider_settings = settings.for_provider(provider)
    api_key = provider_settings.api_key.get_secret_value() if provider_settings.api_key else ""
    if provider_settings.requires_api_key and not api_key:
        raise MissingApiKeyError(provider.value)
    client = OpenAI(base_url=provider_settings.base_url, api_key=api_key or _UNUSED_API_KEY)
    return OpenAICompatibleChatModel(client, default_model=provider_settings.default_model)


def build_scenario_context(
    generation: Generation, is_dry_run: bool, model: str | None = None, rules_path: Path = DEFAULT_RULES_PATH
) -> ScenarioContext:
    rules = load_rules(rules_path)
    context_length = OllamaServerSettings().ollama_context_length or rules.ollama.default_context_length
    # Dry runs skip Settings entirely, so they work without a .env or a running Ollama.
    chat_model = None if is_dry_run else build_chat_model(Provider.OLLAMA, Settings())
    return ScenarioContext(
        rules=rules,
        generation=generation,
        tokenizer=TiktokenTokenizer(rules.tokens.encoding),
        context_length=context_length,
        chat_model=chat_model,
        ollama_model=model,
    )


def build_proxy_app(
    upstream_url: str,
    generation: Generation,
    console: Console,
    log_directory: Path,
    price_as: str | None = None,
    rules_path: Path = DEFAULT_RULES_PATH,
) -> Starlette:
    rules = load_rules(rules_path)
    priced_as = price_as or rules.for_generation(generation).default_model
    rules.price(priced_as)  # fail at startup, not on the first request, if the fallback has no price
    service = SimulationService(
        rules=rules,
        generation=generation,
        tokenizer=TiktokenTokenizer(rules.tokens.encoding),
        clock=SystemClock(),
        price_as=priced_as,
    )
    upstream = Upstream(upstream_url, httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT))
    return create_app(upstream, service, ConsoleJsonlRecorder(console, log_directory))
