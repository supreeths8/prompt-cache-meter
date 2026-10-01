import pytest
from pydantic import ValidationError

from prompt_cache_meter.bootstrap import build_chat_model
from prompt_cache_meter.config import Provider, Settings
from prompt_cache_meter.llm.errors import MissingApiKeyError

MODELS_ENV = {
    "OLLAMA__DEFAULT_MODEL": "local-model",
    "OPENROUTER__DEFAULT_MODEL": "hosted-model",
}


def load_settings(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> Settings:
    for name in [*MODELS_ENV, "OPENROUTER__API_KEY"]:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)


def test_default_model_comes_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = load_settings(monkeypatch, MODELS_ENV)

    assert settings.for_provider(Provider.OLLAMA).default_model == "local-model"
    assert settings.for_provider(Provider.OPENROUTER).default_model == "hosted-model"


def test_missing_default_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="default_model"):
        load_settings(monkeypatch, {"OLLAMA__DEFAULT_MODEL": "local-model"})


def test_partial_override_keeps_class_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = load_settings(monkeypatch, {**MODELS_ENV, "OPENROUTER__API_KEY": "sk-test"})

    openrouter = settings.for_provider(Provider.OPENROUTER)
    assert openrouter.base_url == "https://openrouter.ai/api/v1"
    assert build_chat_model(Provider.OPENROUTER, settings) is not None


def test_openrouter_without_key_raises_missing_key_error(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = load_settings(monkeypatch, MODELS_ENV)

    with pytest.raises(MissingApiKeyError, match="openrouter"):
        build_chat_model(Provider.OPENROUTER, settings)


def test_ollama_builds_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = load_settings(monkeypatch, MODELS_ENV)

    assert build_chat_model(Provider.OLLAMA, settings) is not None
