from enum import StrEnum

from pydantic import BaseModel, PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(StrEnum):
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


class ProviderSettings(BaseModel):
    base_url: str
    default_model: str
    api_key: SecretStr | None = None
    requires_api_key: bool = True


# Defaults live on per-provider classes so a partial override in .env
# (say, only OPENROUTER__API_KEY) merges with them instead of replacing them.
# default_model has no default on purpose: it always comes from .env.
class OllamaSettings(ProviderSettings):
    base_url: str = "http://127.0.0.1:11434/v1"
    requires_api_key: bool = False


class OpenRouterSettings(ProviderSettings):
    base_url: str = "https://openrouter.ai/api/v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")

    ollama: OllamaSettings
    openrouter: OpenRouterSettings

    def for_provider(self, provider: Provider) -> ProviderSettings:
        match provider:
            case Provider.OLLAMA:
                return self.ollama
            case Provider.OPENROUTER:
                return self.openrouter


class OllamaServerSettings(BaseSettings):
    """Reads the same variable the Ollama server uses, so the warning matches the server's real limit."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_context_length: PositiveInt | None = None
