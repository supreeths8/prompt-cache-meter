class LLMError(Exception):
    """Base class for every failure talking to a model provider."""


class MissingApiKeyError(LLMError):
    def __init__(self, provider: str) -> None:
        super().__init__(f"{provider} needs an API key. Set it in .env (see .env.example).")


class ProviderUnavailableError(LLMError):
    """The provider could not be reached (server down, network, timeout)."""


class ProviderResponseError(LLMError):
    """The provider answered, but with an error or a response we cannot use."""


class TokenizerError(LLMError):
    """The tokenizer could not be loaded (unknown encoding, or the encoding file could not be downloaded)."""
