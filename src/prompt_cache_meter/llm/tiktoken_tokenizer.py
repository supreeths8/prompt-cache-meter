import tiktoken

from prompt_cache_meter.llm.errors import TokenizerError


class TiktokenTokenizer:
    """Counts tokens with OpenAI's own encodings, so simulated bills match what OpenAI would see."""

    def __init__(self, encoding_name: str) -> None:
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        # requests errors subclass OSError; tiktoken downloads the encoding file on first use.
        except (ValueError, OSError) as error:
            raise TokenizerError(f"could not load tiktoken encoding {encoding_name!r}: {error}") from error

    def encode(self, text: str) -> list[int]:
        # Prompts are plain text, so special-token strings inside them are counted as ordinary text.
        return self._encoding.encode(text, disallowed_special=())
