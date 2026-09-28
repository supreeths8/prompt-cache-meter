import zlib
from typing import Any

from prompt_cache_meter.rules import CacheRules, Generation, Ttl
from prompt_cache_meter.simulator import Breakpoint, CacheSimulator, FakeClock, SimMessage, SimRole

# No chat-format overhead, so a message of n tokens is exactly its role token plus n - 1 words.
RULES: dict[str, Any] = {
    "tokens": {"encoding": "fake", "tokens_per_message": 0, "reply_priming_tokens": 0},
    "limits": {
        "max_writes_per_request": 4,
        "implicit_lookback_messages": 20,
        "explicit_lookup_first": 2,
        "explicit_lookup_latest": 50,
    },
    "ollama": {"default_context_length": 4096, "context_warning_ratio": 0.9},
    "ttls": {
        "in_memory": {"minutes": 10, "is_refreshed_on_read": True, "write_multiplier": 1.0},
        "24h": {"minutes": 1440, "is_refreshed_on_read": False, "write_multiplier": 1.0},
        "30m": {"minutes": 30, "is_refreshed_on_read": True, "write_multiplier": 1.25},
        "12h": {"minutes": 720, "is_refreshed_on_read": False, "write_multiplier": 2.5},
    },
    "generations": {
        "legacy": {
            "default_model": "legacy-model",
            "default_ttl": "in_memory",
            "allowed_ttls": ["in_memory", "24h"],
            "cache_policy": "fixed_interval",
            "min_cacheable_tokens": 1024,
            "breakpoint_interval_tokens": 128,
            "reported_multiple_tokens": 128,
        },
        "new": {
            "default_model": "new-model",
            "default_ttl": "30m",
            "allowed_ttls": ["30m", "12h"],
            "cache_policy": "message_boundary",
            "min_cacheable_tokens": 1024,
        },
    },
    "prices": {
        "legacy-model": {"input_per_million": 1.0, "cached_read_multiplier": 0.1, "output_per_million": 10.0},
        "new-model": {"input_per_million": 1.0, "cached_read_multiplier": 0.1, "output_per_million": 10.0},
        "other-model": {"input_per_million": 1.0, "cached_read_multiplier": 0.1, "output_per_million": 10.0},
    },
}


class WordTokenizer:
    """One token per whitespace-separated word, so tests can build prompts of exact sizes."""

    def encode(self, text: str) -> list[int]:
        return [zlib.crc32(word.encode()) for word in text.split()]


def make_rules() -> CacheRules:
    return CacheRules.model_validate(RULES)


def make_simulator(generation: Generation, clock: FakeClock | None = None) -> CacheSimulator:
    return CacheSimulator(make_rules(), generation, WordTokenizer(), clock or FakeClock())


def message(
    role: SimRole, tokens: int, seed: str = "w", breakpoint_ttl: Ttl | None = None, has_breakpoint: bool = False
) -> SimMessage:
    """A message that renders to exactly `tokens` tokens (one role token plus words)."""
    content = " ".join(f"{seed}{index}" for index in range(tokens - 1))
    cache_breakpoint = Breakpoint(ttl=breakpoint_ttl) if has_breakpoint or breakpoint_ttl else None
    return SimMessage(role=role, content=content, cache_breakpoint=cache_breakpoint)
