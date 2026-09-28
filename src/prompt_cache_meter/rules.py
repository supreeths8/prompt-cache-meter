import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    ValidationError,
    model_validator,
)

DEFAULT_RULES_PATH = Path(__file__).with_name("cache_rules.toml")


class RulesError(Exception):
    """Base class for problems loading or using the cache rules."""


class InvalidRulesError(RulesError):
    """The rules file is missing, unreadable or does not validate."""


class UnknownModelError(RulesError):
    def __init__(self, model: str) -> None:
        super().__init__(f"no price for model {model!r}; add it under [prices] in the rules file")


class Ttl(StrEnum):
    IN_MEMORY = "in_memory"
    HOURS_24 = "24h"
    MINUTES_30 = "30m"
    HOURS_12 = "12h"


class Generation(StrEnum):
    LEGACY = "legacy"
    NEW = "new"


class CachePolicyName(StrEnum):
    # Earlier models: breakpoints every N tokens, counted from the start of the prompt.
    FIXED_INTERVAL = "fixed_interval"
    # GPT-5.6 and later: breakpoints at message ends, plus optional explicit breakpoints.
    MESSAGE_BOUNDARY = "message_boundary"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TokenRules(_Frozen):
    encoding: str
    tokens_per_message: NonNegativeInt
    reply_priming_tokens: NonNegativeInt


class LimitRules(_Frozen):
    max_writes_per_request: PositiveInt
    implicit_lookback_messages: NonNegativeInt
    explicit_lookup_first: NonNegativeInt
    explicit_lookup_latest: NonNegativeInt


class OllamaRules(_Frozen):
    default_context_length: PositiveInt
    context_warning_ratio: float = Field(gt=0.0, le=1.0)


class TtlRules(_Frozen):
    minutes: PositiveInt
    is_refreshed_on_read: bool
    write_multiplier: float = Field(ge=1.0)


class GenerationRules(_Frozen):
    default_model: str
    default_ttl: Ttl
    allowed_ttls: tuple[Ttl, ...] = Field(min_length=1)
    cache_policy: CachePolicyName
    min_cacheable_tokens: PositiveInt
    breakpoint_interval_tokens: PositiveInt = 1
    reported_multiple_tokens: PositiveInt = 1

    @model_validator(mode="after")
    def _default_ttl_is_allowed(self) -> Self:
        if self.default_ttl not in self.allowed_ttls:
            raise ValueError(f"default_ttl {self.default_ttl} is not in allowed_ttls")
        return self


class ModelPrice(_Frozen):
    input_per_million: NonNegativeFloat
    cached_read_multiplier: float = Field(ge=0.0, le=1.0)
    output_per_million: NonNegativeFloat


class CacheRules(_Frozen):
    tokens: TokenRules
    limits: LimitRules
    ollama: OllamaRules
    ttls: dict[Ttl, TtlRules]
    generations: dict[Generation, GenerationRules]
    prices: dict[str, ModelPrice]

    @model_validator(mode="after")
    def _every_ttl_and_model_is_defined(self) -> Self:
        for name, generation in self.generations.items():
            missing = [ttl for ttl in generation.allowed_ttls if ttl not in self.ttls]
            if missing:
                raise ValueError(f"generation {name} uses undefined ttls {missing}")
            if generation.default_model not in self.prices:
                raise ValueError(f"generation {name} default_model {generation.default_model!r} has no price")
        return self

    def for_generation(self, generation: Generation) -> GenerationRules:
        rules = self.generations.get(generation)
        if rules is None:
            raise InvalidRulesError(f"rules file has no [generations.{generation.value}] section")
        return rules

    def price(self, model: str) -> ModelPrice:
        price = self.prices.get(model)
        if price is None:
            raise UnknownModelError(model)
        return price


def load_rules(path: Path = DEFAULT_RULES_PATH) -> CacheRules:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        return CacheRules.model_validate(raw)
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as error:
        raise InvalidRulesError(f"{path}: {error}") from error
