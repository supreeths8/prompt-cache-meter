"""Simulates OpenAI prompt caching and billing for one request at a time.

Pure logic: no network, no files, and a fake clock so scenarios can skip ahead in time.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from itertools import pairwise
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, NonNegativeFloat, NonNegativeInt

from prompt_cache_meter.llm.ports import Tokenizer
from prompt_cache_meter.ports import Clock
from prompt_cache_meter.rules import CachePolicyName, CacheRules, Generation, GenerationRules, TokenRules, Ttl

# Stand-ins for chat-format tokens (message start, separators, reply priming). Real token ids are never negative.
_MESSAGE_OVERHEAD_TOKEN = -1
_REPLY_PRIMING_TOKEN = -2
_START_TIME = datetime(2026, 1, 1, tzinfo=UTC)


class SimulatorError(Exception):
    """Base class for requests the simulator refuses, the way OpenAI would return a 400."""


class UnsupportedFeatureError(SimulatorError):
    """The request uses a caching feature this model generation does not have."""


class UnsupportedTtlError(SimulatorError):
    def __init__(self, ttl: Ttl, allowed: tuple[Ttl, ...]) -> None:
        super().__init__(f"ttl {ttl.value!r} is not supported here; use one of {[t.value for t in allowed]}")


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FakeClock:
    def __init__(self, start: datetime = _START_TIME) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        if delta < timedelta(0):
            raise SimulatorError("the clock cannot go backwards")
        self._now += delta


class SimRole(StrEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# Roles that make up the initial instruction block.
_INSTRUCTION_ROLES = frozenset({SimRole.SYSTEM, SimRole.DEVELOPER})


class CacheMode(StrEnum):
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Breakpoint(_Frozen):
    # None means the request-level ttl.
    ttl: Ttl | None = None


class SimMessage(_Frozen):
    role: SimRole
    content: str
    cache_breakpoint: Breakpoint | None = None


class SimRequest(_Frozen):
    model: str
    messages: tuple[SimMessage, ...] = Field(min_length=1)
    tools: tuple[dict[str, JsonValue], ...] = ()
    response_schema: dict[str, JsonValue] | None = None
    # Other settings OpenAI renders into the prefix (reasoning_effort, verbosity, parallel_tool_calls).
    prefix_options: dict[str, JsonValue] = Field(default_factory=dict)
    prompt_cache_key: str = ""
    mode: CacheMode = CacheMode.IMPLICIT
    ttl: Ttl | None = None


class Bill(_Frozen):
    model: str
    input_tokens: NonNegativeInt
    cached_tokens: NonNegativeInt
    written_tokens_by_ttl: dict[Ttl, NonNegativeInt]
    uncached_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    cost_usd: NonNegativeFloat

    @property
    def written_tokens(self) -> int:
        return sum(self.written_tokens_by_ttl.values())

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / self.input_tokens if self.input_tokens else 0.0


@dataclass(frozen=True)
class RenderedPrompt:
    tokens: tuple[int, ...]
    # Token offset just after each message, in message order.
    message_ends: tuple[int, ...]


@dataclass
class CacheEntry:
    tokens: tuple[int, ...]
    ttl: Ttl
    expires_at: datetime


@dataclass(frozen=True)
class CacheWrite:
    end: int
    ttl: Ttl


@dataclass(frozen=True)
class CacheOutcome:
    cached_tokens: int
    written_tokens_by_ttl: dict[Ttl, int]
    writes: tuple[CacheWrite, ...]
    read_entries: tuple[CacheEntry, ...]


Namespace = tuple[str, str]


class CacheStore:
    """Live cache entries, separated by (prompt_cache_key, model) so tenants and models never share."""

    def __init__(self) -> None:
        self._entries: dict[Namespace, list[CacheEntry]] = {}

    def live_entries(self, namespace: Namespace, now: datetime) -> list[CacheEntry]:
        live = [entry for entry in self._entries.get(namespace, []) if entry.expires_at > now]
        self._entries[namespace] = live
        return live

    def put(self, namespace: Namespace, entry: CacheEntry) -> None:
        entries = self._entries.setdefault(namespace, [])
        same = next((e for e in entries if e.tokens == entry.tokens and e.ttl == entry.ttl), None)
        if same is None:
            entries.append(entry)
            return
        same.expires_at = max(same.expires_at, entry.expires_at)


class CachePolicy(Protocol):
    def plan(self, request: SimRequest, prompt: RenderedPrompt, entries: list[CacheEntry]) -> CacheOutcome:
        """Decide which tokens are read from cache, which are written, and which entries to store."""
        ...


def render(request: SimRequest, tokenizer: Tokenizer, rules: TokenRules) -> RenderedPrompt:
    tokens = _encode_header(request, tokenizer)
    ends: list[int] = []
    for message in request.messages:
        tokens.extend([_MESSAGE_OVERHEAD_TOKEN] * rules.tokens_per_message)
        tokens.extend(tokenizer.encode(message.role.value))
        tokens.extend(tokenizer.encode(message.content))
        ends.append(len(tokens))
    tokens.extend([_REPLY_PRIMING_TOKEN] * rules.reply_priming_tokens)
    return RenderedPrompt(tokens=tuple(tokens), message_ends=tuple(ends))


def _encode_header(request: SimRequest, tokenizer: Tokenizer) -> list[int]:
    # Tools, the response schema and other prefix settings are rendered ahead of the messages,
    # so changing any of them misses from token 0.
    if not request.tools and request.response_schema is None and not request.prefix_options:
        return []
    header = {
        "tools": list(request.tools),
        "response_schema": request.response_schema,
        "prefix_options": request.prefix_options,
    }
    return tokenizer.encode(json.dumps(header, sort_keys=True))


def common_prefix_length(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    # Binary search over slice equality keeps the comparison in C, which matters for 60-turn chats.
    low, high = 0, min(len(first), len(second))
    while low < high:
        middle = (low + high + 1) // 2
        if first[:middle] == second[:middle]:
            low = middle
        else:
            high = middle - 1
    return low


def _round_down(value: int, multiple: int) -> int:
    return value - value % multiple


def _has_breakpoints(request: SimRequest) -> bool:
    return any(message.cache_breakpoint is not None for message in request.messages)


class FixedIntervalPolicy:
    """Earlier models: automatic prefix caching, hits counted in fixed steps, writes are free."""

    def __init__(self, generation: GenerationRules) -> None:
        self._generation = generation

    def plan(self, request: SimRequest, prompt: RenderedPrompt, entries: list[CacheEntry]) -> CacheOutcome:
        if request.mode is CacheMode.EXPLICIT or _has_breakpoints(request):
            raise UnsupportedFeatureError("explicit cache breakpoints need a GPT-5.6 or later model")
        matches = [(common_prefix_length(entry.tokens, prompt.tokens), entry) for entry in entries]
        cached = self._reported_hit(max((length for length, _ in matches), default=0))
        read_entries = tuple(entry for length, entry in matches if cached and length >= cached)
        total = len(prompt.tokens)
        is_cacheable = total >= self._generation.min_cacheable_tokens
        ttl = request.ttl or self._generation.default_ttl
        writes = (CacheWrite(end=total, ttl=ttl),) if is_cacheable else ()
        return CacheOutcome(cached_tokens=cached, written_tokens_by_ttl={}, writes=writes, read_entries=read_entries)

    def _reported_hit(self, matched: int) -> int:
        at_breakpoint = _round_down(matched, self._generation.breakpoint_interval_tokens)
        reported = _round_down(at_breakpoint, self._generation.reported_multiple_tokens)
        return reported if reported >= self._generation.min_cacheable_tokens else 0


class MessageBoundaryPolicy:
    """GPT-5.6 and later: reads and writes only at message boundaries or explicit breakpoints."""

    def __init__(self, generation: GenerationRules, rules: CacheRules) -> None:
        self._generation = generation
        self._limits = rules.limits
        self._minutes = {ttl: ttl_rules.minutes for ttl, ttl_rules in rules.ttls.items()}

    def plan(self, request: SimRequest, prompt: RenderedPrompt, entries: list[CacheEntry]) -> CacheOutcome:
        default_ttl = request.ttl or self._generation.default_ttl
        lookups = self._lookup_ends(request, prompt, default_ttl)
        writes = self._write_slots(request, prompt, default_ttl)
        coverage = [(common_prefix_length(entry.tokens, prompt.tokens), entry) for entry in entries]
        read_ends = {ttl: self._read_end(lookups, coverage, ttl) for ttl in self._generation.allowed_ttls}
        cached, written = self._classify(len(prompt.tokens), writes, read_ends)
        longest_read = max(read_ends.values(), default=0)
        read_entries = tuple(entry for length, entry in coverage if longest_read and length >= longest_read)
        new_writes = tuple(write for write in writes if write.end > read_ends[write.ttl])
        return CacheOutcome(cached, written, new_writes, read_entries)

    def _lookup_ends(self, request: SimRequest, prompt: RenderedPrompt, default_ttl: Ttl) -> list[int]:
        explicit = [write.end for write in _explicit_breakpoints(request, prompt, default_ttl)]
        latest = self._limits.explicit_lookup_latest
        ends = explicit[: self._limits.explicit_lookup_first] + (explicit[-latest:] if latest else [])
        if request.mode is CacheMode.IMPLICIT:
            eligible = _implicit_ends(request, prompt)
            ends += eligible[-(self._limits.implicit_lookback_messages + 1) :]
            ends += [end for end in [_system_block_end(request, prompt)] if end is not None]
        return sorted({end for end in ends if end >= self._generation.min_cacheable_tokens})

    def _write_slots(self, request: SimRequest, prompt: RenderedPrompt, default_ttl: Ttl) -> list[CacheWrite]:
        minimum = self._generation.min_cacheable_tokens
        explicit = [w for w in _explicit_breakpoints(request, prompt, default_ttl) if w.end >= minimum]
        implicit: list[CacheWrite] = []
        latest = next(reversed(_implicit_ends(request, prompt)), None)
        if request.mode is CacheMode.IMPLICIT and latest is not None and latest >= minimum:
            implicit = [CacheWrite(end=latest, ttl=default_ttl)]
        # The implicit breakpoint takes one of the write slots; explicit ones fill the rest, latest first.
        explicit_slots = self._limits.max_writes_per_request - len(implicit)
        return implicit + (explicit[-explicit_slots:] if explicit_slots > 0 else [])

    def _read_end(self, lookups: list[int], coverage: list[tuple[int, CacheEntry]], ttl: Ttl) -> int:
        # Longest lookup boundary covered by a live entry that lives at least as long as `ttl`.
        floor = self._minutes[ttl]
        lived = [length for length, entry in coverage if self._minutes[entry.ttl] >= floor]
        return max((end for end in lookups if any(length >= end for length in lived)), default=0)

    def _classify(self, total: int, writes: list[CacheWrite], read_ends: dict[Ttl, int]) -> tuple[int, dict[Ttl, int]]:
        cuts = sorted({0, total, *(write.end for write in writes), *read_ends.values()})
        cached = 0
        written: dict[Ttl, int] = {}
        for start, stop in pairwise(cuts):
            covering = [write.ttl for write in writes if write.end >= stop]
            # Each token is billed once, at the longest lifetime among the writes that cover it.
            ttl = max(covering, key=lambda candidate: self._minutes[candidate], default=None)
            read_end = read_ends[ttl] if ttl is not None else max(read_ends.values(), default=0)
            if read_end >= stop:
                cached += stop - start
            elif ttl is not None:
                written[ttl] = written.get(ttl, 0) + stop - start
        return cached, written


def _system_block_end(request: SimRequest, prompt: RenderedPrompt) -> int | None:
    leading = 0
    for message in request.messages:
        if message.role not in _INSTRUCTION_ROLES:
            break
        leading += 1
    return prompt.message_ends[leading - 1] if leading else None


def _implicit_ends(request: SimRequest, prompt: RenderedPrompt) -> list[int]:
    # Eligible implicit boundaries: the end of the initial instruction block, the end of every user message,
    # and the last tool response of each consecutive group of tool responses.
    roles = [message.role for message in request.messages]
    following = [*roles[1:], None]
    message_ends = [
        end
        for role, next_role, end in zip(roles, following, prompt.message_ends, strict=True)
        if role is SimRole.USER or (role is SimRole.TOOL and next_role is not SimRole.TOOL)
    ]
    system_end = _system_block_end(request, prompt)
    return sorted({*message_ends, *([system_end] if system_end is not None else [])})


def _explicit_breakpoints(request: SimRequest, prompt: RenderedPrompt, default_ttl: Ttl) -> list[CacheWrite]:
    return [
        CacheWrite(end=end, ttl=message.cache_breakpoint.ttl or default_ttl)
        for message, end in zip(request.messages, prompt.message_ends, strict=True)
        if message.cache_breakpoint is not None
    ]


def build_policy(generation: GenerationRules, rules: CacheRules) -> CachePolicy:
    match generation.cache_policy:
        case CachePolicyName.FIXED_INTERVAL:
            return FixedIntervalPolicy(generation)
        case CachePolicyName.MESSAGE_BOUNDARY:
            return MessageBoundaryPolicy(generation, rules)


class CacheSimulator:
    def __init__(self, rules: CacheRules, generation: Generation, tokenizer: Tokenizer, clock: Clock) -> None:
        self._rules = rules
        self._generation = rules.for_generation(generation)
        self._policy = build_policy(self._generation, rules)
        self._tokenizer = tokenizer
        self._clock = clock
        self._store = CacheStore()

    @property
    def generation(self) -> GenerationRules:
        return self._generation

    def count_input_tokens(self, request: SimRequest) -> int:
        return len(render(request, self._tokenizer, self._rules.tokens).tokens)

    def simulate(self, request: SimRequest, output_tokens: int) -> Bill:
        self._check_ttls(request)
        price = self._rules.price(request.model)
        prompt = render(request, self._tokenizer, self._rules.tokens)
        namespace = (request.prompt_cache_key, request.model)
        now = self._clock.now()
        outcome = self._policy.plan(request, prompt, self._store.live_entries(namespace, now))
        self._apply(namespace, prompt, outcome, now)
        written = sum(outcome.written_tokens_by_ttl.values())
        uncached = len(prompt.tokens) - outcome.cached_tokens - written
        write_units = sum(
            count * self._rules.ttls[ttl].write_multiplier for ttl, count in outcome.written_tokens_by_ttl.items()
        )
        input_units = uncached + outcome.cached_tokens * price.cached_read_multiplier + write_units
        cost = (input_units * price.input_per_million + output_tokens * price.output_per_million) / 1_000_000
        return Bill(
            model=request.model,
            input_tokens=len(prompt.tokens),
            cached_tokens=outcome.cached_tokens,
            written_tokens_by_ttl=outcome.written_tokens_by_ttl,
            uncached_tokens=uncached,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    def _check_ttls(self, request: SimRequest) -> None:
        allowed = self._generation.allowed_ttls
        breakpoint_ttls = [m.cache_breakpoint.ttl for m in request.messages if m.cache_breakpoint is not None]
        for ttl in [request.ttl, *breakpoint_ttls]:
            if ttl is not None and ttl not in allowed:
                raise UnsupportedTtlError(ttl, allowed)

    def _apply(self, namespace: Namespace, prompt: RenderedPrompt, outcome: CacheOutcome, now: datetime) -> None:
        for entry in outcome.read_entries:
            ttl_rules = self._rules.ttls[entry.ttl]
            if ttl_rules.is_refreshed_on_read:
                entry.expires_at = max(entry.expires_at, now + timedelta(minutes=ttl_rules.minutes))
        for write in outcome.writes:
            expires_at = now + timedelta(minutes=self._rules.ttls[write.ttl].minutes)
            self._store.put(
                namespace, CacheEntry(tokens=prompt.tokens[: write.end], ttl=write.ttl, expires_at=expires_at)
            )
