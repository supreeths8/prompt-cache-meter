"""The seven caching scenarios. Each one replays requests through the simulator and, unless dry-run, Ollama."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta

from pydantic import BaseModel, ConfigDict, NonNegativeFloat, NonNegativeInt

from prompt_cache_meter.llm.models import ChatRequest, Message, Role
from prompt_cache_meter.llm.ports import ChatModel, Tokenizer
from prompt_cache_meter.rules import CachePolicyName, CacheRules, Generation, Ttl
from prompt_cache_meter.simulator import (
    Bill,
    Breakpoint,
    CacheMode,
    CacheSimulator,
    FakeClock,
    SimMessage,
    SimRequest,
    SimRole,
)

REPLY_MAX_TOKENS = 200
SYSTEM_POLICY_COUNT = 40
DOCUMENT_SECTION_COUNT = 35
CHAT_TURNS = 60
SLIDING_WINDOW_TURNS = 20
SUMMARY_TOKEN_BUDGET = 3000
VERBATIM_TURNS_AFTER_SUMMARY = 5
IDLE_GAP_TURNS = (20, 40)
RAG_QUESTION_GAP = timedelta(hours=1)
TURN_GAP = timedelta(minutes=1)


class OllamaObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    prompt_tokens: NonNegativeInt
    cached_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    latency_seconds: NonNegativeFloat


class Row(BaseModel):
    model_config = ConfigDict(frozen=True)

    variant: str
    label: str
    bill: Bill
    ollama: OllamaObservation | None
    warnings: tuple[str, ...]


class ScenarioResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    title: str
    notes: tuple[str, ...]
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class ScenarioContext:
    rules: CacheRules
    generation: Generation
    tokenizer: Tokenizer
    context_length: int
    chat_model: ChatModel | None = None
    ollama_model: str | None = None

    @property
    def sim_model(self) -> str:
        return self.rules.for_generation(self.generation).default_model

    def new_session(self, variant: str) -> "Session":
        return Session(self, variant)


@dataclass
class Session:
    """One fresh simulated cache and clock. Ollama's own KV cache is not reset between sessions."""

    context: ScenarioContext
    variant: str
    clock: FakeClock = field(default_factory=FakeClock)
    rows: list[Row] = field(default_factory=list)

    def __post_init__(self) -> None:
        ctx = self.context
        self.simulator = CacheSimulator(ctx.rules, ctx.generation, ctx.tokenizer, self.clock)

    def send(self, label: str, request: SimRequest, canned_reply: str) -> str:
        """Bill the request, call Ollama unless dry-run, and return the reply text."""
        input_tokens = self.simulator.count_input_tokens(request)
        warnings = self._pre_call_warnings(input_tokens)
        observation, reply = self._call_ollama(request, canned_reply)
        if observation is not None and observation.prompt_tokens >= self.context.context_length:
            warnings.append(f"Ollama saw {observation.prompt_tokens} tokens: likely truncated")
        bill = self.simulator.simulate(request, output_tokens=len(self.context.tokenizer.encode(reply)))
        self.rows.append(
            Row(variant=self.variant, label=label, bill=bill, ollama=observation, warnings=tuple(warnings))
        )
        return reply

    def _pre_call_warnings(self, input_tokens: int) -> list[str]:
        limit = self.context.context_length
        if input_tokens <= limit * self.context.rules.ollama.context_warning_ratio:
            return []
        return [f"{input_tokens} tiktoken tokens is near OLLAMA_CONTEXT_LENGTH={limit}"]

    def _call_ollama(self, request: SimRequest, canned_reply: str) -> tuple[OllamaObservation | None, str]:
        chat_model = self.context.chat_model
        if chat_model is None:
            return None, canned_reply
        messages = tuple(Message(role=_chat_role(m.role), content=m.content) for m in request.messages)
        response = chat_model.complete(
            ChatRequest(messages=messages, model=self.context.ollama_model, max_tokens=REPLY_MAX_TOKENS)
        )
        usage = response.usage
        observation = OllamaObservation(
            model=response.model,
            prompt_tokens=usage.input_tokens,
            cached_tokens=usage.cached_input_tokens,
            output_tokens=usage.output_tokens,
            latency_seconds=response.latency_seconds,
        )
        return observation, response.text


def _chat_role(role: SimRole) -> Role:
    # Scenarios only build system, user and assistant messages; developer goes to Ollama as system,
    # because Ollama's OpenAI endpoint drops the developer role.
    match role:
        case SimRole.SYSTEM | SimRole.DEVELOPER:
            return Role.SYSTEM
        case SimRole.USER | SimRole.TOOL:
            return Role.USER
        case SimRole.ASSISTANT:
            return Role.ASSISTANT


# Prompt material. Deterministic so the simulator sees the same tokens on every run.


def support_policies() -> str:
    lines = [
        f"Policy {index}: for order type {index}, refunds are allowed within {index + 7} days "
        f"if the item is unused and the customer supplies receipt number format R-{index:03d}."
        for index in range(1, SYSTEM_POLICY_COUNT + 1)
    ]
    return "You are a support assistant for Acme Widgets. Answer in one short sentence.\n" + "\n".join(lines)


def document(name: str) -> str:
    sections = [
        f"{name} section {index}: model {name}-{index} has a battery life of {index * 3 + 5} hours, "
        f"weighs {index * 7 + 100} grams and ships from warehouse {index % 4 + 1}."
        for index in range(1, DOCUMENT_SECTION_COUNT + 1)
    ]
    return f"Document {name}:\n" + "\n".join(sections)


def system(content: str, cache_breakpoint: Breakpoint | None = None) -> SimMessage:
    return SimMessage(role=SimRole.SYSTEM, content=content, cache_breakpoint=cache_breakpoint)


def user(content: str, cache_breakpoint: Breakpoint | None = None) -> SimMessage:
    return SimMessage(role=SimRole.USER, content=content, cache_breakpoint=cache_breakpoint)


def assistant(content: str) -> SimMessage:
    return SimMessage(role=SimRole.ASSISTANT, content=content)


def _result(name: str, title: str, sessions: list[Session], notes: tuple[str, ...] = ()) -> ScenarioResult:
    rows = tuple(row for session in sessions for row in session.rows)
    return ScenarioResult(name=name, title=title, notes=notes, rows=rows)


# 1. Cold, warm, broken.


def cold_warm_broken(ctx: ScenarioContext) -> ScenarioResult:
    session = ctx.new_session("timestamp")
    question = user("What is the refund window for order type 12?")
    request = SimRequest(model=ctx.sim_model, messages=(system(support_policies()), question))
    session.send("cold", request, "Order type 12 has a 19 day refund window.")
    session.clock.advance(TURN_GAP)
    session.send("warm", request, "Order type 12 has a 19 day refund window.")
    session.clock.advance(TURN_GAP)
    stamped = system(f"Current time: {session.clock.now().isoformat()}\n{support_policies()}")
    broken = SimRequest(model=ctx.sim_model, messages=(stamped, question))
    session.send("broken: timestamp at top", broken, "Order type 12 has a 19 day refund window.")
    return _result("cold-warm-broken", "Cold, warm, then a timestamp at the very top", [session])


# 2. Question wording.


def question_wording(ctx: ScenarioContext) -> ScenarioResult:
    session = ctx.new_session("same system prompt")
    for label in ("What's the capital of France?", "What is the capital of France?"):
        request = SimRequest(model=ctx.sim_model, messages=(system(support_policies()), user(label)))
        session.send(label, request, "Paris.")
        session.clock.advance(TURN_GAP)
    notes = ("Only the system prompt can be shared; the question differs from its second token.",)
    return _result("question-wording", "Same system prompt, two wordings of one question", [session], notes)


# 3. User name placement.


def name_placement(ctx: ScenarioContext) -> ScenarioResult:
    sessions = []
    for variant in ("name at start", "name at end"):
        session = ctx.new_session(variant)
        for name in ("Alice", "Bob"):
            prompt = _with_name(support_policies(), name, is_at_start=variant == "name at start")
            request = SimRequest(model=ctx.sim_model, messages=(system(prompt), user("What is policy 3?")))
            session.send(f"user {name}", request, "Order type 3 has a 10 day refund window.")
            session.clock.advance(TURN_GAP)
        sessions.append(session)
    notes = (
        "Legacy caches any 128-token step, so a name at the end still shares the policies.",
        "New implicit caching only reads at message ends, so a name anywhere in the system message misses;"
        " put per-user text in its own later message instead.",
    )
    return _result("name-placement", "User name at the start vs the end of the system prompt", sessions, notes)


def _with_name(prompt: str, name: str, is_at_start: bool) -> str:
    return f"The user's name is {name}.\n{prompt}" if is_at_start else f"{prompt}\nThe user's name is {name}."


# 4. Long chat.


@dataclass(frozen=True)
class Turn:
    order_type: int
    question: str
    answer: str

    def messages(self) -> tuple[SimMessage, SimMessage]:
        return user(self.question), assistant(self.answer)


def chat_turn(index: int) -> Turn:
    order_type = index % SYSTEM_POLICY_COUNT + 1
    question = f"Turn {index}: what is the refund window for order type {order_type}, and what receipt do I need?"
    answer = (
        f"For order type {order_type} you have {order_type + 7} days to request a refund, "
        f"the item must be unused, and you need a receipt in format R-{order_type:03d}."
    )
    return Turn(order_type=order_type, question=question, answer=answer)


def _history(turns: list[Turn]) -> tuple[SimMessage, ...]:
    return tuple(message for turn in turns for message in turn.messages())


def _chat_request(ctx: ScenarioContext, prefix: tuple[SimMessage, ...], turn: Turn) -> SimRequest:
    messages = (system(support_policies()), *prefix, user(turn.question))
    return SimRequest(model=ctx.sim_model, messages=messages)


def _summary_messages(summary: str) -> tuple[SimMessage, ...]:
    # The summary goes after the system prompt as a normal exchange, so the system prompt stays byte-identical.
    return user(f"Summary of our conversation so far: {summary}"), assistant("Understood, I will keep that in mind.")


def _summarise(session: Session, ctx: ScenarioContext, turns: list[Turn], summary: str) -> str:
    transcript = "\n".join(f"Q: {turn.question}\nA: {turn.answer}" for turn in turns)
    instructions = system("Summarise this support conversation in at most five sentences. Keep order types.")
    request = SimRequest(model=ctx.sim_model, messages=(instructions, user(f"{summary}\n{transcript}")))
    order_types = sorted({turn.order_type for turn in turns})
    canned = f"The customer asked about refund windows and receipts for order types {order_types}."
    return session.send(f"summarise {len(turns)} turns", request, canned)


def sliding_window(ctx: ScenarioContext) -> Session:
    session = ctx.new_session("a: sliding window")
    turns: list[Turn] = []
    for index in range(1, CHAT_TURNS + 1):
        turn = chat_turn(index)
        window = turns[-SLIDING_WINDOW_TURNS:]
        session.send(f"turn {index}", _chat_request(ctx, _history(window), turn), turn.answer)
        turns.append(turn)
        session.clock.advance(TURN_GAP)
    return session


def budget_summary(ctx: ScenarioContext) -> Session:
    session = ctx.new_session("b: summarise at budget")
    summary = ""
    verbatim: list[Turn] = []
    for index in range(1, CHAT_TURNS + 1):
        turn = chat_turn(index)
        prefix = (_summary_messages(summary) if summary else ()) + _history(verbatim)
        if session.simulator.count_input_tokens(_chat_request(ctx, prefix, turn)) > SUMMARY_TOKEN_BUDGET:
            summary = _summarise(session, ctx, verbatim[:-VERBATIM_TURNS_AFTER_SUMMARY], summary)
            verbatim = verbatim[-VERBATIM_TURNS_AFTER_SUMMARY:]
            prefix = _summary_messages(summary) + _history(verbatim)
        session.send(f"turn {index}", _chat_request(ctx, prefix, turn), turn.answer)
        verbatim.append(turn)
        session.clock.advance(TURN_GAP)
    return session


def idle_summary(ctx: ScenarioContext) -> Session:
    session = ctx.new_session("c: summarise after idle gap")
    ttl = ctx.rules.for_generation(ctx.generation).default_ttl
    idle_gap = timedelta(minutes=ctx.rules.ttls[ttl].minutes + 5)
    summary = ""
    verbatim: list[Turn] = []
    for index in range(1, CHAT_TURNS + 1):
        turn = chat_turn(index)
        if index - 1 in IDLE_GAP_TURNS:
            # The cache has expired anyway, so rewriting the prefix now costs nothing extra.
            session.clock.advance(idle_gap)
            summary = _summarise(session, ctx, verbatim[:-VERBATIM_TURNS_AFTER_SUMMARY], summary)
            verbatim = verbatim[-VERBATIM_TURNS_AFTER_SUMMARY:]
        prefix = (_summary_messages(summary) if summary else ()) + _history(verbatim)
        session.send(f"turn {index}", _chat_request(ctx, prefix, turn), turn.answer)
        verbatim.append(turn)
        session.clock.advance(TURN_GAP)
    return session


def long_chat(ctx: ScenarioContext) -> ScenarioResult:
    sessions = [sliding_window(ctx), budget_summary(ctx), idle_summary(ctx)]
    notes = (
        f"{CHAT_TURNS} turns, one minute apart. (a) keeps the last {SLIDING_WINDOW_TURNS} turns, "
        "so the prefix changes after the system prompt on every turn once the window is full.",
        f"(b) summarises once the prompt passes {SUMMARY_TOKEN_BUDGET} tokens and keeps the last "
        f"{VERBATIM_TURNS_AFTER_SUMMARY} turns verbatim. (c) summarises only after idle gaps longer than the TTL.",
        "Summarisation calls are billed too; they appear as their own rows.",
    )
    return _result("long-chat", f"{CHAT_TURNS}-turn chat: three ways to keep the context bounded", sessions, notes)


# 5. RAG with explicit breakpoints.

RAG_QUESTIONS = (
    "How long does model Alpha-3 last on battery?",
    "Which warehouse ships model Beta-10?",
    "Which is heavier, Alpha-7 or Beta-7?",
    "List the battery life of Beta-1 and Alpha-1.",
)


def rag_ttl(ctx: ScenarioContext) -> ScenarioResult:
    generation = ctx.rules.for_generation(ctx.generation)
    is_explicit = generation.cache_policy is CachePolicyName.MESSAGE_BOUNDARY
    sessions = [_rag_session(ctx, ttl, is_explicit) for ttl in generation.allowed_ttls]
    notes: tuple[str, ...] = (f"Questions arrive {RAG_QUESTION_GAP} apart on the fake clock.",)
    if not is_explicit:
        notes += ("Explicit breakpoints need GPT-5.6 or later; this generation runs implicit caching instead.",)
    return _result("rag-ttl", "RAG: system prompt and two documents, compared across TTLs", sessions, notes)


def _rag_session(ctx: ScenarioContext, ttl: Ttl, is_explicit: bool) -> Session:
    session = ctx.new_session(f"ttl {ttl.value}")
    marker = Breakpoint(ttl=ttl) if is_explicit else None
    context = (
        system("Answer only from the documents. Be brief.\n" + support_policies(), marker),
        user(document("Alpha"), marker),
        user(document("Beta"), marker),
    )
    mode = CacheMode.EXPLICIT if is_explicit else CacheMode.IMPLICIT
    for question in RAG_QUESTIONS:
        request = SimRequest(model=ctx.sim_model, messages=(*context, user(question)), mode=mode, ttl=ttl)
        session.send(question, request, "The documents say so in one short sentence.")
        session.clock.advance(RAG_QUESTION_GAP)
    return session


# 6. Two tenants.


def tenants(ctx: ScenarioContext) -> ScenarioResult:
    session = ctx.new_session("shared simulator")
    messages = (system(support_policies()), user("What is policy 5?"))
    for key in ("tenant-a", "tenant-a", "tenant-b", "tenant-b"):
        request = SimRequest(model=ctx.sim_model, messages=messages, prompt_cache_key=key)
        session.send(f"prompt_cache_key={key}", request, "Order type 5 has a 12 day refund window.")
        session.clock.advance(TURN_GAP)
    notes = (
        "Modelled as separate namespaces per key. Real OpenAI caches are per organisation and the key only"
        " steers routing, so treat this as the isolation you get by giving each tenant its own key.",
    )
    return _result("tenants", "Two tenants send identical prompts", [session], notes)


# 7. Short prompt.


def short_prompt(ctx: ScenarioContext) -> ScenarioResult:
    session = ctx.new_session("under minimum")
    messages = (system("You are a helpful assistant."), user("What's the capital of France?"))
    request = SimRequest(model=ctx.sim_model, messages=messages)
    for label in ("first", "repeat"):
        session.send(label, request, "Paris.")
        session.clock.advance(TURN_GAP)
    minimum = ctx.rules.for_generation(ctx.generation).min_cacheable_tokens
    notes = (f"Prompts under {minimum} tokens are never cached or written.",)
    return _result("short-prompt", "A prompt under the cacheable minimum", [session], notes)


Scenario = Callable[[ScenarioContext], ScenarioResult]

SCENARIOS: dict[str, Scenario] = {
    "cold-warm-broken": cold_warm_broken,
    "question-wording": question_wording,
    "name-placement": name_placement,
    "long-chat": long_chat,
    "rag-ttl": rag_ttl,
    "tenants": tenants,
    "short-prompt": short_prompt,
}
