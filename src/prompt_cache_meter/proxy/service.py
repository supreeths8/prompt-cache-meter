from prompt_cache_meter.llm.ports import Tokenizer
from prompt_cache_meter.ports import Clock
from prompt_cache_meter.proxy.models import ChatCompletionBody, ProxyError, ProxyRecord, ReplySummary
from prompt_cache_meter.proxy.translate import parse_body, translate
from prompt_cache_meter.rules import CacheRules, Generation, RulesError
from prompt_cache_meter.simulator import Bill, CacheSimulator, SimulatorError


class SimulationService:
    """Bills each successful proxied call with the simulator. Never raises: problems become warnings."""

    def __init__(
        self,
        rules: CacheRules,
        generation: Generation,
        tokenizer: Tokenizer,
        clock: Clock,
        price_as: str,
    ) -> None:
        self._rules = rules
        self._generation = generation
        self._tokenizer = tokenizer
        self._clock = clock
        self._price_as = price_as
        # One simulator for the proxy's lifetime, so the cache carries across calls the way OpenAI's does.
        self._simulator = CacheSimulator(rules, generation, tokenizer, clock)

    def observe(self, raw_body: bytes, reply: ReplySummary, latency_seconds: float, is_stream: bool) -> ProxyRecord:
        body: ChatCompletionBody | None = None
        bill: Bill | None = None
        warnings: list[str] = []
        try:
            body = parse_body(raw_body)
            bill = self._bill(body, reply, warnings)
        except (ProxyError, RulesError, SimulatorError) as error:
            warnings.append(f"not simulated: {type(error).__name__}: {error}")
        return ProxyRecord(
            time=self._clock.now(),
            requested_model=body.model if body else None,
            priced_model=bill.model if bill else None,
            generation=self._generation,
            prompt_cache_key=body.prompt_cache_key if body else None,
            is_stream=is_stream,
            latency_seconds=latency_seconds,
            bill=bill,
            upstream_usage=reply.usage,
            warnings=tuple(warnings),
        )

    def _bill(self, body: ChatCompletionBody, reply: ReplySummary, warnings: list[str]) -> Bill:
        generation = self._rules.for_generation(self._generation)
        translation = translate(body, self._rules.prices.keys(), self._price_as, generation.allowed_ttls)
        warnings.extend(translation.warnings)
        output_tokens = len(self._tokenizer.encode(reply.text))
        return self._simulator.simulate(translation.request, output_tokens=output_tokens)
