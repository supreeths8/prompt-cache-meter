import copy
import json
from pathlib import Path

import pytest

from prompt_cache_meter.llm.models import ChatRequest, ChatResponse, Usage
from prompt_cache_meter.report import save_result
from prompt_cache_meter.rules import CacheRules, Generation
from prompt_cache_meter.scenarios import SCENARIOS, ScenarioContext, cold_warm_broken, tenants
from tests.unit.simulator_fixtures import RULES, WordTokenizer


class FakeChatModel:
    def __init__(self, prompt_tokens: int = 50) -> None:
        self.requests: list[ChatRequest] = []
        self._prompt_tokens = prompt_tokens

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        usage = Usage(input_tokens=self._prompt_tokens, output_tokens=3, cached_input_tokens=0)
        return ChatResponse(text="a short reply", model="fake-ollama", usage=usage, latency_seconds=0.5)


def make_context(
    chat_model: FakeChatModel | None = None, context_length: int = 100_000, generation: Generation = Generation.NEW
) -> ScenarioContext:
    raw = copy.deepcopy(RULES)
    # The word tokenizer undercounts compared with tiktoken, so lower the minimum to keep prompts cacheable.
    for generation_rules in raw["generations"].values():
        generation_rules["min_cacheable_tokens"] = 256
    return ScenarioContext(
        rules=CacheRules.model_validate(raw),
        generation=generation,
        tokenizer=WordTokenizer(),
        context_length=context_length,
        chat_model=chat_model,
    )


@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("generation", [Generation.LEGACY, Generation.NEW])
def test_every_scenario_runs_in_dry_run(name: str, generation: Generation) -> None:
    result = SCENARIOS[name](make_context(generation=generation))

    assert result.rows
    assert all(row.ollama is None for row in result.rows)


def test_warm_request_hits_and_timestamp_breaks_cache() -> None:
    cold, warm, broken = cold_warm_broken(make_context()).rows

    assert cold.bill.cached_tokens == 0
    assert warm.bill.cached_tokens > 0
    assert broken.bill.cached_tokens == 0


def test_second_tenant_does_not_reuse_first_tenants_cache() -> None:
    rows = tenants(make_context()).rows

    assert [row.bill.cached_tokens > 0 for row in rows] == [False, True, False, True]


def test_live_run_records_ollama_observation() -> None:
    chat_model = FakeChatModel()

    rows = cold_warm_broken(make_context(chat_model)).rows

    assert len(chat_model.requests) == 3
    assert all(row.ollama is not None and row.ollama.latency_seconds == 0.5 for row in rows)


def test_prompt_at_context_length_warns_of_truncation() -> None:
    rows = cold_warm_broken(make_context(FakeChatModel(prompt_tokens=4096), context_length=4096)).rows

    assert any("truncated" in warning for warning in rows[0].warnings)


def test_prompt_near_context_length_warns_before_calling() -> None:
    rows = cold_warm_broken(make_context(context_length=500)).rows

    assert any("OLLAMA_CONTEXT_LENGTH" in warning for warning in rows[0].warnings)


def test_saved_result_has_one_line_per_call_with_tokens_and_model(tmp_path: Path) -> None:
    result = cold_warm_broken(make_context(FakeChatModel()))

    lines = save_result(result, tmp_path).read_text(encoding="utf-8").splitlines()

    first = json.loads(lines[0])
    assert len(lines) == 3
    assert first["simulated"]["model"] == "new-model"
    assert first["ollama"]["model"] == "fake-ollama"
    assert "cached_tokens" in first["simulated"]
