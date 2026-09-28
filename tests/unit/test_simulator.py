from datetime import timedelta
from typing import Any

import pytest

from prompt_cache_meter.rules import Generation, Ttl, UnknownModelError
from prompt_cache_meter.simulator import (
    Bill,
    CacheMode,
    FakeClock,
    SimMessage,
    SimRequest,
    SimRole,
    UnsupportedFeatureError,
    UnsupportedTtlError,
)
from tests.unit.simulator_fixtures import make_simulator, message

SYSTEM = message(SimRole.SYSTEM, 2000, seed="s")
QUESTION = message(SimRole.USER, 100, seed="q")


def new_request(*messages: SimMessage, **overrides: Any) -> SimRequest:
    return SimRequest(model="new-model", messages=messages, **overrides)


def legacy_request(*messages: SimMessage, **overrides: Any) -> SimRequest:
    return SimRequest(model="legacy-model", messages=messages, **overrides)


def send_twice(generation: Generation, first: SimRequest, second: SimRequest) -> Bill:
    simulator = make_simulator(generation)
    simulator.simulate(first, output_tokens=0)
    return simulator.simulate(second, output_tokens=0)


def test_identical_prompt_second_call_hits() -> None:
    request = new_request(SYSTEM, QUESTION)

    bill = send_twice(Generation.NEW, request, request)

    assert bill.cached_tokens == 2100
    assert bill.written_tokens == 0


def test_first_call_writes_whole_prompt_at_latest_user_message() -> None:
    bill = make_simulator(Generation.NEW).simulate(new_request(SYSTEM, QUESTION), output_tokens=0)

    assert bill.cached_tokens == 0
    assert bill.written_tokens_by_ttl == {Ttl.MINUTES_30: 2100}


def test_changed_first_token_misses() -> None:
    changed = message(SimRole.SYSTEM, 2000, seed="x")

    bill = send_twice(Generation.NEW, new_request(SYSTEM, QUESTION), new_request(changed, QUESTION))

    assert bill.cached_tokens == 0


def test_tool_change_misses() -> None:
    first = new_request(SYSTEM, QUESTION, tools=({"name": "lookup"},))
    second = new_request(SYSTEM, QUESTION, tools=({"name": "search"},))

    assert send_twice(Generation.NEW, first, second).cached_tokens == 0


def test_response_schema_change_misses() -> None:
    first = legacy_request(SYSTEM, QUESTION, response_schema={"type": "object"})
    second = legacy_request(SYSTEM, QUESTION, response_schema={"type": "array"})

    assert send_twice(Generation.LEGACY, first, second).cached_tokens == 0


def test_model_change_misses() -> None:
    first = new_request(SYSTEM, QUESTION)
    second = SimRequest(model="other-model", messages=(SYSTEM, QUESTION))

    assert send_twice(Generation.NEW, first, second).cached_tokens == 0


@pytest.mark.parametrize("generation", [Generation.LEGACY, Generation.NEW])
def test_prompt_under_minimum_never_caches(generation: Generation) -> None:
    model = "new-model" if generation is Generation.NEW else "legacy-model"
    request = SimRequest(model=model, messages=(message(SimRole.SYSTEM, 500), QUESTION))

    bill = send_twice(generation, request, request)

    assert bill.cached_tokens == 0
    assert bill.written_tokens == 0


def test_legacy_rounds_down_to_128() -> None:
    first = legacy_request(message(SimRole.SYSTEM, 1300), message(SimRole.USER, 10, seed="a"))
    second = legacy_request(message(SimRole.SYSTEM, 1300), message(SimRole.USER, 10, seed="b"))

    bill = send_twice(Generation.LEGACY, first, second)

    # 1,301 tokens match (the second message's role token too), reported as 1,280.
    assert bill.cached_tokens == 1280


def test_legacy_has_no_write_charge() -> None:
    bill = make_simulator(Generation.LEGACY).simulate(legacy_request(SYSTEM, QUESTION), output_tokens=0)

    assert bill.written_tokens == 0
    assert bill.uncached_tokens == 2100
    assert bill.cost_usd == pytest.approx(2100 / 1_000_000)


def test_new_implicit_does_not_read_inside_a_message() -> None:
    first = new_request(message(SimRole.SYSTEM, 1500), message(SimRole.USER, 600, seed="u"))
    extended = SimMessage(role=SimRole.USER, content=first.messages[1].content + " more")

    bill = send_twice(Generation.NEW, first, new_request(first.messages[0], extended))

    # 2,100 tokens match, but the only boundary inside the match is the end of the system block.
    assert bill.cached_tokens == 1500


def test_new_implicit_reads_system_block_end_when_question_changes() -> None:
    first = new_request(SYSTEM, message(SimRole.USER, 100, seed="a"))
    second = new_request(SYSTEM, message(SimRole.USER, 100, seed="b"))

    bill = send_twice(Generation.NEW, first, second)

    assert bill.cached_tokens == 2000
    assert bill.written_tokens_by_ttl == {Ttl.MINUTES_30: 100}


def test_new_implicit_reads_resume_at_earlier_user_message() -> None:
    first = new_request(SYSTEM, QUESTION)
    second = new_request(SYSTEM, QUESTION, message(SimRole.ASSISTANT, 100, "r"), message(SimRole.USER, 100, "f"))

    bill = send_twice(Generation.NEW, first, second)

    assert bill.cached_tokens == 2100
    assert bill.written_tokens_by_ttl == {Ttl.MINUTES_30: 200}


def test_write_billed_at_125x_for_30m() -> None:
    bill = make_simulator(Generation.NEW).simulate(new_request(SYSTEM, QUESTION), output_tokens=0)

    assert bill.cost_usd == pytest.approx(2100 * 1.25 / 1_000_000)


def test_12h_write_billed_at_250x() -> None:
    request = new_request(SYSTEM, QUESTION, ttl=Ttl.HOURS_12)

    bill = make_simulator(Generation.NEW).simulate(request, output_tokens=0)

    assert bill.written_tokens_by_ttl == {Ttl.HOURS_12: 2100}
    assert bill.cost_usd == pytest.approx(2100 * 2.5 / 1_000_000)


def test_output_tokens_billed_at_output_price() -> None:
    request = new_request(message(SimRole.USER, 10))

    bill = make_simulator(Generation.NEW).simulate(request, output_tokens=1000)

    assert bill.cost_usd == pytest.approx((10 + 1000 * 10) / 1_000_000)


def test_explicit_without_breakpoints_does_not_cache() -> None:
    request = new_request(SYSTEM, QUESTION, mode=CacheMode.EXPLICIT)

    bill = send_twice(Generation.NEW, request, request)

    assert bill.cached_tokens == 0
    assert bill.written_tokens == 0
    assert bill.uncached_tokens == 2100


def test_explicit_tail_after_last_breakpoint_billed_1x() -> None:
    system = message(SimRole.SYSTEM, 2000, has_breakpoint=True)
    request = new_request(system, QUESTION, mode=CacheMode.EXPLICIT)
    simulator = make_simulator(Generation.NEW)

    first = simulator.simulate(request, output_tokens=0)
    second = simulator.simulate(request, output_tokens=0)

    assert (first.written_tokens, first.uncached_tokens) == (2000, 100)
    assert (second.cached_tokens, second.uncached_tokens) == (2000, 100)


def test_more_than_four_writes_uses_latest_four() -> None:
    first = message(SimRole.SYSTEM, 1500, breakpoint_ttl=Ttl.HOURS_12)
    later = [message(SimRole.USER, 500, seed=f"u{index}", has_breakpoint=True) for index in range(4)]

    bill = make_simulator(Generation.NEW).simulate(new_request(first, *later, mode=CacheMode.EXPLICIT), 0)

    # Five breakpoints qualify; the earliest (the only 12h one) loses its write slot.
    assert bill.written_tokens_by_ttl == {Ttl.MINUTES_30: 3500}


def test_implicit_breakpoint_uses_one_write_slot() -> None:
    first = message(SimRole.SYSTEM, 1500, breakpoint_ttl=Ttl.HOURS_12)
    marked = [message(SimRole.USER, 500, seed=f"u{index}", has_breakpoint=True) for index in range(3)]
    latest = message(SimRole.USER, 500, seed="latest")

    bill = make_simulator(Generation.NEW).simulate(new_request(first, *marked, latest), 0)

    assert bill.written_tokens_by_ttl == {Ttl.MINUTES_30: 3500}


def test_30m_entry_expires_after_clock_advance() -> None:
    clock = FakeClock()
    simulator = make_simulator(Generation.NEW, clock)
    request = new_request(SYSTEM, QUESTION)

    simulator.simulate(request, 0)
    clock.advance(timedelta(minutes=31))

    assert simulator.simulate(request, 0).cached_tokens == 0


def test_30m_read_refreshes_expiry() -> None:
    clock = FakeClock()
    simulator = make_simulator(Generation.NEW, clock)
    request = new_request(SYSTEM, QUESTION)

    simulator.simulate(request, 0)
    clock.advance(timedelta(minutes=20))
    simulator.simulate(request, 0)
    clock.advance(timedelta(minutes=25))

    assert simulator.simulate(request, 0).cached_tokens == 2100


def test_12h_entry_is_not_refreshed_by_reads() -> None:
    clock = FakeClock()
    simulator = make_simulator(Generation.NEW, clock)
    request = new_request(SYSTEM, QUESTION, ttl=Ttl.HOURS_12)

    simulator.simulate(request, 0)
    clock.advance(timedelta(hours=11))
    assert simulator.simulate(request, 0).cached_tokens == 2100
    clock.advance(timedelta(hours=1, minutes=1))

    assert simulator.simulate(request, 0).cached_tokens == 0


def test_legacy_in_memory_expires_after_idle_gap() -> None:
    clock = FakeClock()
    simulator = make_simulator(Generation.LEGACY, clock)
    request = legacy_request(SYSTEM, QUESTION)

    simulator.simulate(request, 0)
    clock.advance(timedelta(minutes=11))

    assert simulator.simulate(request, 0).cached_tokens == 0


def test_legacy_24h_survives_two_hours() -> None:
    clock = FakeClock()
    simulator = make_simulator(Generation.LEGACY, clock)
    request = legacy_request(SYSTEM, QUESTION, ttl=Ttl.HOURS_24)

    simulator.simulate(request, 0)
    clock.advance(timedelta(hours=2))

    assert simulator.simulate(request, 0).cached_tokens == 2048


def test_cache_keys_are_isolated() -> None:
    simulator = make_simulator(Generation.NEW)

    simulator.simulate(new_request(SYSTEM, QUESTION, prompt_cache_key="tenant-a"), 0)
    other_tenant = simulator.simulate(new_request(SYSTEM, QUESTION, prompt_cache_key="tenant-b"), 0)
    same_tenant = simulator.simulate(new_request(SYSTEM, QUESTION, prompt_cache_key="tenant-a"), 0)

    assert other_tenant.cached_tokens == 0
    assert same_tenant.cached_tokens == 2100


def test_uncached_equals_input_minus_cached_minus_written() -> None:
    first = new_request(SYSTEM, message(SimRole.USER, 100, seed="a"))
    second = new_request(SYSTEM, message(SimRole.USER, 100, seed="a"), message(SimRole.USER, 50, seed="b"))

    bill = send_twice(Generation.NEW, first, second)

    assert bill.uncached_tokens == bill.input_tokens - bill.cached_tokens - bill.written_tokens


def test_valid_12h_entry_stays_a_read_under_a_later_30m_write() -> None:
    # Docs example: valid 12h entry through 2,000 tokens, "30m" breakpoint at 5,000.
    head = message(SimRole.SYSTEM, 2000, breakpoint_ttl=Ttl.HOURS_12)
    tail = message(SimRole.USER, 3000, breakpoint_ttl=Ttl.MINUTES_30)
    first = new_request(head, mode=CacheMode.EXPLICIT)

    bill = send_twice(Generation.NEW, first, new_request(head, tail, mode=CacheMode.EXPLICIT))

    assert bill.cached_tokens == 2000
    assert bill.written_tokens_by_ttl == {Ttl.MINUTES_30: 3000}


def test_upgrading_30m_entry_to_12h_charges_a_12h_write() -> None:
    first = new_request(message(SimRole.SYSTEM, 2000, has_breakpoint=True), mode=CacheMode.EXPLICIT)
    second = new_request(message(SimRole.SYSTEM, 2000, breakpoint_ttl=Ttl.HOURS_12), mode=CacheMode.EXPLICIT)

    bill = send_twice(Generation.NEW, first, second)

    assert bill.cached_tokens == 0
    assert bill.written_tokens_by_ttl == {Ttl.HOURS_12: 2000}


def test_docs_usage_example_costs_0_01515() -> None:
    # Docs example: 12,000 input = 4,000 reads + 4,000 12h writes + 3,000 30m writes + 1,000 ordinary.
    system = message(SimRole.SYSTEM, 4000, breakpoint_ttl=Ttl.HOURS_12)
    simulator = make_simulator(Generation.NEW)
    simulator.simulate(new_request(system, mode=CacheMode.EXPLICIT), 0)
    second = new_request(
        system,
        message(SimRole.USER, 4000, seed="a", breakpoint_ttl=Ttl.HOURS_12),
        message(SimRole.USER, 3000, seed="b", breakpoint_ttl=Ttl.MINUTES_30),
        message(SimRole.USER, 1000, seed="c"),
        mode=CacheMode.EXPLICIT,
    )

    bill = simulator.simulate(second, output_tokens=0)

    assert bill.cached_tokens == 4000
    assert bill.written_tokens_by_ttl == {Ttl.HOURS_12: 4000, Ttl.MINUTES_30: 3000}
    assert bill.uncached_tokens == 1000
    assert bill.cost_usd == pytest.approx(0.01515)


def test_legacy_breakpoint_raises_unsupported_feature() -> None:
    request = legacy_request(message(SimRole.SYSTEM, 2000, has_breakpoint=True))

    with pytest.raises(UnsupportedFeatureError):
        make_simulator(Generation.LEGACY).simulate(request, 0)


def test_ttl_from_other_generation_raises_unsupported_ttl() -> None:
    with pytest.raises(UnsupportedTtlError, match="30m"):
        make_simulator(Generation.LEGACY).simulate(legacy_request(SYSTEM, ttl=Ttl.MINUTES_30), 0)


def test_unknown_model_raises_unknown_model_error() -> None:
    request = SimRequest(model="mystery", messages=(SYSTEM,))

    with pytest.raises(UnknownModelError, match="mystery"):
        make_simulator(Generation.NEW).simulate(request, 0)


def test_hit_rate_is_cached_over_input() -> None:
    first = new_request(SYSTEM, message(SimRole.USER, 100, seed="a"))
    second = new_request(SYSTEM, message(SimRole.USER, 500, seed="b"))

    assert send_twice(Generation.NEW, first, second).hit_rate == pytest.approx(2000 / 2500)


def test_developer_block_end_is_a_read_point() -> None:
    developer = message(SimRole.DEVELOPER, 2000, seed="d")
    first = new_request(developer, message(SimRole.USER, 100, seed="a"))

    bill = send_twice(Generation.NEW, first, new_request(developer, message(SimRole.USER, 100, seed="b")))

    assert bill.cached_tokens == 2000


def test_last_tool_response_of_a_group_is_a_read_point() -> None:
    tools = (message(SimRole.TOOL, 100, seed="t1"), message(SimRole.TOOL, 100, seed="t2"))
    call = message(SimRole.ASSISTANT, 50, seed="call")
    first = new_request(SYSTEM, QUESTION, call, *tools, message(SimRole.ASSISTANT, 50, seed="r1"))
    second = new_request(SYSTEM, QUESTION, call, *tools, message(SimRole.ASSISTANT, 50, seed="r2"))
    simulator = make_simulator(Generation.NEW)
    simulator.simulate(first, 0)
    # The first request's latest eligible boundary was the second tool response, so that is what was written.
    bill = simulator.simulate(second, 0)

    assert bill.cached_tokens == 2000 + 100 + 50 + 200


def test_prefix_option_change_misses() -> None:
    first = new_request(SYSTEM, QUESTION, prefix_options={"reasoning_effort": "low"})
    second = new_request(SYSTEM, QUESTION, prefix_options={"reasoning_effort": "high"})

    assert send_twice(Generation.NEW, first, second).cached_tokens == 0
