import copy
from pathlib import Path

import pytest

from prompt_cache_meter.rules import CacheRules, Generation, InvalidRulesError, Ttl, load_rules
from tests.unit.simulator_fixtures import RULES


def test_shipped_rules_file_loads() -> None:
    rules = load_rules()

    assert rules.for_generation(Generation.NEW).default_ttl is Ttl.MINUTES_30
    assert rules.for_generation(Generation.LEGACY).reported_multiple_tokens == 128


def test_missing_rules_file_raises_invalid_rules(tmp_path: Path) -> None:
    with pytest.raises(InvalidRulesError):
        load_rules(tmp_path / "missing.toml")


def test_malformed_toml_raises_invalid_rules(tmp_path: Path) -> None:
    path = tmp_path / "rules.toml"
    path.write_text("[tokens\n", encoding="utf-8")

    with pytest.raises(InvalidRulesError):
        load_rules(path)


def test_default_ttl_outside_allowed_ttls_is_rejected() -> None:
    raw = copy.deepcopy(RULES)
    raw["generations"]["new"]["default_ttl"] = "24h"

    with pytest.raises(ValueError, match="allowed_ttls"):
        CacheRules.model_validate(raw)


def test_generation_without_price_is_rejected() -> None:
    raw = copy.deepcopy(RULES)
    raw["generations"]["new"]["default_model"] = "unpriced"

    with pytest.raises(ValueError, match="no price"):
        CacheRules.model_validate(raw)
