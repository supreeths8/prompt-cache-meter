import pytest
from pydantic import ValidationError

from prompt_cache_meter.llm.models import ChatRequest, Message, Role


def test_request_rejects_empty_messages() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(messages=())


def test_request_rejects_temperature_above_two() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(messages=(Message(role=Role.USER, content="hi"),), temperature=2.5)


def test_message_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "tool", "content": "hi"})
