import pytest

from prompt_cache_meter.bootstrap import build_chat_model
from prompt_cache_meter.config import Provider, Settings
from prompt_cache_meter.llm.models import ChatRequest, Message, Role


@pytest.mark.integration
def test_ollama_answers_a_prompt() -> None:
    chat_model = build_chat_model(Provider.OLLAMA, Settings())

    response = chat_model.complete(ChatRequest(messages=(Message(role=Role.USER, content="Reply with: ok"),)))

    assert response.text
    assert response.usage.input_tokens > 0
