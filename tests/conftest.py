import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--integration", action="store_true", help="also run tests that call a real model provider")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Skip rather than deselect, so running an integration test file explains itself instead of doing nothing.
    if config.getoption("--integration") or config.getoption("markexpr") == "integration":
        return
    skip_integration = pytest.mark.skip(reason="calls a real provider; run with --integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
