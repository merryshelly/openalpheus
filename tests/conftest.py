"""Shared test fixtures."""

import pytest
from openalph.provider import _client_cache


@pytest.fixture(autouse=True)
def clear_provider_client_cache():
    """Clear the provider client cache before each test.

    This prevents cross-test pollution: tests that patch anthropic.AsyncAnthropic
    or openai.AsyncOpenAI expect a fresh client to be constructed, but the cache
    would return a previously-created (potentially stale or mocked) client.
    """
    _client_cache.clear()
    yield
    _client_cache.clear()


@pytest.fixture(autouse=True)
def ensure_test_workspace_dirs():
    """Ensure hardcoded workspace paths used by config tests exist.

    Several test TOMLs reference /tmp/test or /tmp/test-workspace.
    Config validation checks that workspace.path is a real directory.
    """
    from pathlib import Path
    for d in ("/tmp/test", "/tmp/test-workspace"):
        Path(d).mkdir(exist_ok=True)
