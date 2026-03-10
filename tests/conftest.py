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


# Tests use tmp_path (built-in pytest fixture) and construct
# their own configs/workspaces as needed.
