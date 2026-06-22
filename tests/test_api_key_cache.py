"""Tests for web_search API-key resolution caching (_resolve_cached_api_key).

Background
----------
web_search resolves its API key via `api_key_cmd` (e.g. `op read op://...`) inside
execute_tool. Historically this ran on EVERY call, so each search hit the 1Password
API. Under burst load (parallel sub-agents each firing web_search), this exhausts the
service-account rate limit; `op` then returns empty and web_search reports
"no API key configured" even though the credential is correct.

These tests pin the caching contract that fixes it:
  * resolve once, reuse within a TTL (no subprocess on a cache hit)
  * cache only non-empty results (never negative-cache a transient failure)
  * stale-while-error: once resolved, a later failed/empty resolution falls back to
    the last known-good value, so web_search keeps working through a throttle window
  * distinct commands are cached independently
  * ttl=0 disables hit-caching (re-resolves every call) but stale fallback still works
"""

import subprocess

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

import openalph.tools as tools
from openalph.tools import _resolve_cached_api_key, execute_tool


@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts and ends with an empty module-level cache."""
    tools._api_key_cache.clear()
    yield
    tools._api_key_cache.clear()


def _completed(stdout="", returncode=0, stderr=""):
    """Build a subprocess.CompletedProcess like subprocess.run would return."""
    return subprocess.CompletedProcess(
        args="cmd", returncode=returncode, stdout=stdout, stderr=stderr
    )


def _mock_brave_client():
    """A mocked httpx.AsyncClient context manager returning one Brave result."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "web": {"results": [{"title": "T", "url": "https://t.com", "description": "d"}]}
    }
    resp.raise_for_status = MagicMock()
    client = AsyncMock()
    client.get.return_value = resp
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


# ---------------------------------------------------------------------------
# Unit tests: _resolve_cached_api_key
# ---------------------------------------------------------------------------

class TestResolveCachedApiKey:

    def test_cache_miss_resolves_and_strips(self):
        """First resolution runs the command and returns stripped stdout."""
        with patch("openalph.tools.subprocess.run",
                   return_value=_completed(stdout="  SECRET\n")) as run:
            key = _resolve_cached_api_key("op read x", ttl=3600)
        assert key == "SECRET"
        assert run.call_count == 1

    def test_cache_hit_within_ttl_does_not_rerun(self):
        """A second call within the TTL is served from cache (no subprocess)."""
        with patch("openalph.tools.subprocess.run",
                   return_value=_completed(stdout="SECRET")) as run, \
             patch("openalph.tools.time.monotonic", side_effect=[100.0, 200.0]):
            k1 = _resolve_cached_api_key("op read x", ttl=3600)
            k2 = _resolve_cached_api_key("op read x", ttl=3600)
        assert k1 == k2 == "SECRET"
        assert run.call_count == 1

    def test_cache_expiry_triggers_rerun(self):
        """After the TTL elapses, the command runs again."""
        with patch("openalph.tools.subprocess.run",
                   side_effect=[_completed(stdout="K1"), _completed(stdout="K2")]) as run, \
             patch("openalph.tools.time.monotonic", side_effect=[100.0, 4100.0]):
            k1 = _resolve_cached_api_key("op read x", ttl=3600)
            k2 = _resolve_cached_api_key("op read x", ttl=3600)
        assert k1 == "K1"
        assert k2 == "K2"
        assert run.call_count == 2

    def test_monotonic_called_once_per_invocation(self):
        """Implementation must read the clock exactly once per call.

        If it reads more than once, the side_effect list is exhausted and this
        raises StopIteration -- a guard against accidental double reads that would
        desync the TTL math.
        """
        with patch("openalph.tools.subprocess.run",
                   return_value=_completed(stdout="SECRET")), \
             patch("openalph.tools.time.monotonic", side_effect=[100.0]) as mono:
            _resolve_cached_api_key("op read x", ttl=3600)
        assert mono.call_count == 1

    def test_empty_result_no_prior_cache_returns_empty(self):
        """A throttled/empty resolution with no prior value returns '' and caches nothing."""
        with patch("openalph.tools.subprocess.run",
                   return_value=_completed(stdout="", returncode=1, stderr="Too many requests")):
            key = _resolve_cached_api_key("op read x", ttl=3600)
        assert key == ""
        assert "op read x" not in tools._api_key_cache

    def test_empty_result_serves_stale_cached_value(self):
        """Stale-while-error: after a good resolve, an empty re-resolve serves the cached key."""
        with patch("openalph.tools.subprocess.run",
                   side_effect=[_completed(stdout="SECRET"),
                                _completed(stdout="", returncode=1, stderr="rate limited")]) as run, \
             patch("openalph.tools.time.monotonic", side_effect=[100.0, 4100.0]):
            k1 = _resolve_cached_api_key("op read x", ttl=3600)
            k2 = _resolve_cached_api_key("op read x", ttl=3600)
        assert k1 == "SECRET"
        assert k2 == "SECRET"
        assert run.call_count == 2

    def test_timeout_serves_stale_cached_value(self):
        """A TimeoutExpired during re-resolve falls back to the cached key."""
        with patch("openalph.tools.subprocess.run",
                   side_effect=[_completed(stdout="SECRET"),
                                subprocess.TimeoutExpired("op read x", 10)]), \
             patch("openalph.tools.time.monotonic", side_effect=[100.0, 4100.0]):
            k1 = _resolve_cached_api_key("op read x", ttl=3600)
            k2 = _resolve_cached_api_key("op read x", ttl=3600)
        assert k1 == "SECRET"
        assert k2 == "SECRET"

    def test_exception_serves_stale_cached_value(self):
        """Any subprocess exception during re-resolve falls back to the cached key."""
        with patch("openalph.tools.subprocess.run",
                   side_effect=[_completed(stdout="SECRET"), OSError("boom")]), \
             patch("openalph.tools.time.monotonic", side_effect=[100.0, 4100.0]):
            k1 = _resolve_cached_api_key("op read x", ttl=3600)
            k2 = _resolve_cached_api_key("op read x", ttl=3600)
        assert k1 == "SECRET"
        assert k2 == "SECRET"

    def test_timeout_no_prior_cache_returns_empty(self):
        """A timeout on the very first resolution returns '' (nothing to fall back to)."""
        with patch("openalph.tools.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("op read x", 10)):
            key = _resolve_cached_api_key("op read x", ttl=3600)
        assert key == ""

    def test_recovers_after_initial_failure(self):
        """A transient first-call failure is not negative-cached; the next call resolves."""
        with patch("openalph.tools.subprocess.run",
                   side_effect=[_completed(stdout="", returncode=1), _completed(stdout="GOOD")]) as run, \
             patch("openalph.tools.time.monotonic", side_effect=[100.0, 101.0]):
            k1 = _resolve_cached_api_key("op read x", ttl=3600)
            k2 = _resolve_cached_api_key("op read x", ttl=3600)
        assert k1 == ""
        assert k2 == "GOOD"
        assert run.call_count == 2

    def test_distinct_commands_cached_independently(self):
        """Different api_key_cmd strings get independent cache entries."""
        with patch("openalph.tools.subprocess.run",
                   side_effect=[_completed(stdout="KEY_A"), _completed(stdout="KEY_B")]) as run, \
             patch("openalph.tools.time.monotonic", side_effect=[100.0, 100.0]):
            ka = _resolve_cached_api_key("cmd A", ttl=3600)
            kb = _resolve_cached_api_key("cmd B", ttl=3600)
        assert ka == "KEY_A"
        assert kb == "KEY_B"
        assert run.call_count == 2

    def test_ttl_zero_reresolves_each_call(self):
        """ttl=0 disables hit-caching: every call re-runs the command."""
        with patch("openalph.tools.subprocess.run",
                   side_effect=[_completed(stdout="K1"), _completed(stdout="K2")]) as run, \
             patch("openalph.tools.time.monotonic", side_effect=[100.0, 100.0]):
            k1 = _resolve_cached_api_key("op read x", ttl=0)
            k2 = _resolve_cached_api_key("op read x", ttl=0)
        assert k1 == "K1"
        assert k2 == "K2"
        assert run.call_count == 2

    def test_default_ttl_used_when_none(self):
        """ttl=None falls back to the module default (long, positive) and caches a hit."""
        assert tools._DEFAULT_API_KEY_CACHE_TTL > 0
        with patch("openalph.tools.subprocess.run",
                   return_value=_completed(stdout="SECRET")) as run, \
             patch("openalph.tools.time.monotonic", side_effect=[100.0, 200.0]):
            k1 = _resolve_cached_api_key("op read x")
            k2 = _resolve_cached_api_key("op read x")
        assert k1 == k2 == "SECRET"
        assert run.call_count == 1

    def test_nonzero_returncode_not_used_or_cached(self):
        """A nonzero exit code never yields a key, even with non-empty stdout."""
        with patch("openalph.tools.subprocess.run",
                   side_effect=[_completed(stdout="ERROR_TEXT", returncode=1),
                                _completed(stdout="GOOD", returncode=0)]) as run, \
             patch("openalph.tools.time.monotonic", side_effect=[100.0, 101.0]):
            k1 = _resolve_cached_api_key("op read x", ttl=3600)
            k2 = _resolve_cached_api_key("op read x", ttl=3600)
        assert k1 == ""                 # nonzero rc -> stdout ignored, not negative-cached
        assert k2 == "GOOD"             # next call re-resolves to success
        assert run.call_count == 2
        assert tools._api_key_cache["op read x"][0] == "GOOD"

    def test_nonzero_returncode_serves_stale_not_error_text(self):
        """A nonzero exit with non-empty stdout must not overwrite a good cached key."""
        with patch("openalph.tools.subprocess.run",
                   side_effect=[_completed(stdout="GOODKEY", returncode=0),
                                _completed(stdout="ERROR_TEXT", returncode=1)]), \
             patch("openalph.tools.time.monotonic", side_effect=[100.0, 4100.0]):
            k1 = _resolve_cached_api_key("op read x", ttl=3600)
            k2 = _resolve_cached_api_key("op read x", ttl=3600)
        assert k1 == "GOODKEY"
        assert k2 == "GOODKEY"          # stale good key, NOT the error text on stdout

    def test_negative_ttl_clamped_to_zero(self):
        """A negative TTL is clamped to 0 (re-resolves each call)."""
        with patch("openalph.tools.subprocess.run",
                   side_effect=[_completed(stdout="K1"), _completed(stdout="K2")]) as run, \
             patch("openalph.tools.time.monotonic", side_effect=[100.0, 100.0]):
            k1 = _resolve_cached_api_key("op read x", ttl=-5)
            k2 = _resolve_cached_api_key("op read x", ttl=-5)
        assert k1 == "K1"
        assert k2 == "K2"
        assert run.call_count == 2

    def test_whitespace_only_stdout_not_cached(self):
        """stdout that is only whitespace strips to empty and is not cached."""
        with patch("openalph.tools.subprocess.run",
                   return_value=_completed(stdout="   \n\t ")):
            key = _resolve_cached_api_key("op read x", ttl=3600)
        assert key == ""
        assert "op read x" not in tools._api_key_cache

    def test_stale_replaced_on_successful_reresolution(self):
        """After serving stale through a failure, a later success replaces the cache."""
        with patch("openalph.tools.subprocess.run",
                   side_effect=[_completed(stdout="OLD"),
                                _completed(stdout="", returncode=1),
                                _completed(stdout="NEW")]) as run, \
             patch("openalph.tools.time.monotonic", side_effect=[100.0, 4100.0, 4300.0]):
            k1 = _resolve_cached_api_key("op read x", ttl=3600)
            k2 = _resolve_cached_api_key("op read x", ttl=3600)
            k3 = _resolve_cached_api_key("op read x", ttl=3600)
        assert k1 == "OLD"
        assert k2 == "OLD"              # stale served during failure
        assert k3 == "NEW"              # fresh resolution replaces stale
        assert run.call_count == 3

    def test_stale_backoff_serves_cached_without_rerunning(self):
        """After a failed re-resolve serves stale, a call within the backoff window
        is a cache hit and does NOT re-run the (doomed, event-loop-blocking)
        subprocess -- the key liveness property under sustained throttling."""
        with patch("openalph.tools.subprocess.run",
                   side_effect=[_completed(stdout="GOOD"),
                                _completed(stdout="", returncode=1)]) as run, \
             patch("openalph.tools.time.monotonic", side_effect=[100.0, 4100.0, 4110.0]):
            k1 = _resolve_cached_api_key("op read x", ttl=3600)  # GOOD cached
            k2 = _resolve_cached_api_key("op read x", ttl=3600)  # expired+fail -> stale, backoff
            k3 = _resolve_cached_api_key("op read x", ttl=3600)  # within backoff -> hit
        assert k1 == k2 == k3 == "GOOD"
        assert run.call_count == 2      # third call served from cache, no 3rd subprocess

    def test_ttl_zero_stale_while_error(self):
        """With ttl=0, a failed re-resolve still serves the prior cached value."""
        with patch("openalph.tools.subprocess.run",
                   side_effect=[_completed(stdout="K1"),
                                _completed(stdout="", returncode=1)]) as run, \
             patch("openalph.tools.time.monotonic", side_effect=[100.0, 100.0]):
            k1 = _resolve_cached_api_key("op read x", ttl=0)
            k2 = _resolve_cached_api_key("op read x", ttl=0)
        assert k1 == "K1"
        assert k2 == "K1"               # stale served despite ttl=0
        assert run.call_count == 2


# ---------------------------------------------------------------------------
# Integration tests: execute_tool("web_search", ...)
# ---------------------------------------------------------------------------

class TestExecuteToolWebSearchCaching:

    @pytest.mark.asyncio
    async def test_web_search_resolves_key_once_across_calls(self):
        """Two web_search calls resolve the api_key_cmd only once (cache hit on #2)."""
        with patch("openalph.tools.subprocess.run",
                   return_value=_completed(stdout="RESOLVED")) as run, \
             patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _mock_brave_client()
            cfg = {"api_key_cmd": "op read x"}
            r1 = await execute_tool("web_search", {"query": "a"}, cfg, None)
            r2 = await execute_tool("web_search", {"query": "b"}, cfg, None)
        assert r1.is_error is False
        assert r2.is_error is False
        assert run.call_count == 1

    @pytest.mark.asyncio
    async def test_web_search_direct_api_key_skips_resolution(self):
        """A direct api_key never invokes the resolver subprocess."""
        with patch("openalph.tools.subprocess.run") as run, \
             patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _mock_brave_client()
            r = await execute_tool("web_search", {"query": "a"}, {"api_key": "DIRECT"}, None)
        assert r.is_error is False
        assert run.call_count == 0

    @pytest.mark.asyncio
    async def test_web_search_respects_configured_ttl_zero(self):
        """api_key_cache_ttl=0 in tool config re-resolves on every call."""
        with patch("openalph.tools.subprocess.run",
                   side_effect=[_completed(stdout="K1"), _completed(stdout="K2")]) as run, \
             patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _mock_brave_client()
            cfg = {"api_key_cmd": "op read x", "api_key_cache_ttl": 0}
            await execute_tool("web_search", {"query": "a"}, cfg, None)
            await execute_tool("web_search", {"query": "b"}, cfg, None)
        assert run.call_count == 2
