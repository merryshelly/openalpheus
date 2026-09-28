"""STT tool tests (im7t.68).

Contract (spec: memory/projects/openalph/specs/im7t.68-tts-stt-native-tools-spec.md):

  stt(path, language=None, prompt=None, timestamps=False, tool_config=None,
      agent_config=None, callbacks=None) -> ToolResult

  - endpoint/language/prompt/limits come from workspace/tools/stt.toml [config]; endpoint
    defaults to "" and an enabled-but-unconfigured tool FAILS LOUDLY with steering naming
    the TOML path.
  - `path` accepts a workspace-relative path, an absolute path, or a pasted
    `[media: media/<hash>/<file> (mime, size)]` tag (models copy the tag verbatim).
  - multipart POST: the audio in `files={"file": (name, bytes, mime)}`, scalar fields in
    `data={...}` — language/prompt only when resolved non-empty (prompt="off" is passed
    through verbatim as the service's opt-out), response_format=verbose_json when
    timestamps else json.

Mocking boundary: httpx (HTTP) only.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from openalph.config import AgentConfig, ProviderConfig
from openalph.tools import BUILTIN_TOOLS, discover_tools, execute_tool
from openalph.tools.speech import stt

ENDPOINT = "http://10.0.20.104:8001/v1/audio/transcriptions"


# --- fixtures / helpers ------------------------------------------------------


def make_agent_config(tmp_path, **kwargs):
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={
            "anthropic": ProviderConfig(
                key="anthropic", type="anthropic", api_key="sk-test",
                base_url=None, quirks=[],
            )
        },
        workspace=tmp_path,
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def cfg(**overrides):
    base = dict(BUILTIN_TOOLS["stt"]["config"])
    base.update(overrides)
    return base


def make_response(status_code=200, content=b"", json_data=None, text=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.text = text if text is not None else content.decode("utf-8", "replace")
    resp.json = MagicMock(return_value=json_data)
    resp.headers = {"content-type": "application/json"}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )
    return resp


def stream_cm(resp, chunk_size=0):
    """Async context manager over a mocked response exposing aiter_bytes().

    The tool reads response bodies with a bounded stream (so a broken endpoint
    cannot OOM the process), so the mock models the streaming API.
    """
    payload = resp.content if isinstance(resp.content, (bytes, bytearray)) else b""
    # The tool reads raw bytes and parses JSON itself, so a mocked JSON response
    # must arrive as the serialized body (json_data wins over the placeholder
    # `content` an audio-oriented helper defaults to).
    json_value = getattr(getattr(resp, "json", None), "return_value", None)
    if json_value is not None:
        import json as _json
        payload = _json.dumps(json_value).encode("utf-8")
    size = chunk_size or max(len(payload), 1)

    async def _aiter():
        for i in range(0, len(payload), size):
            yield payload[i:i + size]

    resp.aiter_bytes = _aiter
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def make_client(responses=None, side_effect=None, chunk_size=0):
    client = AsyncMock()
    if side_effect is not None:
        client.stream = MagicMock(side_effect=side_effect)
    elif responses is not None:
        client.stream = MagicMock(side_effect=[stream_cm(r, chunk_size) for r in responses])
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def patch_httpx(client):
    mock_cls = MagicMock(return_value=client)
    return patch("openalph.tools.speech.httpx.AsyncClient", mock_cls), mock_cls


def audio_file(tmp_path, rel="media/abc123/voice.ogg", data=b"OGGDATA"):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


async def run_stt(tmp_path, path, client=None, **overrides):
    config = cfg(endpoint=ENDPOINT)
    config.update(overrides)
    if client is None:
        client = make_client(responses=[make_response(json_data={"text": "hello"})])
    patcher, _ = patch_httpx(client)
    with patcher:
        return await stt(
            path=path, tool_config=config, agent_config=make_agent_config(tmp_path),
        )


def posted(client, index=0):
    return client.stream.call_args_list[index].kwargs


# --- registry defaults -------------------------------------------------------


def test_registry_defaults_match_spec():
    assert BUILTIN_TOOLS["stt"]["config"] == {
        "endpoint": "",
        "timeout": 300.0,
        "language": "",
        "prompt": "",
        "max_bytes": 104857600,
    }


def test_registry_schema_requires_path():
    schema = BUILTIN_TOOLS["stt"]["parameters"]
    assert schema["required"] == ["path"]
    assert set(schema["properties"]) == {"path", "language", "prompt", "timestamps"}


def test_description_carries_steering_and_media_tag_note():
    desc = BUILTIN_TOOLS["stt"]["description"]
    assert "stt.toml" in desc
    assert "endpoint" in desc
    assert "media" in desc.lower()
    assert "IMPORTANT" in desc or "NEVER" in desc


def test_toml_config_overlays_defaults(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "stt.toml").write_text(
        f'[config]\nendpoint = "{ENDPOINT}"\nprompt = "Merry, Kokoro, Tailscale"\n'
    )
    tools = {t.name: t for t in discover_tools(tmp_path)}
    assert tools["stt"].config["endpoint"] == ENDPOINT
    assert tools["stt"].config["prompt"] == "Merry, Kokoro, Tailscale"
    assert tools["stt"].config["language"] == ""


# --- configuration failures --------------------------------------------------


@pytest.mark.asyncio
async def test_unconfigured_endpoint_fails_loudly(tmp_path):
    audio_file(tmp_path)
    result = await run_stt(tmp_path, "media/abc123/voice.ogg", endpoint="")
    assert result.is_error is True
    assert "no endpoint configured" in result.content
    assert "workspace/tools/stt.toml" in result.content


@pytest.mark.asyncio
async def test_non_string_endpoint_fails_loudly(tmp_path):
    audio_file(tmp_path)
    result = await run_stt(tmp_path, "media/abc123/voice.ogg", endpoint=["nope"])
    assert result.is_error is True
    assert "invalid" in result.content.lower()


# --- path handling -----------------------------------------------------------


@pytest.mark.asyncio
async def test_relative_path_resolves_against_workspace(tmp_path):
    audio_file(tmp_path, data=b"REL")
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    result = await run_stt(tmp_path, "media/abc123/voice.ogg", client=client)
    assert result.is_error is False
    assert posted(client)["files"]["file"][1] == b"REL"


@pytest.mark.asyncio
async def test_absolute_path_accepted(tmp_path):
    path = audio_file(tmp_path, data=b"ABS")
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    result = await run_stt(tmp_path, str(path), client=client)
    assert result.is_error is False
    assert posted(client)["files"]["file"][1] == b"ABS"


@pytest.mark.asyncio
async def test_media_tag_string_is_parsed(tmp_path):
    """Models copy the `[media: ...]` tag verbatim; the tool must accept it."""
    audio_file(tmp_path, data=b"TAGGED")
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    tag = "[media: media/abc123/voice.ogg (audio/ogg, 45KB)]"
    result = await run_stt(tmp_path, tag, client=client)
    assert result.is_error is False
    assert posted(client)["files"]["file"][1] == b"TAGGED"


@pytest.mark.asyncio
async def test_media_tag_inside_sentence_is_parsed(tmp_path):
    audio_file(tmp_path, data=b"EMBEDDED")
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    result = await run_stt(
        tmp_path, "please transcribe [media: media/abc123/voice.ogg (audio/ogg, 45KB)]",
        client=client,
    )
    assert result.is_error is False
    assert posted(client)["files"]["file"][1] == b"EMBEDDED"


@pytest.mark.asyncio
async def test_missing_file_reports_not_found(tmp_path):
    result = await run_stt(tmp_path, "media/nope/voice.ogg")
    assert result.is_error is True
    assert "not found" in result.content.lower()
    assert "media/nope/voice.ogg" in result.content


@pytest.mark.asyncio
async def test_directory_path_rejected(tmp_path):
    (tmp_path / "media" / "dir").mkdir(parents=True)
    result = await run_stt(tmp_path, "media/dir")
    assert result.is_error is True
    assert "not a file" in result.content.lower()


@pytest.mark.asyncio
async def test_empty_file_rejected(tmp_path):
    audio_file(tmp_path, data=b"")
    result = await run_stt(tmp_path, "media/abc123/voice.ogg")
    assert result.is_error is True
    assert "empty" in result.content.lower()


@pytest.mark.asyncio
async def test_oversize_file_rejected(tmp_path):
    audio_file(tmp_path, data=b"x" * 100)
    result = await run_stt(tmp_path, "media/abc123/voice.ogg", max_bytes=10)
    assert result.is_error is True
    assert "too large" in result.content.lower()
    assert "10" in result.content


# --- request shape -----------------------------------------------------------


@pytest.mark.asyncio
async def test_multipart_file_field_and_mime(tmp_path):
    audio_file(tmp_path, data=b"OGGDATA")
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    await run_stt(tmp_path, "media/abc123/voice.ogg", client=client)
    call = posted(client)
    name, payload, mime = call["files"]["file"]
    assert name == "voice.ogg"
    assert payload == b"OGGDATA"
    assert mime == "audio/ogg"
    assert client.stream.call_args_list[0].args[1] == ENDPOINT


@pytest.mark.asyncio
async def test_language_omitted_by_default(tmp_path):
    audio_file(tmp_path)
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    await run_stt(tmp_path, "media/abc123/voice.ogg", client=client)
    assert "language" not in posted(client)["data"]


@pytest.mark.asyncio
async def test_language_from_config(tmp_path):
    audio_file(tmp_path)
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    await run_stt(tmp_path, "media/abc123/voice.ogg", client=client, language="es")
    assert posted(client)["data"]["language"] == "es"


@pytest.mark.asyncio
async def test_language_param_overrides_config(tmp_path):
    audio_file(tmp_path)
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    patcher, _ = patch_httpx(client)
    with patcher:
        await stt(
            path="media/abc123/voice.ogg", language="fr",
            tool_config=cfg(endpoint=ENDPOINT, language="es"),
            agent_config=make_agent_config(tmp_path),
        )
    assert posted(client)["data"]["language"] == "fr"


@pytest.mark.asyncio
async def test_prompt_omitted_by_default(tmp_path):
    audio_file(tmp_path)
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    await run_stt(tmp_path, "media/abc123/voice.ogg", client=client)
    assert "prompt" not in posted(client)["data"]


@pytest.mark.asyncio
async def test_prompt_from_config_is_sent(tmp_path):
    audio_file(tmp_path)
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    await run_stt(
        tmp_path, "media/abc123/voice.ogg", client=client,
        prompt="Merry and SB work on OpenAlph.",
    )
    assert posted(client)["data"]["prompt"] == "Merry and SB work on OpenAlph."


@pytest.mark.asyncio
async def test_prompt_param_overrides_config(tmp_path):
    audio_file(tmp_path)
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    patcher, _ = patch_httpx(client)
    with patcher:
        await stt(
            path="media/abc123/voice.ogg", prompt="Call me Merry.",
            tool_config=cfg(endpoint=ENDPOINT, prompt="from-config"),
            agent_config=make_agent_config(tmp_path),
        )
    assert posted(client)["data"]["prompt"] == "Call me Merry."


@pytest.mark.asyncio
async def test_prompt_off_sentinel_passes_through(tmp_path):
    """`off` is the SERVICE's opt-out sentinel — the tool must not swallow it."""
    audio_file(tmp_path)
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    patcher, _ = patch_httpx(client)
    with patcher:
        await stt(
            path="media/abc123/voice.ogg", prompt="off",
            tool_config=cfg(endpoint=ENDPOINT, prompt="from-config"),
            agent_config=make_agent_config(tmp_path),
        )
    assert posted(client)["data"]["prompt"] == "off"


@pytest.mark.asyncio
async def test_response_format_json_by_default(tmp_path):
    audio_file(tmp_path)
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    await run_stt(tmp_path, "media/abc123/voice.ogg", client=client)
    assert posted(client)["data"]["response_format"] == "json"


@pytest.mark.asyncio
async def test_timestamps_requests_verbose_json(tmp_path):
    audio_file(tmp_path)
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    patcher, _ = patch_httpx(client)
    with patcher:
        await stt(
            path="media/abc123/voice.ogg", timestamps=True,
            tool_config=cfg(endpoint=ENDPOINT),
            agent_config=make_agent_config(tmp_path),
        )
    assert posted(client)["data"]["response_format"] == "verbose_json"


@pytest.mark.asyncio
async def test_timeout_from_config_used(tmp_path):
    audio_file(tmp_path)
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    patcher, mock_cls = patch_httpx(client)
    with patcher:
        await stt(
            path="media/abc123/voice.ogg", tool_config=cfg(endpoint=ENDPOINT, timeout=77.0),
            agent_config=make_agent_config(tmp_path),
        )
    assert mock_cls.call_args.kwargs["timeout"] == 77.0


# --- response handling -------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_returns_transcript(tmp_path):
    audio_file(tmp_path)
    client = make_client(responses=[make_response(json_data={"text": "Hi Merry, all good."})])
    result = await run_stt(tmp_path, "media/abc123/voice.ogg", client=client)
    assert result.is_error is False
    assert result.content == "Hi Merry, all good."


@pytest.mark.asyncio
async def test_audio_bytes_never_in_result(tmp_path):
    audio_file(tmp_path, data=b"RAWAUDIOBYTES")
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    result = await run_stt(tmp_path, "media/abc123/voice.ogg", client=client)
    assert "RAWAUDIOBYTES" not in result.content


@pytest.mark.asyncio
async def test_timestamps_formatting(tmp_path):
    audio_file(tmp_path)
    payload = {
        "text": "The transcribed text content.",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "The transcribed"},
            {"start": 2.5, "end": 4.0, "text": " text content."},
        ],
    }
    client = make_client(responses=[make_response(json_data=payload)])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await stt(
            path="media/abc123/voice.ogg", timestamps=True,
            tool_config=cfg(endpoint=ENDPOINT),
            agent_config=make_agent_config(tmp_path),
        )
    assert result.is_error is False
    assert "The transcribed text content." in result.content
    assert "en" in result.content
    assert "0.0" in result.content and "2.5" in result.content
    assert "[segments]" in result.content


@pytest.mark.asyncio
async def test_timestamps_segments_capped(tmp_path):
    audio_file(tmp_path)
    payload = {
        "text": "long",
        "language": "en",
        "segments": [
            {"start": float(i), "end": float(i) + 1.0, "text": f"seg{i}"} for i in range(400)
        ],
    }
    client = make_client(responses=[make_response(json_data=payload)])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await stt(
            path="media/abc123/voice.ogg", timestamps=True,
            tool_config=cfg(endpoint=ENDPOINT),
            agent_config=make_agent_config(tmp_path),
        )
    assert result.is_error is False
    assert "seg0" in result.content
    assert "seg399" not in result.content
    assert "truncated" in result.content.lower()


@pytest.mark.asyncio
async def test_empty_transcript_is_not_an_error(tmp_path):
    audio_file(tmp_path)
    client = make_client(responses=[make_response(json_data={"text": ""})])
    result = await run_stt(tmp_path, "media/abc123/voice.ogg", client=client)
    assert result.is_error is False
    assert "empty transcript" in result.content.lower()


@pytest.mark.asyncio
async def test_malformed_json_response_fails_soft(tmp_path):
    audio_file(tmp_path)
    client = make_client(responses=[make_response(content=b"<html>nope</html>")])
    result = await run_stt(tmp_path, "media/abc123/voice.ogg", client=client)
    assert result.is_error is True
    assert "malformed response" in result.content
    assert '{"text"' in result.content


@pytest.mark.asyncio
async def test_json_without_text_key_fails_soft(tmp_path):
    audio_file(tmp_path)
    client = make_client(responses=[make_response(json_data={"error": "boom"})])
    result = await run_stt(tmp_path, "media/abc123/voice.ogg", client=client)
    assert result.is_error is True
    assert "malformed response" in result.content


@pytest.mark.asyncio
async def test_non_string_text_fails_soft(tmp_path):
    audio_file(tmp_path)
    client = make_client(responses=[make_response(json_data={"text": ["not", "a", "str"]})])
    result = await run_stt(tmp_path, "media/abc123/voice.ogg", client=client)
    assert result.is_error is True
    assert "malformed response" in result.content


@pytest.mark.asyncio
async def test_http_400_fails_soft_with_body(tmp_path):
    audio_file(tmp_path)
    client = make_client(responses=[make_response(400, content=b"unsupported format")])
    result = await run_stt(tmp_path, "media/abc123/voice.ogg", client=client)
    assert result.is_error is True
    assert "service error HTTP 400" in result.content
    assert "unsupported format" in result.content


@pytest.mark.asyncio
async def test_timeout_fails_soft_with_endpoint(tmp_path):
    audio_file(tmp_path)
    client = make_client(side_effect=httpx.TimeoutException("too slow"))
    result = await run_stt(tmp_path, "media/abc123/voice.ogg", client=client)
    assert result.is_error is True
    assert "timed out" in result.content
    assert ENDPOINT in result.content


@pytest.mark.asyncio
async def test_connect_error_fails_soft(tmp_path):
    audio_file(tmp_path)
    client = make_client(side_effect=httpx.ConnectError("refused"))
    result = await run_stt(tmp_path, "media/abc123/voice.ogg", client=client)
    assert result.is_error is True
    assert "cannot reach" in result.content


@pytest.mark.parametrize(
    "kwargs",
    [
        {"path": 123},
        {"path": "media/abc123/voice.ogg", "language": 5},
        {"path": "media/abc123/voice.ogg", "prompt": 5},
        {"path": "media/abc123/voice.ogg", "timestamps": "yes"},
    ],
)
@pytest.mark.asyncio
async def test_bad_param_types_rejected_not_raised(tmp_path, kwargs):
    audio_file(tmp_path)
    result = await run_stt(tmp_path, **kwargs)
    assert result.is_error is True
    assert "invalid" in result.content.lower()


# --- dispatch integration ----------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_dispatches_stt_and_joins_workspace_path(tmp_path):
    """The dispatch chain must workspace-join stt's `path` (path-join tuple)."""
    audio_file(tmp_path, data=b"VIADISPATCH")
    client = make_client(responses=[make_response(json_data={"text": "dispatched"})])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await execute_tool(
            "stt",
            {"path": "media/abc123/voice.ogg"},
            cfg(endpoint=ENDPOINT),
            make_agent_config(tmp_path),
        )
    assert result.is_error is False
    assert result.content == "dispatched"
    assert posted(client)["files"]["file"][1] == b"VIADISPATCH"


@pytest.mark.asyncio
async def test_execute_tool_rejects_missing_path(tmp_path):
    result = await execute_tool("stt", {}, cfg(endpoint=ENDPOINT), make_agent_config(tmp_path))
    assert result.is_error is True
    assert "path" in result.content
