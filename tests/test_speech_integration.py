"""Speech tools — REAL-PATH integration tests (im7t.68).

House rule (tool-management § "The one lesson"): a green unit suite proves nothing about
the wiring. These tests build a REAL Agent and drive the REAL
`MatrixBot._build_agent_callbacks` seam, mocking only the provider (never called) and the
nio transport. They pin that:

  - the `send_to_room` path reaches the real Matrix upload callback,
  - the CLI/no-callback path reports the file instead of failing,
  - the standard redaction tail covers speech tool output,
  - the new tools actually reach the model via `tool_schemas`.

Spec: memory/projects/openalph/specs/im7t.68-tts-stt-native-tools-spec.md
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from openalph.agent import Agent
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig
from openalph.matrix import MatrixBot
from openalph.tools import execute_tool, tool_schemas

ROOM_A = "!roomA:matrix.local"
AGENT_USER = "@agent:matrix.local"
TTS_ENDPOINT = "http://10.0.20.104:8006/v1/audio/speech"
STT_ENDPOINT = "http://10.0.20.104:8001/v1/audio/transcriptions"


# --- real-agent harness (mirrors tests/test_guidance_integration.py) ---------


def _cfg(workspace, **kw):
    defaults = dict(
        name="test-speech-integ",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
            base_url=None, quirks=[],
        )},
        workspace=workspace,
        max_iterations=100,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
        reminders=True,
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _setup_workspace(tmp_path, tts_config="", stt_config=""):
    """Real workspace/tools/ TOMLs — discovery runs for real."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(exist_ok=True)
    for name in ("shell", "file_read", "file_write", "send_media"):
        (tools_dir / f"{name}.toml").write_text("[config]\n")
    (tools_dir / "tts.toml").write_text(tts_config)
    (tools_dir / "stt.toml").write_text(stt_config)
    return tmp_path


def _make_bot_with_real_agent(tmp_path, tts_config="", stt_config=""):
    ws = _setup_workspace(tmp_path, tts_config=tts_config, stt_config=stt_config)
    agent = Agent(_cfg(ws))

    matrix_config = MatrixConfig(
        homeserver="https://matrix.local",
        user_id=AGENT_USER,
        device_id="TEST",
        password="test-password",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
    )

    bot = MatrixBot.__new__(MatrixBot)
    bot.config = matrix_config
    bot.agent = agent
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock(return_value=MagicMock(event_id="$resp1"))
    bot.client.room_typing = AsyncMock()
    bot.client.upload = AsyncMock(
        return_value=(MagicMock(content_uri="mxc://matrix.local/abc123"), None)
    )
    bot._current_room = None
    bot._synced = True
    bot._active_rooms = set()
    bot._room_effort = {}
    bot._room_cache_ttl = {}
    bot._room_timesense = {}
    bot._halted_rooms = set()
    bot._background_tasks = set()
    bot._session_locks = {}
    bot.session_log = MagicMock()
    bot.session_log.append = MagicMock()
    bot.session_log.build_context = MagicMock(return_value=[])
    bot.session_log.read = MagicMock(return_value=[])
    bot.session_log.last_event_id = MagicMock(return_value=None)
    bot.session_log.usage_totals = MagicMock(return_value={})
    bot.heartbeat = MagicMock()
    bot.heartbeat.is_active = MagicMock(return_value=False)
    bot.umbral = MagicMock()
    bot.umbral.is_active = MagicMock(return_value=False)
    bot._steering_inbox = {}
    bot._active_turns = set()
    return bot, agent


def _tool_config(agent, name):
    """Config as discovered from the workspace TOML (the real source of truth)."""
    return next(t.config for t in agent.tools if t.name == name)


def stream_cm(resp, chunk_size=0):
    """Async context manager over a mocked response exposing aiter_bytes()."""
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


def make_client(responses=None, side_effect=None):
    client = AsyncMock()
    if side_effect is not None:
        client.stream = MagicMock(side_effect=side_effect)
    elif responses is not None:
        client.stream = MagicMock(side_effect=[stream_cm(r) for r in responses])
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def make_response(status_code=200, content=b"AUDIO", json_data=None, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.text = content.decode("utf-8", "replace")
    resp.json = MagicMock(return_value=json_data)
    resp.headers = headers if headers is not None else {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )
    return resp


def patch_httpx(client):
    mock_cls = MagicMock(return_value=client)
    return patch("openalph.tools.speech.httpx.AsyncClient", mock_cls), mock_cls


# --- the real callback seam --------------------------------------------------


def test_real_callbacks_expose_send_media(tmp_path):
    bot, agent = _make_bot_with_real_agent(tmp_path)
    cb = bot._build_agent_callbacks(ROOM_A, None)
    assert "send_media" in cb
    assert callable(cb["send_media"])
    assert cb["room_id"] == ROOM_A


@pytest.mark.asyncio
async def test_tts_send_to_room_reaches_real_matrix_upload(tmp_path):
    """tts(send_to_room=True) → real send_media executor → real upload callback."""
    bot, agent = _make_bot_with_real_agent(
        tmp_path,
        tts_config=f'[config]\nendpoint = "{TTS_ENDPOINT}"\nvoice = "af_heart"\n',
    )
    cb = bot._build_agent_callbacks(ROOM_A, None)
    client = make_client(responses=[make_response(content=b"MP3-FROM-STUDIO")])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await execute_tool(
            "tts",
            {"text": "Hello from the real seam.", "send_to_room": True, "caption": "Voice reply"},
            _tool_config(agent, "tts"),
            agent.config,
            tools=agent.tools,
            callbacks=cb,
        )
    assert result.is_error is False, result.content
    bot.client.upload.assert_awaited_once()
    upload_kwargs = bot.client.upload.await_args.kwargs
    assert upload_kwargs["content_type"] == "audio/mpeg"
    assert upload_kwargs["filename"].endswith(".mp3")
    sent = bot.client.room_send.await_args.args[2]
    assert sent["msgtype"] == "m.audio"
    assert sent["body"] == "Voice reply"


@pytest.mark.asyncio
async def test_tts_without_send_to_room_never_uploads(tmp_path):
    bot, agent = _make_bot_with_real_agent(
        tmp_path,
        tts_config=f'[config]\nendpoint = "{TTS_ENDPOINT}"\n',
    )
    cb = bot._build_agent_callbacks(ROOM_A, None)
    client = make_client(responses=[make_response()])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await execute_tool(
            "tts", {"text": "Just a file, please."},
            _tool_config(agent, "tts"), agent.config,
            tools=agent.tools, callbacks=cb,
        )
    assert result.is_error is False
    bot.client.upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_cli_mode_without_callbacks_reports_the_file(tmp_path):
    """Headless/CLI: no callbacks dict at all — generation must still succeed."""
    bot, agent = _make_bot_with_real_agent(
        tmp_path,
        tts_config=f'[config]\nendpoint = "{TTS_ENDPOINT}"\n',
    )
    client = make_client(responses=[make_response()])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await execute_tool(
            "tts", {"text": "Headless speech.", "send_to_room": True},
            _tool_config(agent, "tts"), agent.config,
            tools=agent.tools, callbacks=None,
        )
    assert result.is_error is False, result.content
    assert "media/tts/" in result.content
    # An explicit send request that could not be honoured must say so.
    assert "NOT sent" in result.content


@pytest.mark.asyncio
async def test_stt_through_real_dispatch_and_workspace_join(tmp_path):
    bot, agent = _make_bot_with_real_agent(
        tmp_path,
        stt_config=f'[config]\nendpoint = "{STT_ENDPOINT}"\nprompt = "Merry and SB."\n',
    )
    cb = bot._build_agent_callbacks(ROOM_A, None)
    media = tmp_path / "media" / "abc123" / "voice.ogg"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"OGG-NOTE")
    client = make_client(responses=[make_response(json_data={"text": "Hi Merry."})])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await execute_tool(
            "stt",
            {"path": "[media: media/abc123/voice.ogg (audio/ogg, 45KB)]"},
            _tool_config(agent, "stt"), agent.config,
            tools=agent.tools, callbacks=cb,
        )
    assert result.is_error is False, result.content
    assert result.content == "Hi Merry."
    sent = client.stream.call_args_list[0].kwargs
    assert sent["data"]["prompt"] == "Merry and SB."
    assert sent["files"]["file"][1] == b"OGG-NOTE"


@pytest.mark.asyncio
async def test_redaction_tail_covers_speech_output(tmp_path):
    """A shaped secret echoed by the service must be redacted by the shared tail."""
    bot, agent = _make_bot_with_real_agent(
        tmp_path,
        stt_config=f'[config]\nendpoint = "{STT_ENDPOINT}"\n',
    )
    media = tmp_path / "media" / "abc123" / "voice.ogg"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"OGG-NOTE")
    secret = "sk-ant-api03-ZZZZZZZZZZZZZZZZZZZZZZ"
    client = make_client(responses=[make_response(json_data={"text": f"the key is {secret}"})])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await execute_tool(
            "stt", {"path": "media/abc123/voice.ogg"},
            _tool_config(agent, "stt"), agent.config,
            tools=agent.tools, callbacks=bot._build_agent_callbacks(ROOM_A, None),
        )
    assert secret not in result.content
    assert "[REDACTED" in result.content


def test_tool_schemas_expose_speech_tools(tmp_path):
    bot, agent = _make_bot_with_real_agent(tmp_path)
    schemas = {s["name"]: s for s in tool_schemas(agent.tools)}
    assert "tts" in schemas and "stt" in schemas
    assert schemas["tts"]["input_schema"]["required"] == ["text"]
    assert schemas["stt"]["input_schema"]["required"] == ["path"]
    assert schemas["tts"]["description"].strip()
    assert schemas["stt"]["description"].strip()


def test_speech_tools_are_opt_in_per_agent(tmp_path):
    """No tts.toml/stt.toml → tools absent from the agent's tool list."""
    ws = tmp_path
    (ws / "tools").mkdir()
    (ws / "tools" / "shell.toml").write_text("[config]\n")
    agent = Agent(_cfg(ws))
    names = {t.name for t in agent.tools}
    assert "tts" not in names
    assert "stt" not in names
