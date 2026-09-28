"""TTS tool tests (im7t.68).

Contract (spec: memory/projects/openalph/specs/im7t.68-tts-stt-native-tools-spec.md):

  tts(text, voice=None, speed=None, normalize_text=None, send_to_room=False,
      caption=None, tool_config=None, agent_config=None, callbacks=None) -> ToolResult

  - endpoint/model/voice/speed/chunking/limits come from workspace/tools/tts.toml [config];
    endpoint defaults to "" and an enabled-but-unconfigured tool FAILS LOUDLY with steering
    naming the TOML path.
  - text is normalized (deterministic technical-string normalizer) then chunked at sentence
    boundaries; >1 chunk is concatenated with ffmpeg (stream copy, no re-encode).
  - output: <output_dir>/<UTC ts>-<sha256(normalized text)[:8]>.mp3 — audio bytes NEVER
    appear in the tool result.
  - send_to_room delegates to media.send_media via callbacks["send_media"].

Mocking boundary: httpx (HTTP) only. ffmpeg is a REAL subprocess against a fake binary on
disk, so argv shape, stream-copy, ordering and temp cleanup are all genuinely exercised.
"""

import base64
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from openalph.config import AgentConfig, ProviderConfig
from openalph.tools import BUILTIN_TOOLS, discover_tools, execute_tool
from openalph.tools.speech import normalize, tts

ENDPOINT = "http://10.0.20.104:8006/v1/audio/speech"
KOKORO_MODEL = "mlx-community/Kokoro-82M-bf16"


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
    """Tool config from the REGISTRY defaults (never hand-built), with overrides."""
    base = dict(BUILTIN_TOOLS["tts"]["config"])
    base.update(overrides)
    return base


def make_response(status_code=200, content=b"AUDIO", json_data=None, text=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.text = text if text is not None else (
        content.decode("utf-8", "replace") if isinstance(content, bytes) else ""
    )
    resp.json = MagicMock(return_value=json_data)
    resp.headers = {"content-type": "audio/mpeg"}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )
    return resp


def stream_cm(resp, chunk_size=0):
    """Async context manager over a mocked response exposing aiter_bytes().

    The tool reads response bodies with a bounded stream (so a hostile or broken
    endpoint cannot OOM the process), so the mock must model the streaming API.
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


def make_client(responses=None, side_effect=None, repeat=None, chunk_size=0):
    """Patched httpx.AsyncClient instance.

    responses: consumed in order (one per request).
    repeat:    the same response returned for every request (unknown chunk count).
    """
    client = AsyncMock()
    if side_effect is not None:
        client.stream = MagicMock(side_effect=side_effect)
    elif repeat is not None:
        client.stream = MagicMock(side_effect=lambda *a, **k: stream_cm(repeat, chunk_size))
    elif responses is not None:
        client.stream = MagicMock(side_effect=[stream_cm(r, chunk_size) for r in responses])
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def patch_httpx(client):
    """Patch the module's httpx.AsyncClient; returns the mock class."""
    mock_cls = MagicMock(return_value=client)
    return patch("openalph.tools.speech.httpx.AsyncClient", mock_cls), mock_cls


FAKE_FFMPEG_SRC = '''#!/usr/bin/env python3
"""Fake ffmpeg for tests: honours the concat-demuxer contract.

Records its argv to {log}, reads the `-i <listfile>` concat list, and writes the
byte-concatenation of the listed files (in order) to the final argv argument.
"""
import pathlib
import sys

argv = sys.argv[1:]
pathlib.Path({log!r}).write_text("\\n".join(argv))

if {fail!r}:
    sys.stderr.write("fake-ffmpeg: simulated failure\\n")
    sys.exit({exit_code})

listfile = argv[argv.index("-i") + 1]
out = pathlib.Path(argv[-1])
data = b""
for line in pathlib.Path(listfile).read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    if line.startswith("file "):
        line = line[5:].strip()
    if line[:1] in ("'", '"') and line[-1:] == line[:1]:
        line = line[1:-1]
    data += pathlib.Path(line).read_bytes()
out.write_bytes(data)
'''


def make_fake_ffmpeg(tmp_path, *, fail=False, exit_code=1):
    """Real executable fake ffmpeg; returns (path, argv_log_path)."""
    log = tmp_path / "ffmpeg-argv.txt"
    path = tmp_path / "fake-ffmpeg"
    path.write_text(FAKE_FFMPEG_SRC.format(log=str(log), fail=fail, exit_code=exit_code))
    path.chmod(0o755)
    return path, log


async def run_tts(tmp_path, text="Hello there.", client=None, fake_ffmpeg=None,
                  callback=None, **overrides):
    """Drive tts() with a real workspace, patched HTTP, real fake ffmpeg."""
    config = cfg(endpoint=ENDPOINT, model=KOKORO_MODEL, voice="af_heart")
    config.update(overrides)
    if fake_ffmpeg is not None:
        config["ffmpeg_path"] = str(fake_ffmpeg)
    agent_config = make_agent_config(tmp_path)
    callbacks = {"send_media": callback} if callback is not None else None
    if client is None:
        client = make_client(responses=[make_response()])
    patcher, _ = patch_httpx(client)
    with patcher:
        return await tts(
            text=text,
            tool_config=config,
            agent_config=agent_config,
            callbacks=callbacks,
        )


# --- registry defaults (the config contract) ---------------------------------


def test_registry_defaults_match_spec():
    assert BUILTIN_TOOLS["tts"]["config"] == {
        "endpoint": "",
        "model": "",
        "voice": "",
        "speed": 1.0,
        "normalize": True,
        "chunk_tokens": 200,
        "ffmpeg_path": "ffmpeg",
        "timeout": 120.0,
        "max_input_chars": 20000,
        "max_audio_bytes": 20971520,
        "output_dir": "media/tts",
    }


def test_registry_schema_requires_text():
    schema = BUILTIN_TOOLS["tts"]["parameters"]
    assert schema["required"] == ["text"]
    assert set(schema["properties"]) == {
        "text", "voice", "speed", "normalize_text", "send_to_room", "caption",
    }


def test_description_carries_steering_and_when_not_to_use():
    desc = BUILTIN_TOOLS["tts"]["description"]
    assert "endpoint" in desc  # config guidance
    assert "tts.toml" in desc  # where to configure it
    assert "NEVER" in desc or "IMPORTANT" in desc
    # House phrasing for when-NOT guidance is "NOT for X — use Y" (see web_search);
    # "When NOT to use" is the equivalent long form.
    assert "NOT for" in desc or "When NOT" in desc or "not to use" in desc.lower()


def test_empty_toml_enables_with_defaults(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "tts.toml").write_text("")
    tools = {t.name: t for t in discover_tools(tmp_path)}
    assert "tts" in tools
    assert tools["tts"].config["endpoint"] == ""


def test_toml_config_overlays_registry_defaults(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "tts.toml").write_text(
        f'[config]\nendpoint = "{ENDPOINT}"\nvoice = "af_bella"\nchunk_tokens = 0\n'
    )
    tools = {t.name: t for t in discover_tools(tmp_path)}
    assert tools["tts"].config["endpoint"] == ENDPOINT
    assert tools["tts"].config["voice"] == "af_bella"
    assert tools["tts"].config["chunk_tokens"] == 0
    assert tools["tts"].config["model"] == ""  # untouched default


# --- configuration failures --------------------------------------------------


@pytest.mark.asyncio
async def test_unconfigured_endpoint_fails_loudly(tmp_path):
    result = await run_tts(tmp_path, endpoint="")
    assert result.is_error is True
    assert "no endpoint configured" in result.content
    assert "workspace/tools/tts.toml" in result.content


@pytest.mark.asyncio
async def test_missing_endpoint_key_fails_loudly(tmp_path):
    """A config dict without the key at all must not crash the turn."""
    config = cfg()
    del config["endpoint"]
    result = await tts(
        text="hi", tool_config=config, agent_config=make_agent_config(tmp_path),
    )
    assert result.is_error is True
    assert "no endpoint configured" in result.content


@pytest.mark.asyncio
async def test_non_string_endpoint_fails_loudly(tmp_path):
    result = await run_tts(tmp_path, endpoint=12345)
    assert result.is_error is True
    assert "invalid" in result.content.lower()


# --- input validation --------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_text_rejected(tmp_path):
    result = await run_tts(tmp_path, text="")
    assert result.is_error is True
    assert "text is empty" in result.content


@pytest.mark.asyncio
async def test_whitespace_only_text_rejected(tmp_path):
    result = await run_tts(tmp_path, text="   \n  ")
    assert result.is_error is True
    assert "text is empty" in result.content


@pytest.mark.asyncio
async def test_text_over_cap_rejected_with_steering(tmp_path):
    result = await run_tts(tmp_path, text="x" * 50, max_input_chars=10)
    assert result.is_error is True
    assert "text too long" in result.content
    assert "max_input_chars" in result.content


@pytest.mark.parametrize(
    "kwargs",
    [
        {"text": 123},
        {"text": "ok", "voice": 7},
        {"text": "ok", "speed": "fast"},
        {"text": "ok", "send_to_room": "yes"},
        {"text": "ok", "normalize_text": "nope"},
        {"text": "ok", "caption": 42},
    ],
)
@pytest.mark.asyncio
async def test_bad_param_types_are_rejected_not_raised(tmp_path, kwargs):
    result = await run_tts(tmp_path, **kwargs)
    assert result.is_error is True
    assert "invalid" in result.content.lower()


@pytest.mark.asyncio
async def test_bad_chunk_tokens_type_rejected(tmp_path):
    result = await run_tts(tmp_path, chunk_tokens="lots")
    assert result.is_error is True
    assert "invalid" in result.content.lower()


@pytest.mark.asyncio
async def test_bad_timeout_type_rejected(tmp_path):
    result = await run_tts(tmp_path, timeout="soon")
    assert result.is_error is True
    assert "invalid" in result.content.lower()


# --- happy path --------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_chunk_writes_audio_and_returns_path(tmp_path):
    client = make_client(responses=[make_response(content=b"MP3BYTES")])
    result = await run_tts(tmp_path, text="Hello there.", client=client)
    assert result.is_error is False
    assert "media/tts/" in result.content
    assert ".mp3" in result.content
    written = list((tmp_path / "media" / "tts").glob("*.mp3"))
    assert len(written) == 1
    assert written[0].read_bytes() == b"MP3BYTES"


@pytest.mark.asyncio
async def test_audio_bytes_never_in_result(tmp_path):
    payload = b"\xff\xfb\x90\x00SECRET-AUDIO-PAYLOAD"
    client = make_client(responses=[make_response(content=payload)])
    result = await run_tts(tmp_path, client=client)
    assert payload not in result.content.encode("utf-8", "replace")
    assert base64.b64encode(payload).decode() not in result.content
    assert "SECRET-AUDIO-PAYLOAD" not in result.content


@pytest.mark.asyncio
async def test_output_dir_created_when_missing(tmp_path):
    assert not (tmp_path / "media").exists()
    client = make_client(responses=[make_response()])
    result = await run_tts(tmp_path, client=client)
    assert result.is_error is False
    assert (tmp_path / "media" / "tts").is_dir()


@pytest.mark.asyncio
async def test_output_dir_config_override(tmp_path):
    client = make_client(responses=[make_response()])
    result = await run_tts(tmp_path, client=client, output_dir="scratch/voice")
    assert result.is_error is False
    assert (tmp_path / "scratch" / "voice").is_dir()
    assert "scratch/voice/" in result.content


@pytest.mark.asyncio
async def test_output_filename_shape(tmp_path):
    client = make_client(responses=[make_response()])
    result = await run_tts(tmp_path, client=client)
    assert result.is_error is False
    names = [p.name for p in (tmp_path / "media" / "tts").glob("*.mp3")]
    assert len(names) == 1
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}\.mp3", names[0]), names[0]


@pytest.mark.asyncio
async def test_repeated_identical_text_does_not_clobber(tmp_path):
    """Two calls in the same second must produce two distinct files."""
    client = make_client(responses=[make_response(content=b"ONE"), make_response(content=b"TWO")])
    first = await run_tts(tmp_path, text="Same words.", client=client)
    second = await run_tts(tmp_path, text="Same words.", client=client)
    assert first.is_error is False and second.is_error is False
    files = list((tmp_path / "media" / "tts").glob("*.mp3"))
    assert len(files) == 2
    assert {f.read_bytes() for f in files} == {b"ONE", b"TWO"}


# --- request body ------------------------------------------------------------


def posted_bodies(client):
    return [call.kwargs["json"] for call in client.stream.call_args_list]


@pytest.mark.asyncio
async def test_request_body_shape(tmp_path):
    client = make_client(responses=[make_response()])
    await run_tts(tmp_path, text="Hello there.", client=client)
    body = posted_bodies(client)[0]
    assert set(body) == {"model", "input", "voice", "speed"}
    assert body["model"] == KOKORO_MODEL
    assert body["voice"] == "af_heart"
    assert body["speed"] == 1.0
    assert body["input"] == "Hello there."


@pytest.mark.asyncio
async def test_model_omitted_when_unconfigured(tmp_path):
    client = make_client(responses=[make_response()])
    await run_tts(tmp_path, client=client, model="")
    body = posted_bodies(client)[0]
    assert "model" not in body


@pytest.mark.asyncio
async def test_voice_omitted_when_unconfigured_and_no_param(tmp_path):
    client = make_client(responses=[make_response()])
    await run_tts(tmp_path, client=client, voice="")
    body = posted_bodies(client)[0]
    assert "voice" not in body


@pytest.mark.asyncio
async def test_param_overrides_config_voice_and_speed(tmp_path):
    client = make_client(responses=[make_response()])
    config = cfg(endpoint=ENDPOINT, model=KOKORO_MODEL, voice="af_heart")
    patcher, _ = patch_httpx(client)
    with patcher:
        await tts(
            text="hi", voice="bm_george", speed=1.25,
            tool_config=config, agent_config=make_agent_config(tmp_path),
        )
    body = posted_bodies(client)[0]
    assert body["voice"] == "bm_george"
    assert body["speed"] == 1.25


@pytest.mark.asyncio
async def test_post_targets_configured_endpoint(tmp_path):
    client = make_client(responses=[make_response()])
    await run_tts(tmp_path, client=client)
    assert client.stream.call_args_list[0].args[1] == ENDPOINT


@pytest.mark.asyncio
async def test_timeout_from_config_is_used(tmp_path):
    client = make_client(responses=[make_response()])
    patcher, mock_cls = patch_httpx(client)
    config = cfg(endpoint=ENDPOINT, timeout=42.0)
    with patcher:
        await tts(
            text="hi", tool_config=config, agent_config=make_agent_config(tmp_path),
        )
    assert mock_cls.call_args.kwargs["timeout"] == 42.0


# --- normalization -----------------------------------------------------------


@pytest.mark.asyncio
async def test_normalization_on_by_default(tmp_path):
    client = make_client(responses=[make_response()])
    raw = "Version 0.19.0 on host 10.0.20.104"
    await run_tts(tmp_path, text=raw, client=client)
    assert posted_bodies(client)[0]["input"] == normalize(raw)
    assert "zero point nineteen" in posted_bodies(client)[0]["input"]


@pytest.mark.asyncio
async def test_normalize_false_param_preserves_raw_text(tmp_path):
    client = make_client(responses=[make_response()])
    raw = "Version 0.19.0"
    await run_tts(tmp_path, text=raw, client=client, normalize_text=False)
    assert posted_bodies(client)[0]["input"] == raw


@pytest.mark.asyncio
async def test_normalize_config_false_respected_when_param_absent(tmp_path):
    client = make_client(responses=[make_response()])
    raw = "Version 0.19.0"
    await run_tts(tmp_path, text=raw, client=client, normalize=False)
    assert posted_bodies(client)[0]["input"] == raw


@pytest.mark.asyncio
async def test_param_wins_over_config_for_normalize(tmp_path):
    client = make_client(responses=[make_response()])
    raw = "Version 0.19.0"
    await run_tts(tmp_path, text=raw, client=client, normalize=False, normalize_text=True)
    assert posted_bodies(client)[0]["input"] == normalize(raw)


# --- chunking + ffmpeg -------------------------------------------------------


@pytest.mark.asyncio
async def test_single_chunk_never_invokes_ffmpeg(tmp_path):
    fake, log = make_fake_ffmpeg(tmp_path)
    client = make_client(responses=[make_response()])
    result = await run_tts(tmp_path, text="Short sentence.", client=client, fake_ffmpeg=fake)
    assert result.is_error is False
    assert not log.exists(), "ffmpeg must not run for a single-chunk utterance"


@pytest.mark.asyncio
async def test_chunk_tokens_zero_disables_chunking(tmp_path):
    fake, log = make_fake_ffmpeg(tmp_path)
    client = make_client(responses=[make_response()])
    result = await run_tts(
        tmp_path, text="One. Two. Three. Four.", client=client,
        fake_ffmpeg=fake, chunk_tokens=0,
    )
    assert result.is_error is False
    assert client.stream.call_count == 1
    assert not log.exists()


@pytest.mark.asyncio
async def test_multi_chunk_concatenates_in_order(tmp_path):
    fake, log = make_fake_ffmpeg(tmp_path)
    text = "First sentence here. Second sentence here. Third sentence here."
    # 3 sentences, each under the 24-char budget (chunk_tokens=6 -> 6*4 chars) -> 3 chunks.
    # Extra responses are supplied so a wrong chunk count fails as an ASSERTION, not as a
    # StopIteration leaking out of the mock.
    client = make_client(responses=[
        make_response(content=b"AAA"), make_response(content=b"BBB"),
        make_response(content=b"CCC"), make_response(content=b"DDD"),
        make_response(content=b"EEE"),
    ])
    result = await run_tts(tmp_path, text=text, client=client, fake_ffmpeg=fake, chunk_tokens=6)
    assert result.is_error is False
    assert client.stream.call_count == 3
    out = list((tmp_path / "media" / "tts").glob("*.mp3"))[0]
    assert out.read_bytes() == b"AAABBBCCC"
    assert "3 chunks" in result.content


@pytest.mark.asyncio
async def test_multi_chunk_uses_stream_copy(tmp_path):
    fake, log = make_fake_ffmpeg(tmp_path)
    # chunk_tokens=3 -> 12-char budget, so each sentence is word-split: the chunk count is
    # not the point of this test, so the response repeats.
    client = make_client(repeat=make_response(content=b"A"))
    await run_tts(
        tmp_path, text="Alpha sentence. Beta sentence.", client=client,
        fake_ffmpeg=fake, chunk_tokens=3,
    )
    argv = log.read_text()
    assert "-c" in argv.split() and "copy" in argv.split()
    assert "-f" in argv.split() and "concat" in argv.split()


@pytest.mark.asyncio
async def test_multi_chunk_removes_temp_dir(tmp_path):
    fake, log = make_fake_ffmpeg(tmp_path)
    client = make_client(repeat=make_response(content=b"A"))
    await run_tts(
        tmp_path, text="Alpha sentence. Beta sentence.", client=client,
        fake_ffmpeg=fake, chunk_tokens=3,
    )
    argv = log.read_text().split("\n")
    listfile = Path(argv[argv.index("-i") + 1])
    assert not listfile.parent.exists(), "chunk temp dir must be cleaned up"


@pytest.mark.asyncio
async def test_chunk_bodies_come_from_normalized_text(tmp_path):
    client = make_client(responses=[make_response()])
    await run_tts(
        tmp_path, text="Version 0.19.0 shipped today.", client=client,
        normalize_text=True, chunk_tokens=1000,
    )
    assert posted_bodies(client)[0]["input"] == normalize("Version 0.19.0 shipped today.")


@pytest.mark.asyncio
async def test_ffmpeg_missing_fails_soft_with_steering(tmp_path):
    client = make_client(repeat=make_response(content=b"A"))
    result = await run_tts(
        tmp_path, text="Alpha sentence. Beta sentence.", client=client,
        fake_ffmpeg=None, ffmpeg_path="/nonexistent/ffmpeg", chunk_tokens=3,
    )
    assert result.is_error is True
    assert "ffmpeg not found" in result.content
    assert "ffmpeg_path" in result.content


@pytest.mark.asyncio
async def test_ffmpeg_failure_fails_soft(tmp_path):
    fake, log = make_fake_ffmpeg(tmp_path, fail=True, exit_code=3)
    client = make_client(repeat=make_response(content=b"A"))
    result = await run_tts(
        tmp_path, text="Alpha sentence. Beta sentence.", client=client,
        fake_ffmpeg=fake, chunk_tokens=3,
    )
    assert result.is_error is True
    assert "concatenation failed" in result.content


# --- service failures --------------------------------------------------------


@pytest.mark.asyncio
async def test_http_500_fails_soft_with_status_and_body(tmp_path):
    client = make_client(responses=[make_response(500, content=b"upstream exploded")])
    result = await run_tts(tmp_path, client=client)
    assert result.is_error is True
    assert "service error HTTP 500" in result.content
    assert "upstream exploded" in result.content


@pytest.mark.asyncio
async def test_http_400_fails_soft(tmp_path):
    client = make_client(responses=[make_response(400, content=b"bad voice")])
    result = await run_tts(tmp_path, client=client)
    assert result.is_error is True
    assert "HTTP 400" in result.content


@pytest.mark.asyncio
async def test_timeout_fails_soft_with_endpoint(tmp_path):
    client = make_client(side_effect=httpx.TimeoutException("too slow"))
    result = await run_tts(tmp_path, client=client)
    assert result.is_error is True
    assert "timed out" in result.content
    assert ENDPOINT in result.content


@pytest.mark.asyncio
async def test_connect_error_fails_soft_with_endpoint(tmp_path):
    client = make_client(side_effect=httpx.ConnectError("connection refused"))
    result = await run_tts(tmp_path, client=client)
    assert result.is_error is True
    assert "cannot reach" in result.content
    assert ENDPOINT in result.content


@pytest.mark.asyncio
async def test_json_error_body_on_200_is_detected(tmp_path):
    client = make_client(responses=[make_response(200, content=b'{"detail":"boom"}')])
    result = await run_tts(tmp_path, client=client)
    assert result.is_error is True
    assert "JSON error body instead of audio" in result.content


@pytest.mark.asyncio
async def test_zero_byte_audio_rejected(tmp_path):
    client = make_client(responses=[make_response(200, content=b"")])
    result = await run_tts(tmp_path, client=client)
    assert result.is_error is True
    assert "0 bytes" in result.content


@pytest.mark.asyncio
async def test_oversize_audio_response_rejected(tmp_path):
    client = make_client(responses=[make_response(200, content=b"x" * 100)])
    result = await run_tts(tmp_path, client=client, max_audio_bytes=10)
    assert result.is_error is True
    assert "too large" in result.content.lower()


@pytest.mark.asyncio
async def test_no_file_written_on_service_failure(tmp_path):
    client = make_client(responses=[make_response(500, content=b"nope")])
    result = await run_tts(tmp_path, client=client)
    assert result.is_error is True
    out_dir = tmp_path / "media" / "tts"
    assert not out_dir.exists() or not list(out_dir.glob("*.mp3"))


# --- room delivery -----------------------------------------------------------


@pytest.mark.asyncio
async def test_send_to_room_false_does_not_upload(tmp_path):
    callback = AsyncMock()
    client = make_client(responses=[make_response()])
    result = await run_tts(tmp_path, client=client, callback=callback)
    assert result.is_error is False
    callback.assert_not_awaited()
    assert "sent to room" not in result.content


@pytest.mark.asyncio
async def test_send_to_room_true_uploads_with_caption(tmp_path):
    callback = AsyncMock()
    client = make_client(responses=[make_response()])
    result = await run_tts(
        tmp_path, client=client, callback=callback, send_to_room=True, caption="Voice reply",
    )
    assert result.is_error is False
    callback.assert_awaited_once()
    kwargs = callback.await_args.kwargs
    assert str(kwargs["file_path"]).endswith(".mp3")
    assert kwargs["caption"] == "Voice reply"
    assert "sent to room" in result.content


@pytest.mark.asyncio
async def test_send_to_room_without_callback_still_reports_the_file(tmp_path):
    client = make_client(responses=[make_response()])
    result = await run_tts(tmp_path, client=client, send_to_room=True, callback=None)
    assert result.is_error is False
    assert "media/tts/" in result.content
    # An explicit send request that could not be honoured must SAY so (the bare
    # success marker "; sent to room" must not appear).
    assert "NOT sent" in result.content


@pytest.mark.asyncio
async def test_upload_failure_is_error_but_keeps_path(tmp_path):
    callback = AsyncMock(side_effect=RuntimeError("upload boom"))
    client = make_client(responses=[make_response()])
    result = await run_tts(tmp_path, client=client, callback=callback, send_to_room=True)
    assert result.is_error is True
    assert "media/tts/" in result.content


# --- dispatch integration ----------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_dispatches_tts(tmp_path):
    client = make_client(responses=[make_response(content=b"DISPATCHED")])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await execute_tool(
            "tts",
            {"text": "Hello from dispatch."},
            cfg(endpoint=ENDPOINT, model=KOKORO_MODEL),
            make_agent_config(tmp_path),
        )
    assert result.is_error is False
    assert (tmp_path / "media" / "tts").is_dir()


@pytest.mark.asyncio
async def test_execute_tool_rejects_missing_text(tmp_path):
    result = await execute_tool("tts", {}, cfg(endpoint=ENDPOINT), make_agent_config(tmp_path))
    assert result.is_error is True
    assert "text" in result.content


@pytest.mark.asyncio
async def test_execute_tool_passes_send_media_callback(tmp_path):
    callback = AsyncMock()
    client = make_client(responses=[make_response()])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await execute_tool(
            "tts",
            {"text": "Hello.", "send_to_room": True},
            cfg(endpoint=ENDPOINT),
            make_agent_config(tmp_path),
            callbacks={"send_media": callback},
        )
    assert result.is_error is False
    callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_tts_not_enabled_means_unknown_tool(tmp_path):
    """The tool is opt-in per agent via workspace/tools/tts.toml."""
    assert "tts" in BUILTIN_TOOLS
    (tmp_path / "tools").mkdir()
    assert "tts" not in {t.name for t in discover_tools(tmp_path)}
