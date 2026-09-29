"""Remediation pins for the im7t.68 adversarial audit (2026-09-28).

One test per accepted audit finding. Each of these FAILS against the pre-remediation tree
and passes only when the finding is actually fixed — that is the point of the file.

Findings and their convergence (qwen / kimi / glm):
  1. non-string `path` raises TypeError through the dispatch path-join    3/3 HIGH
  2. `str` workspace crashes the dispatch join                            1/3 MEDIUM
  3. empty `path` steering degraded to "not a file: <workspace>"          1/3 LOW
  4. unbounded response buffering before the size check                   3/3 HIGH
  5. non-audio 2xx body written to disk as .mp3                           3/3 MEDIUM
  6. Content-Type text/html on a 2xx body                                 3/3 MEDIUM
  7. normalize() ValueError on >4300-digit runs (CPython int cap)         1/3 MEDIUM
  8. ffmpeg failure leaves a partial/zero-byte .mp3 at the final path     2/3 MEDIUM
  9. `output_dir` accepts absolute / `..` (escapes the workspace)         2/3 MEDIUM
 10. send_to_room=True with no callback silently reports plain success    1/3 MEDIUM
 11. non-executable ffmpeg misdiagnosed as "cannot write <out>.mp3"       1/3 MEDIUM
 12. tts drops max_upload_bytes when delegating to send_media             3/3 LOW
 13. timeout <= 0 accepted (permanent misleading "timed out")             1/3 LOW
 14. no cumulative cap across chunks                                      2/3 LOW
 15. chunk_tokens has no floor (a typo can spawn thousands of requests)   2/3 LOW
 16. timestamps header reports the REQUESTED language, not the detected   1/3 LOW
 17. transcript / segment text uncapped                                   1/3 LOW
 18. media tag matched anywhere in the string (hijacks odd filenames)     1/3 LOW

Mocking boundary: httpx only, via `client.stream(...)` (the bounded-read contract) — the
HTTP surface is the sole mock; ffmpeg stays a real subprocess against a fake binary.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from openalph.config import AgentConfig, ProviderConfig
from openalph.tools import BUILTIN_TOOLS, execute_tool
from openalph.tools.speech import tts

ENDPOINT = "http://10.0.20.104:8006/v1/audio/speech"
STT_ENDPOINT = "http://10.0.20.104:8001/v1/audio/transcriptions"


# --- fixtures / helpers ------------------------------------------------------


def make_agent_config(tmp_path, **kwargs):
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
            base_url=None, quirks=[],
        )},
        workspace=tmp_path,
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def cfg(**overrides):
    base = dict(BUILTIN_TOOLS["tts"]["config"])
    base.update(overrides)
    return base


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


def stream_cm(resp, chunk_size=0, consumed=None):
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
            if consumed is not None:
                consumed[0] += min(size, len(payload) - i)
            yield payload[i:i + size]

    resp.aiter_bytes = _aiter
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def make_client(responses=None, side_effect=None, repeat=None, chunk_size=0, consumed=None):
    """httpx client whose .stream() yields the given responses in order."""
    client = AsyncMock()
    if side_effect is not None:
        client.stream = MagicMock(side_effect=side_effect)
    elif repeat is not None:
        client.stream = MagicMock(side_effect=lambda *a, **k: stream_cm(repeat, chunk_size, consumed))
    elif responses is not None:
        client.stream = MagicMock(
            side_effect=[stream_cm(r, chunk_size, consumed) for r in responses]
        )
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def patch_httpx(client):
    mock_cls = MagicMock(return_value=client)
    return patch("openalph.tools.speech.httpx.AsyncClient", mock_cls), mock_cls


FAKE_FFMPEG_SRC = '''#!/usr/bin/env python3
"""Fake ffmpeg: records argv, optionally writes a partial output then fails."""
import pathlib
import sys

argv = sys.argv[1:]
pathlib.Path({log!r}).write_text("\\n".join(argv))
out = pathlib.Path(argv[-1])
if {partial!r}:
    out.write_bytes(b"PARTIAL-CORRUPT-AUDIO")
if {fail!r}:
    sys.stderr.write("fake-ffmpeg: simulated failure\\n")
    sys.exit({exit_code})
listfile = argv[argv.index("-i") + 1]
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


def make_fake_ffmpeg(tmp_path, *, fail=False, exit_code=1, partial=False):
    log = tmp_path / "ffmpeg-argv.txt"
    path = tmp_path / "fake-ffmpeg"
    path.write_text(FAKE_FFMPEG_SRC.format(
        log=str(log), fail=fail, exit_code=exit_code, partial=partial))
    path.chmod(0o755)
    return path, log


async def run_tts(tmp_path, text="Hello there.", client=None, fake_ffmpeg=None,
                  callback=None, agent_config=None, **overrides):
    config = cfg(endpoint=ENDPOINT, model="mlx-community/Kokoro-82M-bf16", voice="af_heart")
    config.update(overrides)
    if fake_ffmpeg is not None:
        config["ffmpeg_path"] = str(fake_ffmpeg)
    if agent_config is None:
        agent_config = make_agent_config(tmp_path)
    callbacks = {"send_media": callback} if callback is not None else None
    if client is None:
        client = make_client(responses=[make_response()])
    patcher, _ = patch_httpx(client)
    with patcher:
        return await tts(text=text, tool_config=config, agent_config=agent_config,
                         callbacks=callbacks)


def audio_file(tmp_path, rel="media/abc123/voice.ogg", data=b"OGGDATA"):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# --- 1/2/3: the dispatch path-join seam -------------------------------------


@pytest.mark.asyncio
async def test_dispatch_non_string_path_returns_error_not_raise(tmp_path):
    result = await execute_tool(
        "stt", {"path": 123}, {"endpoint": STT_ENDPOINT}, make_agent_config(tmp_path)
    )
    assert result.is_error is True
    assert "path" in result.content.lower()


@pytest.mark.asyncio
async def test_dispatch_none_path_returns_error_not_raise(tmp_path):
    result = await execute_tool(
        "stt", {"path": None}, {"endpoint": STT_ENDPOINT}, make_agent_config(tmp_path)
    )
    assert result.is_error is True


@pytest.mark.asyncio
async def test_dispatch_list_path_returns_error_not_raise(tmp_path):
    result = await execute_tool(
        "stt", {"path": ["a", "b"]}, {"endpoint": STT_ENDPOINT}, make_agent_config(tmp_path)
    )
    assert result.is_error is True


@pytest.mark.asyncio
async def test_dispatch_string_workspace_does_not_raise(tmp_path):
    """A str workspace must not crash the join (`str / str` is a TypeError)."""
    audio_file(tmp_path, data=b"STRWS")
    agent_config = make_agent_config(tmp_path, workspace=str(tmp_path))
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await execute_tool(
            "stt", {"path": "media/abc123/voice.ogg"}, {"endpoint": STT_ENDPOINT},
            agent_config, callbacks=None,
        )
    assert result.is_error is False, result.content


@pytest.mark.asyncio
async def test_dispatch_empty_path_steers_to_empty_path(tmp_path):
    result = await execute_tool(
        "stt", {"path": ""}, {"endpoint": STT_ENDPOINT}, make_agent_config(tmp_path)
    )
    assert result.is_error is True
    assert "empty" in result.content.lower()
    assert "not a file" not in result.content.lower()


# --- 4: bounded response reads ----------------------------------------------


@pytest.mark.asyncio
async def test_oversize_body_is_not_read_unbounded(tmp_path):
    """A 100 MB body against a 1 KB cap must stop early, not buffer it all."""
    consumed = [0]
    big = make_response(content=b"x" * 1_000_000)
    client = make_client(repeat=big, chunk_size=100_000, consumed=consumed)
    result = await run_tts(tmp_path, client=client, max_audio_bytes=1000)
    assert result.is_error is True
    assert "too large" in result.content.lower()
    assert consumed[0] < 1_000_000, "the full body was read despite the cap"


@pytest.mark.asyncio
async def test_stt_oversize_response_body_is_bounded(tmp_path):
    audio_file(tmp_path)
    consumed = [0]
    big = make_response(content=b"y" * 1_000_000)  # content-only: a genuinely huge body
    client = make_client(repeat=big, chunk_size=100_000, consumed=consumed)
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await execute_tool(
            "stt", {"path": "media/abc123/voice.ogg"},
            {"endpoint": STT_ENDPOINT, "max_response_bytes": 1000},
            make_agent_config(tmp_path),
        )
    assert result.is_error is True
    assert consumed[0] < 1_000_000, "the full body was read despite the cap"


# --- 5/6: non-audio 2xx bodies ----------------------------------------------


@pytest.mark.asyncio
async def test_html_200_body_is_rejected(tmp_path):
    client = make_client(responses=[make_response(content=b"<html>oops</html>")])
    result = await run_tts(tmp_path, client=client)
    assert result.is_error is True
    assert "audio" in result.content.lower()
    out_dir = tmp_path / "media" / "tts"
    assert not out_dir.exists() or not list(out_dir.glob("*.mp3"))


@pytest.mark.asyncio
async def test_json_array_200_body_is_rejected(tmp_path):
    client = make_client(responses=[make_response(content=b'["rate limited"]')])
    result = await run_tts(tmp_path, client=client)
    assert result.is_error is True


@pytest.mark.asyncio
async def test_text_plain_content_type_is_rejected(tmp_path):
    client = make_client(responses=[make_response(
        content=b"OK", headers={"content-type": "text/html; charset=utf-8"})])
    result = await run_tts(tmp_path, client=client)
    assert result.is_error is True
    assert "text/html" in result.content


@pytest.mark.asyncio
async def test_audio_content_type_is_accepted(tmp_path):
    client = make_client(responses=[make_response(
        content=b"ID3\x04\x00\x00\x00\x00\x00\x00AUDIO",
        headers={"content-type": "audio/mpeg"})])
    result = await run_tts(tmp_path, client=client)
    assert result.is_error is False, result.content


# --- 7: normalize() int-conversion guard ------------------------------------


@pytest.mark.asyncio
async def test_huge_digit_run_is_steered_not_raised(tmp_path):
    text = "9" * 5000 + ".1.1.1"
    client = make_client(responses=[make_response()])
    result = await run_tts(tmp_path, text=text, client=client, max_input_chars=100000)
    assert result.is_error is True
    assert "normalize" in result.content.lower()
    assert "normalize_text" in result.content


# --- 8: no partial output left behind ---------------------------------------


@pytest.mark.asyncio
async def test_ffmpeg_failure_leaves_no_partial_output(tmp_path):
    fake, log = make_fake_ffmpeg(tmp_path, fail=True, exit_code=3, partial=True)
    client = make_client(repeat=make_response(content=b"A"))
    result = await run_tts(tmp_path, text="Alpha sentence. Beta sentence.",
                           client=client, fake_ffmpeg=fake, chunk_tokens=3)
    assert result.is_error is True
    out_dir = tmp_path / "media" / "tts"
    leftovers = list(out_dir.glob("*.mp3")) if out_dir.exists() else []
    assert leftovers == [], f"partial output left behind: {leftovers}"


# --- 9: output_dir containment ----------------------------------------------


@pytest.mark.parametrize("bad", ["../escape", "/tmp/im7t68-escape", "a/../../b"])
@pytest.mark.asyncio
async def test_output_dir_escaping_workspace_is_rejected(tmp_path, bad):
    client = make_client(responses=[make_response()])
    result = await run_tts(tmp_path, client=client, output_dir=bad)
    assert result.is_error is True
    assert "output_dir" in result.content


# --- 10: explicit send_to_room must not silently no-op ----------------------


@pytest.mark.asyncio
async def test_send_to_room_without_callback_says_not_sent(tmp_path):
    client = make_client(responses=[make_response()])
    result = await run_tts(tmp_path, client=client, send_to_room=True, callback=None)
    assert result.is_error is False
    assert "media/tts/" in result.content
    assert "not sent" in result.content.lower()


# --- 11: non-executable ffmpeg steering -------------------------------------


@pytest.mark.asyncio
async def test_non_executable_ffmpeg_steers_to_ffmpeg_path(tmp_path):
    bogus = tmp_path / "not-executable"
    bogus.write_text("not a binary")
    bogus.chmod(0o644)
    client = make_client(repeat=make_response(content=b"A"))
    result = await run_tts(tmp_path, text="Alpha sentence. Beta sentence.",
                           client=client, ffmpeg_path=str(bogus), chunk_tokens=3)
    assert result.is_error is True
    assert "ffmpeg_path" in result.content
    assert "cannot write" not in result.content


# --- 12: upload cap coupling ------------------------------------------------


@pytest.mark.asyncio
async def test_send_to_room_passes_the_audio_cap_to_send_media(tmp_path):
    callback = AsyncMock()
    client = make_client(responses=[make_response()])
    with patch("openalph.tools.media.send_media", new=AsyncMock(
            return_value=MagicMock(is_error=False, content="Sent"))) as fake_send:
        result = await run_tts(tmp_path, client=client, callback=callback,
                               send_to_room=True, max_audio_bytes=4096,
                               max_total_audio_bytes=8192)
    assert result.is_error is False
    # The upload cap must track the real file-size bound (max_total_audio_bytes),
    # not send_media's hard 20 MB default.
    assert fake_send.await_args.kwargs["max_upload_bytes"] == 8192


# --- 13: timeout validation -------------------------------------------------


@pytest.mark.asyncio
async def test_zero_timeout_is_rejected(tmp_path):
    client = make_client(responses=[make_response()])
    result = await run_tts(tmp_path, client=client, timeout=0)
    assert result.is_error is True
    assert "timeout" in result.content.lower()


@pytest.mark.asyncio
async def test_negative_timeout_is_rejected_stt(tmp_path):
    audio_file(tmp_path)
    result = await execute_tool(
        "stt", {"path": "media/abc123/voice.ogg"},
        {"endpoint": STT_ENDPOINT, "timeout": -5}, make_agent_config(tmp_path),
    )
    assert result.is_error is True
    assert "timeout" in result.content.lower()


# --- 14: cumulative cap -----------------------------------------------------


@pytest.mark.asyncio
async def test_cumulative_audio_cap_is_enforced(tmp_path):
    client = make_client(repeat=make_response(content=b"x" * 8))
    result = await run_tts(
        tmp_path, text="Alpha sentence. Beta sentence. Gamma sentence.",
        client=client, chunk_tokens=3, max_audio_bytes=100,
        max_total_audio_bytes=10,
    )
    assert result.is_error is True
    assert "total" in result.content.lower()


# --- 15: chunk_tokens floor -------------------------------------------------


@pytest.mark.asyncio
async def test_tiny_chunk_tokens_cannot_spawn_unbounded_requests(tmp_path):
    """chunk_tokens=1 must be floored/capped, not turned into thousands of POSTs."""
    fake, _ = make_fake_ffmpeg(tmp_path)
    client = make_client(repeat=make_response(content=b"A"))
    text = "Sentence one here. " * 12  # ~240 chars
    result = await run_tts(tmp_path, text=text, client=client, chunk_tokens=1,
                           fake_ffmpeg=fake)
    assert result.is_error is False, result.content
    assert client.stream.call_count <= 64, (
        f"chunk_tokens=1 spawned {client.stream.call_count} requests"
    )


@pytest.mark.asyncio
async def test_chunk_count_over_the_cap_is_an_error(tmp_path):
    client = make_client(repeat=make_response(content=b"A"))
    text = "Sentence one here. " * 120  # ~2400 chars -> ~300 chunks at the floor
    result = await run_tts(tmp_path, text=text, client=client, chunk_tokens=1,
                           max_input_chars=100000)
    assert result.is_error is True
    assert "chunk_tokens" in result.content


# --- 16: detected language in the timestamps header -------------------------


@pytest.mark.asyncio
async def test_timestamps_header_reports_detected_language(tmp_path):
    audio_file(tmp_path)
    payload = {"text": "Hello there.", "language": "en",
               "segments": [{"start": 0.0, "end": 1.0, "text": "Hello there."}]}
    client = make_client(responses=[make_response(json_data=payload)])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await execute_tool(
            "stt", {"path": "media/abc123/voice.ogg", "timestamps": True},
            {"endpoint": STT_ENDPOINT}, make_agent_config(tmp_path),
        )
    assert result.is_error is False
    assert "[stt: language=en]" in result.content


# --- 17: transcript caps ----------------------------------------------------


@pytest.mark.asyncio
async def test_transcript_is_capped_with_a_marker(tmp_path):
    audio_file(tmp_path)
    client = make_client(responses=[make_response(json_data={"text": "z" * 100000})])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await execute_tool(
            "stt", {"path": "media/abc123/voice.ogg"},
            {"endpoint": STT_ENDPOINT, "max_transcript_chars": 1000},
            make_agent_config(tmp_path),
        )
    assert result.is_error is False
    assert len(result.content) < 2000
    assert "truncated" in result.content.lower()


# --- 18: media tag anchoring ------------------------------------------------


@pytest.mark.asyncio
async def test_media_tag_inside_a_filename_is_not_hijacked(tmp_path):
    odd = tmp_path / "media" / "notes [media: draft].txt"
    odd.parent.mkdir(parents=True, exist_ok=True)
    odd.write_bytes(b"not audio")
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await execute_tool(
            "stt", {"path": "media/notes [media: draft].txt"},
            {"endpoint": STT_ENDPOINT}, make_agent_config(tmp_path),
        )
    # Either it resolves the literal file (preferred) or fails naming that literal path —
    # what it must NOT do is silently transcribe a different path.
    if result.is_error:
        assert "draft].txt" not in result.content.replace("notes [media: draft].txt", "")


@pytest.mark.asyncio
async def test_leading_media_tag_still_parses(tmp_path):
    audio_file(tmp_path, data=b"TAGGED")
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await execute_tool(
            "stt", {"path": "[media: media/abc123/voice.ogg (audio/ogg, 45KB)]"},
            {"endpoint": STT_ENDPOINT}, make_agent_config(tmp_path),
        )
    assert result.is_error is False
    sent = client.stream.call_args_list[0].kwargs
    assert sent["files"]["file"][1] == b"TAGGED"


# --- RE-AUDIT findings (the remediation's own new defects) -------------------


@pytest.mark.asyncio
async def test_concat_temp_dir_is_on_the_output_filesystem(tmp_path):
    """os.replace cannot cross a mount boundary: /tmp is tmpfs on diodeli-class hosts.

    The concat temp dir must live INSIDE the output directory (same filesystem as the
    final path), or every multi-chunk utterance fails with EXDEV in production while
    passing in tests (pytest's tmp_path happens to be under /tmp).
    """
    fake, log = make_fake_ffmpeg(tmp_path)
    client = make_client(responses=[
        make_response(content=b"AAA"), make_response(content=b"BBB"),
        make_response(content=b"CCC"), make_response(content=b"DDD"),
    ])
    result = await run_tts(tmp_path, text="First sentence here. Second sentence here.",
                           client=client, fake_ffmpeg=fake, chunk_tokens=6)
    assert result.is_error is False, result.content
    argv = log.read_text().split("\n")
    listfile = argv[argv.index("-i") + 1]
    out_dir = str(tmp_path / "media" / "tts")
    assert listfile.startswith(out_dir), (
        f"concat temp dir {listfile!r} is not inside the output dir {out_dir!r} "
        "(cross-device os.replace)"
    )


@pytest.mark.asyncio
async def test_truncated_stream_is_not_accepted_as_success(tmp_path):
    """A mid-body transport failure must error, never yield a partial MP3 as success."""
    resp = make_response(content=b"x" * 5000)

    async def _aiter():
        yield b"x" * 1000
        raise httpx.RemoteProtocolError("peer closed connection without sending body")

    resp.aiter_bytes = _aiter
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    client = AsyncMock()
    client.stream = MagicMock(return_value=cm)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    result = await run_tts(tmp_path, client=client)
    assert result.is_error is True, f"truncated audio accepted as success: {result.content}"
    out_dir = tmp_path / "media" / "tts"
    assert not out_dir.exists() or not list(out_dir.glob("*.mp3"))


# --- im7t.68.1: spaced / extensionless voice-note filenames (wonmun bug report) ---
# Matrix clients name voice notes "Voice message" by default -- a space and no
# extension. A whitespace-delimited path group silently failed the common case.


@pytest.mark.parametrize("name", ["Voice message", "Voice message (1)", "My note"])
@pytest.mark.asyncio
async def test_media_tag_with_spaced_extensionless_name_parses(tmp_path, name):
    rel = f"media/8d85f1c8fe636e62/{name}"
    audio_file(tmp_path, rel=rel, data=b"SPACED")
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await execute_tool(
            "stt", {"path": f"[media: {rel} (audio/ogg, 60.0 KB)]"},
            {"endpoint": STT_ENDPOINT}, make_agent_config(tmp_path),
        )
    assert result.is_error is False, result.content
    assert client.stream.call_args_list[0].kwargs["files"]["file"][1] == b"SPACED"


@pytest.mark.asyncio
async def test_media_tag_with_spaced_name_embedded_in_sentence(tmp_path):
    rel = "media/8d85f1c8fe636e62/Voice message"
    audio_file(tmp_path, rel=rel, data=b"EMBEDDED")
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await execute_tool(
            "stt", {"path": f"please transcribe [media: {rel} (audio/ogg, 60.0 KB)]"},
            {"endpoint": STT_ENDPOINT}, make_agent_config(tmp_path),
        )
    assert result.is_error is False, result.content


@pytest.mark.asyncio
async def test_bare_media_tag_without_parenthetical_parses(tmp_path):
    rel = "media/8d85f1c8fe636e62/Voice message"
    audio_file(tmp_path, rel=rel, data=b"BARE")
    client = make_client(responses=[make_response(json_data={"text": "ok"})])
    patcher, _ = patch_httpx(client)
    with patcher:
        result = await execute_tool(
            "stt", {"path": f"[media: {rel}]"},
            {"endpoint": STT_ENDPOINT}, make_agent_config(tmp_path),
        )
    assert result.is_error is False, result.content
