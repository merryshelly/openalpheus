"""Speech tools: TTS text normalization, chunking, synthesis, and transcription.

normalize() is a verbatim port of scripts/tts-say's normalizer: the same
regexes, the same ACRONYMS allow-list, the same _n2/spell_number/
_four_as_pairs/_ip_octet helpers, and the same five-pass order. Text no rule
applies to passes through byte-identical. The parity contract is
tests/test_speech_normalize.py -- do not "improve" any rule.

chunk_text() packs sentences for per-chunk synthesis; its contract is
tests/test_speech_chunk.py.
"""

import asyncio
import hashlib
import json
import mimetypes
import os
import re
import shutil
import ssl
import tempfile
from datetime import datetime, timezone

import httpx

from openalph.tools import ToolResult

# Use the system SSL context so httpx picks up system CA certs
_ssl_context = ssl.create_default_context()

ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
TEENS = ("ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen")
TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")

# Only these standalone uppercase runs are spelled out. Never "all caps".
ACRONYMS = frozenset({
    "SSH", "HTTPS", "HTTP", "TLS", "DNS", "API", "CPU", "GPU", "RAM", "SSD",
    "TTS", "STT", "URL", "ID", "IP", "CVE", "RFC", "ALF", "VLAN",
})

# A dotted numeric run (2+ components). The leading (?<![\d.]) prevents
# matching the tail of a longer digit/dot group (e.g. "1.2.3.4" inside
# "11.2.3.4"), while still allowing a trailing sentence period.
_DOTTED_RE = re.compile(r"(?<![\d.])\d+(?:\.\d+)+")
_CVE_RE = re.compile(r"\bCVE-(\d{4})-(\d+)\b", re.IGNORECASE)
_PORT_RE = re.compile(r"\b(port)([\s:]+)(\d{1,6})\b", re.IGNORECASE)
_FOUR_DIGIT_RE = re.compile(r"(?<![\d.])\d{4}(?![\d.])")
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,})\b")


def _n2(n):
    """Spell 0-99; compound tens hyphenated (26 -> 'twenty-six')."""
    if n < 10:
        return ONES[n]
    if n < 20:
        return TEENS[n - 10]
    t, o = divmod(n, 10)
    return TENS[t] + ("-" + ONES[o] if o else "")


def spell_number(n):
    """Spell an integer in cardinal form.

    8000 -> 'eight thousand'; 8001 -> 'eight thousand and one';
    8006 -> 'eight thousand and six'.
    """
    if n < 0:
        raise ValueError("negative numbers are not supported")
    if n < 100:
        return _n2(n)
    if n < 1000:
        h, r = divmod(n, 100)
        return ONES[h] + " hundred" + (" and " + _n2(r) if r else "")
    if n < 1_000_000:
        t, r = divmod(n, 1000)
        return spell_number(t) + " thousand" + (" and " + spell_number(r) if r else "")
    m, r = divmod(n, 1_000_000)
    return spell_number(m) + " million" + (" " + spell_number(r) if r else "")


def _four_as_pairs(n):
    """Read a 4-digit number year-style in pairs: 2026 -> 'twenty twenty-six'."""
    hi, lo = divmod(n, 100)
    return _n2(hi) + " " + _n2(lo)


def _ip_octet(s):
    n = int(s)
    if n >= 100:
        # Three-digit octets are read digit-by-digit, zero as "oh":
        # 104 -> "one oh four", 101 -> "one oh one".
        return " ".join("oh" if c == "0" else ONES[int(c)] for c in s)
    return _n2(n)


def _replace_dotted(m):
    s = m.group(0)
    parts = s.split(".")
    if len(parts) == 4 and all(p.isdigit() and int(p) <= 255 for p in parts):
        return " dot ".join(_ip_octet(p) for p in parts)
    if len(parts) >= 3:
        return " point ".join(spell_number(int(p)) for p in parts)
    return s  # exactly 2 components: a decimal/measurement, leave untouched


def _replace_cve(m):
    year = int(m.group(1))
    ident = m.group(2)
    id_spoken = _four_as_pairs(int(ident)) if len(ident) == 4 else spell_number(int(ident))
    return "C V E " + _four_as_pairs(year) + " " + id_spoken


def _replace_port(m):
    return m.group(1) + m.group(2) + spell_number(int(m.group(3)))


def _replace_four(m):
    return _four_as_pairs(int(m.group(0)))


def _replace_acronym(m):
    w = m.group(1)
    return " ".join(w) if w in ACRONYMS else w


def normalize(text):
    """Deterministically normalize technical strings for TTS speech.

    Order matters: IPs/versions (dotted tokens) first so their digits are
    consumed before the port/4-digit/acronym passes; every pass emits
    lowercase words, so later passes can never match inside earlier output.
    Text no rule applies to is returned byte-identical.
    """
    out = _DOTTED_RE.sub(_replace_dotted, text)
    out = _CVE_RE.sub(_replace_cve, out)
    out = _PORT_RE.sub(_replace_port, out)
    out = _FOUR_DIGIT_RE.sub(_replace_four, out)
    out = _ACRONYM_RE.sub(_replace_acronym, out)
    return out


# --- Chunking ----------------------------------------------------------------


# Sentence boundary: [.!?] followed by whitespace. (Newlines are handled
# separately so a paragraph break is always a boundary even without a
# sentence terminator.)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_oversize_sentence(sentence, chunk_chars):
    """Pack the words of an oversize sentence into <= chunk_chars pieces.

    Splits on spaces; a single word longer than chunk_chars is hard-cut on
    character boundaries (never mid-codepoint).
    """
    pieces = []
    current = ""
    for word in sentence.split(" "):
        if not word:
            continue
        if len(word) > chunk_chars:
            if current:
                pieces.append(current)
                current = ""
            for i in range(0, len(word), chunk_chars):
                pieces.append(word[i:i + chunk_chars])
            continue
        candidate = word if not current else current + " " + word
        if len(candidate) <= chunk_chars:
            current = candidate
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text, chunk_chars):
    """Split text into TTS synthesis chunks of at most chunk_chars chars.

    chunk_chars <= 0 disables chunking: [text.strip()] (or [] when the text
    is whitespace-only). Otherwise, sentences are delimited by [.!?]
    followed by whitespace (and by newlines) and whole sentences are packed
    greedily into chunks joined by a single space. A sentence longer than
    the budget falls back to word boundaries; a word longer than the budget
    falls back to a hard character cut. Deterministic, order-preserving;
    empty chunks are dropped.
    """
    if chunk_chars <= 0:
        stripped = text.strip()
        return [stripped] if stripped else []

    sentences = []
    for part in _SENTENCE_SPLIT_RE.split(text):
        for line in part.split("\n"):
            line = line.strip()
            if line:
                sentences.append(line)

    chunks = []
    current = ""
    for sentence in sentences:
        if len(sentence) > chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_oversize_sentence(sentence, chunk_chars))
            continue
        candidate = sentence if not current else current + " " + sentence
        if len(candidate) <= chunk_chars:
            current = candidate
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


# --- TTS synthesis -----------------------------------------------------------


def _error(msg):
    return ToolResult(content=msg, is_error=True)


# Chunking floors (post-audit, 2026-09-28): chunk_tokens is config-only, so a typo
# must not be able to turn one call into thousands of requests or an unbounded run.
_MIN_CHUNK_CHARS = 8
_MAX_CHUNKS = 64

# STT response/transcript bounds (the service is untrusted input).
_MAX_SEGMENT_TEXT_CHARS = 500


def _content_type(resp):
    """Response content-type as a str, or "" when absent/unusable (mocks, weird SDKs)."""
    headers = getattr(resp, "headers", None)
    try:
        value = headers.get("content-type") if headers is not None else None
    except Exception:
        value = None
    return value if isinstance(value, str) and value else ""


async def _read_bounded(resp, limit):
    """Read at most ``limit`` bytes from a STREAMING response.

    The endpoint is untrusted input: buffering the whole body before checking its
    size lets one oversized (or endless) response OOM the agent process. Mirrors
    web.py's stream-with-a-cap idiom. A stream that dies mid-read is not an error
    here -- the caller's own checks (empty / too large / not-audio) handle it.
    """
    buf = bytearray()
    try:
        async for piece in resp.aiter_bytes():
            if not isinstance(piece, (bytes, bytearray)):
                continue
            buf.extend(piece)
            if len(buf) >= limit:
                break
    except Exception:
        pass
    return bytes(buf[:limit])


def _unlink_quietly(path):
    """Best-effort removal -- never let cleanup mask the real error."""
    try:
        os.unlink(path)
    except OSError:
        pass


def _reserve_output_path(out_dir, base):
    """Atomically reserve ``<out_dir>/<base>[-N].mp3`` (O_EXCL) and return the path.

    Reservation (rather than a check-then-write exists() loop) means two concurrent
    calls with identical text in the same second can never select the same name.
    The reserved file is empty; the caller either fills it (single chunk) or
    os.replace()s the ffmpeg result over it, and unlinks it on every failure path.
    """
    for suffix in range(0, 1000):
        name = base + ("" if suffix == 0 else f"-{suffix}") + ".mp3"
        candidate = os.path.join(out_dir, name)
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        except OSError:
            return None
        os.close(fd)
        return candidate
    return None


def _resolve_workspace(agent_config):
    """agent_config.workspace as a str path, or None if unusable.

    AgentConfig.workspace is usually a pathlib.Path; both PathLike and str
    are accepted.
    """
    ws = getattr(agent_config, "workspace", None) if agent_config is not None else None
    if isinstance(ws, os.PathLike):
        ws = os.fspath(ws)
    if isinstance(ws, str) and ws:
        return ws
    return None


async def tts(text, voice=None, speed=None, normalize_text=None, send_to_room=False,
              caption=None, tool_config=None, agent_config=None, callbacks=None):
    """Synthesize speech with the workspace TTS service and write an .mp3.

    Config (workspace/tools/tts.toml): endpoint (required), model, voice,
    speed, normalize (or the "normalize_text" alias, which wins when both
    are present), send_to_room, caption, chunk_tokens, ffmpeg_path,
    timeout, max_input_chars, max_audio_bytes, output_dir. Explicit
    parameters take precedence over config values, which take precedence
    over built-in defaults.
    Normalization runs first, then the normalized text is chunked at
    chunk_tokens*4 chars; each chunk is POSTed as JSON and the resulting
    audio is concatenated with ffmpeg (a single chunk is written directly).
    Output: <workspace>/<output_dir>/<UTC stamp>-<sha256[:8]>.mp3, never
    clobbered (a -1, -2, ... suffix is appended on collision).

    Returns:
        ToolResult with the workspace-relative output path on success, or an
        is_error result carrying a point-of-need steer. Audio bytes never
        appear in the content.
    """
    tc = tool_config if isinstance(tool_config, dict) else {}

    endpoint = tc.get("endpoint", "")
    model = tc.get("model", "")
    cfg_voice = tc.get("voice", "")
    cfg_speed = tc.get("speed", 1.0)
    cfg_normalize = tc.get("normalize", True)
    chunk_tokens = tc.get("chunk_tokens", 200)
    ffmpeg_path = tc.get("ffmpeg_path", "ffmpeg")
    timeout = tc.get("timeout", 120.0)
    max_input_chars = tc.get("max_input_chars", 20000)
    max_audio_bytes = tc.get("max_audio_bytes", 20971520)
    max_total_audio_bytes = tc.get("max_total_audio_bytes", 104857600)
    output_dir = tc.get("output_dir", "media/tts")

    # Invalid param types are an error, never a raise.
    if not isinstance(text, str):
        return _error(f"tts error: invalid text (expected str, got {type(text).__name__})")
    if not isinstance(endpoint, str):
        return _error(f"tts error: invalid endpoint (expected str, got {type(endpoint).__name__})")
    if not isinstance(max_input_chars, (int, float)) or isinstance(max_input_chars, bool) \
            or max_input_chars <= 0:
        return _error("tts error: invalid max_input_chars (expected a positive number)")
    if not isinstance(max_audio_bytes, (int, float)) or isinstance(max_audio_bytes, bool) \
            or max_audio_bytes <= 0:
        return _error("tts error: invalid max_audio_bytes (expected a positive number)")
    if not isinstance(max_total_audio_bytes, (int, float)) \
            or isinstance(max_total_audio_bytes, bool) or max_total_audio_bytes <= 0:
        return _error("tts error: invalid max_total_audio_bytes (expected a positive number)")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        return _error(
            f"tts error: invalid timeout (expected a positive number of seconds, "
            f"got {timeout!r}) -- fix timeout in workspace/tools/tts.toml"
        )

    # Unconfigured endpoint is the first steer an agent should hit.
    if not endpoint:
        return _error(
            "tts error: no endpoint configured -- set endpoint in "
            "workspace/tools/tts.toml"
        )
    if not text.strip():
        return _error("tts error: text is empty")
    if len(text) > max_input_chars:
        return _error(
            f"tts error: text too long ({len(text)} chars > "
            f"max_input_chars={max_input_chars}) -- shorten the text or raise "
            "max_input_chars in workspace/tools/tts.toml"
        )

    # Optionals may arrive as explicit params OR as tool_config keys; the
    # explicit param wins, then config, then the built-in default.
    cfg_send_to_room = tc.get("send_to_room", False)
    if send_to_room is not None and not isinstance(send_to_room, bool):
        return _error(f"tts error: invalid send_to_room (expected bool, got {type(send_to_room).__name__})")
    if not isinstance(cfg_send_to_room, bool):
        return _error(f"tts error: invalid send_to_room (expected bool, got {type(cfg_send_to_room).__name__})")
    # The dispatch passes input.get("send_to_room", False), so a False param
    # is the "not supplied" value -- only an explicit True overrides config.
    resolved_send_to_room = send_to_room if send_to_room is True else cfg_send_to_room

    resolved_caption = caption if caption is not None else tc.get("caption")
    if resolved_caption is not None and not isinstance(resolved_caption, str):
        return _error(f"tts error: invalid caption (expected str, got {type(resolved_caption).__name__})")

    resolved_voice = voice if voice is not None else cfg_voice
    if not isinstance(resolved_voice, str):
        return _error(f"tts error: invalid voice (expected str, got {type(resolved_voice).__name__})")
    resolved_speed = speed if speed is not None else cfg_speed
    if isinstance(resolved_speed, bool) or not isinstance(resolved_speed, (int, float)):
        return _error(f"tts error: invalid speed (expected a number, got {type(resolved_speed).__name__})")
    if normalize_text is not None:
        resolved_normalize = normalize_text
    elif "normalize_text" in tc:
        # The config also accepts "normalize_text" (a direct alias); it wins
        # over the plain "normalize" key when both are present.
        resolved_normalize = tc["normalize_text"]
    else:
        resolved_normalize = cfg_normalize
    if not isinstance(resolved_normalize, bool):
        return _error(f"tts error: invalid normalize (expected bool, got {type(resolved_normalize).__name__})")
    if isinstance(chunk_tokens, bool) or not isinstance(chunk_tokens, int):
        return _error(f"tts error: invalid chunk_tokens (expected int, got {type(chunk_tokens).__name__})")
    if not isinstance(model, str):
        model = ""
    if not isinstance(ffmpeg_path, str) or not ffmpeg_path:
        return _error(f"tts error: invalid ffmpeg_path (expected str, got {type(ffmpeg_path).__name__})")
    if not isinstance(output_dir, str) or not output_dir:
        return _error("tts error: invalid output_dir (expected a non-empty str)")
    # Containment: an absolute or `..`-bearing output_dir would silently write (and
    # mkdir) outside the documented root and report a misleading relative path.
    if os.path.isabs(output_dir) or ".." in output_dir.replace("\\", "/").split("/"):
        return _error(
            "tts error: invalid output_dir (must be workspace-relative with no '..') "
            "-- fix output_dir in workspace/tools/tts.toml"
        )

    workspace = _resolve_workspace(agent_config)
    if workspace is None:
        return _error("tts error: agent_config.workspace is required to resolve the output path")

    if resolved_normalize:
        # The normalizer calls int() on digit runs; CPython caps str->int at 4300
        # digits, and `text` legitimately embeds untrusted content (logs, pasted
        # data). A raise here would escape into the agent turn.
        try:
            normalized = normalize(text)
        except Exception as e:
            return _error(
                f"tts error: cannot normalize this text ({type(e).__name__}: {e}) -- "
                "pass normalize_text=false to speak it verbatim"
            )
    else:
        normalized = text
    # chunk_tokens <= 0 means "no chunking"; otherwise floor the chunk size so a
    # typo cannot spawn thousands of requests.
    chunk_chars = 0 if chunk_tokens <= 0 else max(int(chunk_tokens) * 4, _MIN_CHUNK_CHARS)
    chunks = chunk_text(normalized, chunk_chars)
    if not chunks:
        return _error("tts error: text is empty")
    if len(chunks) > _MAX_CHUNKS:
        return _error(
            f"tts error: text would need {len(chunks)} chunks (limit {_MAX_CHUNKS}) -- "
            "raise chunk_tokens in workspace/tools/tts.toml or shorten the text"
        )

    # Fetch every chunk BEFORE writing anything: a failed first chunk must
    # not leave a partial file behind.
    audio = []
    total_bytes = 0
    try:
        async with httpx.AsyncClient(timeout=timeout, verify=_ssl_context) as client:
            for chunk in chunks:
                body = {"input": chunk, "speed": resolved_speed}
                if model:
                    body["model"] = model
                if resolved_voice:
                    body["voice"] = resolved_voice
                async with client.stream("POST", endpoint, json=body) as resp:
                    if not (200 <= resp.status_code < 300):
                        snippet = (await _read_bounded(resp, 200)).decode("utf-8", "replace")
                        return _error(
                            f"tts error: service error HTTP {resp.status_code} "
                            f"from {endpoint}: {snippet}"
                        )
                    content_type = _content_type(resp)
                    # One byte past the cap is enough to reject it -- the body is
                    # never fully buffered.
                    data = await _read_bounded(resp, max_audio_bytes + 1)
                if len(data) > max_audio_bytes:
                    return _error(
                        f"tts error: audio body too large (> max_audio_bytes="
                        f"{max_audio_bytes}) from {endpoint}"
                    )
                if len(data) == 0:
                    return _error(f"tts error: service returned 0 bytes of audio from {endpoint}")
                # Body SHAPE beats the declared type: a service that answers with an
                # error document under a 2xx status is exactly the failure this guards.
                head = data.lstrip()[:1]
                if head == b"{":
                    return _error(
                        "tts error: service returned a JSON error body instead of audio"
                    )
                if head in (b"[", b"<"):
                    snippet = data[:200].decode("utf-8", "replace")
                    return _error(
                        "tts error: service returned non-audio content instead of audio "
                        f"({snippet})"
                    )
                if content_type:
                    media_type = content_type.split(";")[0].strip().lower()
                    if media_type.startswith("text/") or media_type in (
                        "application/json", "application/xml", "text/xml",
                        "application/xhtml+xml",
                    ):
                        return _error(
                            f"tts error: service returned {media_type} instead of audio"
                        )
                total_bytes += len(data)
                if total_bytes > max_total_audio_bytes:
                    return _error(
                        f"tts error: total audio exceeds max_total_audio_bytes="
                        f"{max_total_audio_bytes} after {len(audio) + 1} chunks -- shorten "
                        "the text or raise max_total_audio_bytes in workspace/tools/tts.toml"
                    )
                audio.append(data)
    except httpx.TimeoutException:
        return _error(f"tts error: timed out contacting TTS endpoint {endpoint}")
    except (httpx.ConnectError, httpx.HTTPError, OSError):
        return _error(f"tts error: cannot reach TTS endpoint {endpoint}")

    # Output path: <workspace>/<output_dir>/<UTC stamp>-<sha256[:8]>.mp3,
    # disambiguated with a -1, -2, ... suffix so nothing is ever clobbered.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    out_dir = os.path.join(workspace, output_dir)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        return _error(f"tts error: cannot create output directory {out_dir}: {e}")
    base = f"{stamp}-{digest}"
    out_path = _reserve_output_path(out_dir, base)
    if out_path is None:
        return _error(f"tts error: cannot allocate a unique output path in {out_dir}")

    def _fail_reserved(msg):
        """Failure AFTER the output path was reserved: never leave the file behind."""
        _unlink_quietly(out_path)
        return _error(msg)

    try:
        if len(audio) == 1:
            # Single chunk: write straight into the reservation, never invoke ffmpeg.
            with open(out_path, "wb") as f:
                f.write(audio[0])
        else:
            tmp_dir = tempfile.mkdtemp(prefix="openalph-tts-")
            try:
                list_path = os.path.join(tmp_dir, "concat.txt")
                with open(list_path, "w", encoding="utf-8") as lf:
                    for i, part_data in enumerate(audio):
                        part_path = os.path.join(tmp_dir, f"chunk_{i:03d}.mp3")
                        with open(part_path, "wb") as pf:
                            pf.write(part_data)
                        lf.write(f"file '{os.path.abspath(part_path)}'\n")
                # ffmpeg writes inside tmp_dir and the result is committed with
                # os.replace, so a failed or timed-out concat cannot leave a partial
                # .mp3 at the final path.
                out_tmp = os.path.join(tmp_dir, "out.mp3")
                try:
                    proc = await asyncio.create_subprocess_exec(
                        ffmpeg_path, "-f", "concat", "-safe", "0",
                        "-i", list_path, "-c", "copy", "-y", out_tmp,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                except FileNotFoundError:
                    return _fail_reserved(
                        f"tts error: ffmpeg not found ({ffmpeg_path!r}) -- set "
                        "ffmpeg_path in workspace/tools/tts.toml"
                    )
                except OSError as e:
                    return _fail_reserved(
                        f"tts error: cannot execute ffmpeg ({ffmpeg_path!r}): {e} -- set "
                        "ffmpeg_path in workspace/tools/tts.toml to an executable ffmpeg"
                    )
                try:
                    _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    return _fail_reserved(
                        f"tts error: ffmpeg concatenation timed out after {timeout}s"
                    )
                except asyncio.CancelledError:
                    # Cancellation (turn-stall watchdog, /stop) must not orphan ffmpeg.
                    proc.kill()
                    await proc.wait()
                    raise
                if proc.returncode != 0:
                    detail = (stderr or b"").decode("utf-8", "replace")[-300:]
                    return _fail_reserved(
                        f"tts error: concatenation failed (exit {proc.returncode}): {detail}"
                    )
                os.replace(out_tmp, out_path)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
    except OSError as e:
        return _fail_reserved(f"tts error: cannot write {out_path}: {e}")

    rel_path = os.path.relpath(out_path, workspace)
    count = len(audio)
    noun = "chunk" if count == 1 else "chunks"
    content = f"wrote {rel_path} ({os.path.basename(out_path)}, {count} {noun})"

    if resolved_send_to_room:
        upload_callback = callbacks.get("send_media") if callbacks else None
        try:
            from .media import send_media

            # send_media catches a raising callback itself and reports it as
            # an is_error ToolResult, so inspect the result, not exceptions.
            send_result = await send_media(
                path=str(out_path),
                caption=resolved_caption,
                # Couple the upload cap to the real file-size bound, so raising the
                # audio caps cannot produce a "synthesized but cannot send" dead end.
                max_upload_bytes=max_total_audio_bytes,
                upload_callback=upload_callback,
            )
        except Exception as e:
            # Path stays in the content even when the room send blows up.
            return _error(f"{content} (send to room failed: {e})")
        if send_result.is_error:
            if upload_callback is None:
                # No callback: synthesis succeeded and the file is on disk, so this is
                # not an error -- but the caller asked for a room post and did not get
                # one, so say so explicitly instead of reporting a bare success.
                content = (
                    f"{content} (NOT sent to room: no room delivery available in this "
                    "context -- use send_media with this path instead)"
                )
            else:
                return _error(f"{content} (send to room failed: {send_result.content})")
        elif upload_callback is not None:
            content = f"{content}; sent to room"

    return ToolResult(content=content, is_error=False)


# --- STT transcription -------------------------------------------------------

# Pasted media tag, e.g. "[media: media/ab12/note.ogg (audio/ogg, 45KB)]" --
# possibly embedded in a sentence. Extract the first whitespace-delimited
# token after "[media:".
_MEDIA_TAG_RE = re.compile(r"\[media:\s*(\S+)")
_MAX_SEGMENT_LINES = 200


async def stt(path, language=None, prompt=None, timestamps=False,
              tool_config=None, agent_config=None, callbacks=None):
    """Transcribe an audio file with the workspace STT service.

    Config (workspace/tools/stt.toml): endpoint (required), timeout,
    language, prompt, timestamps, max_bytes. ``path`` may be workspace-relative,
    absolute, or a pasted media tag. The file is sent as a multipart POST
    with response_format "json" (or "verbose_json" when timestamps=True);
    language and prompt are forwarded only when their resolved value is
    non-empty ("off" is the service's prompt opt-out sentinel and is
    forwarded verbatim).

    Returns:
        ToolResult with the transcript (plus a language header line and a
        [segments] block when timestamps=True), or an is_error result.
        Audio bytes never appear in the content.
    """
    tc = tool_config if isinstance(tool_config, dict) else {}

    endpoint = tc.get("endpoint", "")
    timeout = tc.get("timeout", 300.0)
    cfg_language = tc.get("language", "")
    cfg_prompt = tc.get("prompt", "")
    max_bytes = tc.get("max_bytes", 104857600)
    max_response_bytes = tc.get("max_response_bytes", 10485760)
    max_transcript_chars = tc.get("max_transcript_chars", 50000)

    # Invalid param types are an error, never a raise.
    if not isinstance(path, str):
        return _error(f"stt error: invalid path (expected str, got {type(path).__name__})")
    if language is not None and not isinstance(language, str):
        return _error(f"stt error: invalid language (expected str, got {type(language).__name__})")
    if prompt is not None and not isinstance(prompt, str):
        return _error(f"stt error: invalid prompt (expected str, got {type(prompt).__name__})")
    cfg_timestamps = tc.get("timestamps", False)
    if timestamps is not None and not isinstance(timestamps, bool):
        return _error(f"stt error: invalid timestamps (expected bool, got {type(timestamps).__name__})")
    if not isinstance(cfg_timestamps, bool):
        return _error(f"stt error: invalid timestamps (expected bool, got {type(cfg_timestamps).__name__})")
    # The dispatch passes input.get("timestamps", False), so a False param is
    # the "not supplied" value -- only an explicit True overrides config.
    resolved_timestamps = timestamps if timestamps is True else cfg_timestamps
    if not isinstance(endpoint, str):
        return _error(f"stt error: invalid endpoint (expected str, got {type(endpoint).__name__})")
    if not endpoint:
        return _error(
            "stt error: no endpoint configured -- set endpoint in "
            "workspace/tools/stt.toml"
        )
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        return _error(
            f"stt error: invalid timeout (expected a positive number of seconds, "
            f"got {timeout!r}) -- fix timeout in workspace/tools/stt.toml"
        )
    if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, (int, float)) \
            or max_response_bytes <= 0:
        return _error("stt error: invalid max_response_bytes (expected a positive number)")
    if isinstance(max_transcript_chars, bool) \
            or not isinstance(max_transcript_chars, (int, float)) \
            or max_transcript_chars <= 0:
        return _error("stt error: invalid max_transcript_chars (expected a positive number)")
    if not isinstance(cfg_language, str):
        return _error(f"stt error: invalid language (expected str, got {type(cfg_language).__name__})")
    if not isinstance(cfg_prompt, str):
        return _error(f"stt error: invalid prompt (expected str, got {type(cfg_prompt).__name__})")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, (int, float)) or max_bytes <= 0:
        return _error("stt error: invalid max_bytes (expected a positive number)")

    # Resolve the path. A pasted `[media: ...]` tag is honoured only when the token it
    # yields actually resolves to a file; otherwise the literal string wins, so a
    # filename that merely CONTAINS "[media:" is never hijacked.
    raw = path.strip()
    if not raw:
        return _error("stt error: path is empty")
    workspace = _resolve_workspace(agent_config)
    tag = _MEDIA_TAG_RE.search(raw)
    candidates = []
    if tag:
        candidates.append(tag.group(1))
    if raw not in candidates:
        candidates.append(raw)
    if workspace is None and not any(os.path.isabs(c) for c in candidates):
        return _error(
            "stt error: agent_config.workspace is required to resolve a relative path"
        )

    def _resolve(candidate):
        if os.path.isabs(candidate) or workspace is None:
            return candidate
        return os.path.join(workspace, candidate)

    path_str = None
    for candidate in candidates:
        resolved = _resolve(candidate)
        if resolved and os.path.isfile(resolved):
            path_str = resolved
            break
    if path_str is None:
        # Nothing resolved -- steer on the literal the caller supplied.
        path_str = _resolve(candidates[-1])

    if not os.path.exists(path_str):
        return _error(f"stt error: file not found: {path_str}")
    if not os.path.isfile(path_str):
        return _error(f"stt error: not a file: {path_str}")
    try:
        size = os.path.getsize(path_str)
    except OSError as e:
        return _error(f"stt error: cannot stat {path_str}: {e}")
    if size == 0:
        return _error(f"stt error: empty file: {path_str}")
    if size > max_bytes:
        return _error(
            f"stt error: file too large ({size} bytes > max_bytes={max_bytes}) -- "
            "raise max_bytes in workspace/tools/stt.toml or use a smaller file"
        )

    resolved_language = language if language is not None else cfg_language
    resolved_prompt = prompt if prompt is not None else cfg_prompt

    try:
        with open(path_str, "rb") as f:
            audio_bytes = f.read()
    except OSError as e:
        return _error(f"stt error: cannot read {path_str}: {e}")

    filename = os.path.basename(path_str)
    guessed_mime, _ = mimetypes.guess_type(filename)
    if not guessed_mime:
        guessed_mime = "application/octet-stream"

    scalars = {"response_format": "verbose_json" if resolved_timestamps else "json"}
    if resolved_language:
        scalars["language"] = resolved_language
    if resolved_prompt:
        # "off" is the service's prompt opt-out sentinel -- forwarded verbatim.
        scalars["prompt"] = resolved_prompt

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=_ssl_context) as client:
            async with client.stream(
                "POST",
                endpoint,
                files={"file": (filename, audio_bytes, guessed_mime)},
                data=scalars,
            ) as resp:
                if not (200 <= resp.status_code < 300):
                    snippet = (await _read_bounded(resp, 200)).decode("utf-8", "replace")
                    return _error(
                        f"stt error: service error HTTP {resp.status_code} from "
                        f"{endpoint}: {snippet}"
                    )
                # Bounded read: the response is untrusted and must not be buffered
                # whole before the cap is applied.
                body = await _read_bounded(resp, max_response_bytes + 1)
    except httpx.TimeoutException:
        return _error(f"stt error: timed out contacting STT endpoint {endpoint}")
    except (httpx.ConnectError, httpx.HTTPError, OSError):
        return _error(f"stt error: cannot reach STT endpoint {endpoint}")

    if len(body) > max_response_bytes:
        return _error(
            f"stt error: response body too large (> max_response_bytes="
            f"{max_response_bytes}) from {endpoint}"
        )
    try:
        data = json.loads(body.decode("utf-8", "replace"))
    except Exception:
        return _error('stt error: malformed response (expected {"text": ...})')
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        return _error('stt error: malformed response (expected {"text": ...})')

    transcript = data["text"]
    if not transcript.strip():
        return ToolResult(content="stt: empty transcript", is_error=False)

    # The transcript is untrusted service output: cap it with a visible marker
    # rather than letting a runaway response reach the model's context.
    if len(transcript) > max_transcript_chars:
        removed = len(transcript) - int(max_transcript_chars)
        transcript = transcript[:int(max_transcript_chars)] + (
            f"\n[transcript truncated: {removed} chars removed -- raise "
            "max_transcript_chars in workspace/tools/stt.toml]"
        )

    if not resolved_timestamps:
        return ToolResult(content=transcript, is_error=False)

    # timestamps=True: language header line + transcript + [segments] block
    # (one "[<start>-<end>] <text>" line per segment, capped, stripped text).
    # The DETECTED language is preferred over the requested one -- auto-detection
    # is the reason to ask for verbose output in the first place.
    detected = data.get("language")
    detected = detected if isinstance(detected, str) and detected else (resolved_language or "auto")
    lines = [f"[stt: language={detected}]"]
    lines.append(transcript)
    lines.append("[segments]")
    segments = data.get("segments")
    segments = segments if isinstance(segments, list) else []
    for seg in segments[:_MAX_SEGMENT_LINES]:
        if not isinstance(seg, dict):
            continue
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        seg_text = str(seg.get("text", "")).strip()
        if len(seg_text) > _MAX_SEGMENT_TEXT_CHARS:
            seg_text = seg_text[:_MAX_SEGMENT_TEXT_CHARS] + "[...truncated]"
        lines.append(f"[{start}-{end}] {seg_text}")
    if len(segments) > _MAX_SEGMENT_LINES:
        lines.append(f"[segments truncated: showing {_MAX_SEGMENT_LINES} of {len(segments)}]")
    return ToolResult(content="\n".join(lines), is_error=False)
