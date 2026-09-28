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
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        return _error(f"tts error: invalid timeout (expected a number, got {type(timeout).__name__})")
    if not isinstance(model, str):
        model = ""
    if not isinstance(ffmpeg_path, str) or not ffmpeg_path:
        return _error(f"tts error: invalid ffmpeg_path (expected str, got {type(ffmpeg_path).__name__})")
    if not isinstance(output_dir, str) or not output_dir:
        return _error("tts error: invalid output_dir (expected a non-empty str)")

    workspace = _resolve_workspace(agent_config)
    if workspace is None:
        return _error("tts error: agent_config.workspace is required to resolve the output path")

    normalized = normalize(text) if resolved_normalize else text
    chunk_chars = int(chunk_tokens) * 4
    chunks = chunk_text(normalized, chunk_chars)
    if not chunks:
        return _error("tts error: text is empty")

    # Fetch every chunk BEFORE writing anything: a failed first chunk must
    # not leave a partial file behind.
    audio = []
    try:
        async with httpx.AsyncClient(timeout=timeout, verify=_ssl_context) as client:
            for chunk in chunks:
                body = {"input": chunk, "speed": resolved_speed}
                if model:
                    body["model"] = model
                if resolved_voice:
                    body["voice"] = resolved_voice
                resp = await client.post(endpoint, json=body)
                if not (200 <= resp.status_code < 300):
                    snippet = resp.content[:200].decode("utf-8", "replace")
                    return _error(
                        f"tts error: service error HTTP {resp.status_code} "
                        f"from {endpoint}: {snippet}"
                    )
                data = resp.content
                if data.strip().startswith(b"{"):
                    return _error(
                        "tts error: service returned a JSON error body instead of audio"
                    )
                if len(data) == 0:
                    return _error(f"tts error: service returned 0 bytes of audio from {endpoint}")
                if len(data) > max_audio_bytes:
                    return _error(
                        f"tts error: audio body too large ({len(data)} bytes > "
                        f"max_audio_bytes={max_audio_bytes})"
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
    out_path = os.path.join(out_dir, base + ".mp3")
    suffix = 1
    while os.path.exists(out_path):
        out_path = os.path.join(out_dir, f"{base}-{suffix}.mp3")
        suffix += 1

    try:
        if len(audio) == 1:
            # Single chunk: write straight, never invoke ffmpeg.
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
                try:
                    proc = await asyncio.create_subprocess_exec(
                        ffmpeg_path, "-f", "concat", "-safe", "0",
                        "-i", list_path, "-c", "copy", "-y", out_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                except FileNotFoundError:
                    return _error(
                        f"tts error: ffmpeg not found ({ffmpeg_path!r}) -- set "
                        "ffmpeg_path in workspace/tools/tts.toml"
                    )
                try:
                    _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    return _error(
                        f"tts error: ffmpeg concatenation timed out after {timeout}s"
                    )
                if proc.returncode != 0:
                    detail = (stderr or b"").decode("utf-8", "replace")[-300:]
                    return _error(
                        f"tts error: concatenation failed (exit {proc.returncode}): {detail}"
                    )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
    except OSError as e:
        return _error(f"tts error: cannot write {out_path}: {e}")

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
                upload_callback=upload_callback,
            )
        except Exception as e:
            # Path stays in the content even when the room send blows up.
            return _error(f"{content} (send to room failed: {e})")
        if send_result.is_error:
            if upload_callback is None:
                # No callback: the room send is a no-op by design; synthesis
                # succeeded and the file is on disk, so this is not an error.
                pass
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
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        return _error(f"stt error: invalid timeout (expected a number, got {type(timeout).__name__})")
    if not isinstance(cfg_language, str):
        return _error(f"stt error: invalid language (expected str, got {type(cfg_language).__name__})")
    if not isinstance(cfg_prompt, str):
        return _error(f"stt error: invalid prompt (expected str, got {type(cfg_prompt).__name__})")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, (int, float)) or max_bytes <= 0:
        return _error("stt error: invalid max_bytes (expected a positive number)")

    # Resolve the path: media tag first, then absolute / workspace-relative.
    tag = _MEDIA_TAG_RE.search(path)
    path_str = tag.group(1) if tag else path.strip()
    if not path_str:
        return _error("stt error: path is empty")
    workspace = _resolve_workspace(agent_config)
    if not os.path.isabs(path_str):
        if workspace is None:
            return _error("stt error: agent_config.workspace is required to resolve a relative path")
        path_str = os.path.join(workspace, path_str)

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
            resp = await client.post(
                endpoint,
                files={"file": (filename, audio_bytes, guessed_mime)},
                data=scalars,
            )
    except httpx.TimeoutException:
        return _error(f"stt error: timed out contacting STT endpoint {endpoint}")
    except (httpx.ConnectError, httpx.HTTPError, OSError):
        return _error(f"stt error: cannot reach STT endpoint {endpoint}")

    if not (200 <= resp.status_code < 300):
        snippet = resp.content[:200].decode("utf-8", "replace")
        return _error(
            f"stt error: service error HTTP {resp.status_code} from {endpoint}: {snippet}"
        )
    try:
        data = resp.json()
    except Exception:
        return _error('stt error: malformed response (expected {"text": ...})')
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        return _error('stt error: malformed response (expected {"text": ...})')

    transcript = data["text"]
    if not transcript.strip():
        return ToolResult(content="stt: empty transcript", is_error=False)

    if not resolved_timestamps:
        return ToolResult(content=transcript, is_error=False)

    # timestamps=True: language header line + transcript + [segments] block
    # (one "[<start>-<end>] <text>" line per segment, capped, stripped text).
    lines = [f"[stt: language={resolved_language or 'auto'}]"]
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
        lines.append(f"[{start}-{end}] {seg_text}")
    if len(segments) > _MAX_SEGMENT_LINES:
        lines.append(f"[segments truncated: showing {_MAX_SEGMENT_LINES} of {len(segments)}]")
    return ToolResult(content="\n".join(lines), is_error=False)
