"""SDK websocket synthesis session.

One websocket connection runs one synthesis turn: parse the Azure Speech
Service message flow (speech.config -> synthesis.context -> ssml), pipe the
text through edge-tts, and stream the audio back in the negotiated format
with word/sentence boundary metadata.

The client speaks the official Speech SDK protocol, so no SDK
modifications are required: point ``SpeechConfig`` at this service with
``endpoint="ws://host:port/tts/cognitiveservices/websocket/v1"`` (or
``host="ws://host:port"``).
"""

import asyncio
import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import AsyncIterator, Dict, List, Optional, Tuple

import edge_tts
from fastapi import WebSocket, WebSocketDisconnect

from . import audio, protocol, sdk_formats
from .protocol import (
    PATH_AUDIO,
    PATH_AUDIO_METADATA,
    PATH_RESPONSE,
    PATH_SPEECH_CONFIG,
    PATH_SSML,
    PATH_SYNTHESIS_CONTEXT,
    PATH_SYNTHESIS_CONTROL,
    PATH_TELEMETRY,
    PATH_TURN_END,
    PATH_TURN_START,
)

logger = logging.getLogger("edge_tts_api.sdk")

# How much audio to send per binary frame (the SDK is happy with 4 KiB+).
_AUDIO_CHUNK_SIZE = 4096

# The audio the edge-tts service produces is always 24 kHz MP3.
_EDGE_NATIVE_RATE = 24000

_VOICE_NAME_RE = re.compile(r"<voice\b[^>]*\bname\s*=\s*[\"']([^\"']+)[\"']", re.I)


class SdkSynthesisError(Exception):
    """A synthesis failure that is reported to the client as turn error."""

    def __init__(self, message: str, close_code: int = 1011) -> None:
        super().__init__(message)
        self.close_code = close_code


def _extract_request(ssml: str) -> Tuple[str, Optional[Dict[str, str]]]:
    """Extract (text, prosody) from an SSML document.

    The Speech SDK wraps plain text in a full SSML document with a single
    <voice>. Prosody elements (rate/pitch/volume set through
    SpeechConfig or SpeechSynthesizer APIs) are honored.
    """
    try:
        root = ET.fromstring(ssml)
    except ET.ParseError:
        return ssml, None

    text = "".join(root.itertext())
    prosody: Dict[str, str] = {}

    voice = root.find(".//{http://www.w3.org/2001/10/synthesis}voice")
    if voice is None:
        # namespace-less documents
        voice = root.find(".//voice")

    def _collect(tag: str, into: Dict[str, str]) -> None:
        for el in root.iter(tag):
            for attr in ("rate", "pitch", "volume"):
                value = el.get(attr)
                if value:
                    into[attr] = value
            return  # only the first (outermost) prosody counts

    _collect("prosody", prosody)
    _collect("{http://www.w3.org/2001/10/synthesis}prosody", prosody)
    return text, (prosody or None)


def _normalize_prosody(value: str) -> str:
    """Coerce a prosody value to edge-tts's signed form."""
    value = value.strip()
    if value.startswith(("+", "-")):
        return value
    return f"+{value}"


async def _synthesize(
    ssml: str,
    voice_hint: Optional[str],
    prosody: Optional[Dict[str, str]],
    boundary: Optional[str],
) -> Tuple[bytes, List[Dict]]:
    """Run one synthesis through edge-tts. Returns (mp3, boundary_events).

    boundary_events: ``[{"type", "offset", "duration", "text"}, ...]``
    with offsets/durations in 100-ns ticks.
    """
    text, ssml_prosody = _extract_request(ssml)
    effective_prosody: Dict[str, str] = {}
    for source in (ssml_prosody, prosody):
        if source:
            for key, value in source.items():
                effective_prosody[key] = _normalize_prosody(value)

    voice = voice_hint
    if not voice:
        match = _VOICE_NAME_RE.search(ssml)
        voice = match.group(1) if match else "en-US-JennyNeural"

    text = text.strip() or " "

    kwargs: Dict[str, object] = dict(effective_prosody)
    if boundary is not None:
        kwargs["boundary"] = boundary

    tts = edge_tts.Communicate(text, voice, **kwargs)  # type: ignore[arg-type]
    chunks: List[bytes] = []
    events: List[Dict] = []
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
        elif boundary is not None and chunk["type"] == boundary:
            events.append(
                {
                    "type": chunk["type"],
                    "offset": chunk["offset"],
                    "duration": chunk["duration"],
                    "text": chunk["text"],
                }
            )
    if not chunks:
        raise SdkSynthesisError("No audio was received from edge-tts.")
    return b"".join(chunks), events


async def _transcode_stream(
    mp3: bytes, fmt: sdk_formats.NegotiatedFormat
) -> Tuple[AsyncIterator[bytes], int]:
    """Yield the synthesized audio in the negotiated format and its total ticks."""
    samples, rate = audio.decode_to_pcm(mp3)
    total_ticks = samples.size * protocol_ticks_per_second() // rate

    if fmt.is_pcm:
        pcm = audio._to_pcm16(
            audio._resample(samples, rate, _raw_rate(fmt.transcode_format))
        ).tobytes()

        async def pcm_iter() -> AsyncIterator[bytes]:
            for i in range(0, len(pcm), _AUDIO_CHUNK_SIZE):
                yield pcm[i : i + _AUDIO_CHUNK_SIZE]

        return pcm_iter(), total_ticks

    # MP3 target formats (no client-side header).
    encoded = audio.transcode_mp3_from_pcm(samples, rate, fmt.transcode_format)

    async def mp3_iter() -> AsyncIterator[bytes]:
        for i in range(0, len(encoded), _AUDIO_CHUNK_SIZE):
            yield encoded[i : i + _AUDIO_CHUNK_SIZE]

    return mp3_iter(), total_ticks


def protocol_ticks_per_second() -> int:
    return 10_000_000


def _raw_rate(fmt: str) -> int:
    """Samplerate of a raw-* transcode format string."""
    return audio.raw_samplerate(fmt)


async def handle_sdk_websocket(websocket: WebSocket) -> None:
    """Serve one Speech SDK synthesis session."""
    await websocket.accept()
    context: Optional[protocol.SynthesisContext] = None
    request_id: Optional[str] = None
    fmt = sdk_formats.negotiate("")

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return

            if "text" in message and message["text"] is not None:
                incoming = protocol.parse_text_message(message["text"])
            elif "bytes" in message and message["bytes"] is not None:
                incoming = protocol.parse_binary_message(message["bytes"])
            else:
                continue

            path = incoming.path

            if path == PATH_SPEECH_CONFIG:
                # Client info; nothing to answer.
                continue

            if path == PATH_TELEMETRY:
                # The SDK logs telemetry after each turn; acknowledge by
                # ignoring (Azure answers nothing either).
                continue

            if path == PATH_SYNTHESIS_CONTEXT:
                context = protocol.SynthesisContext.from_payload(
                    incoming.text or ""
                )
                request_id = incoming.request_id
                fmt = sdk_formats.negotiate(context.output_format)
                continue

            if path == PATH_SYNTHESIS_CONTROL:
                # {"action": "stop"} — best effort: drop the connection.
                return

            if path == PATH_SSML:
                if context is None or request_id is None:
                    raise SdkSynthesisError(
                        "ssml received before synthesis.context", close_code=1002
                    )
                await _run_turn(websocket, request_id, context, fmt, incoming.text or "")
                continue

            logger.debug("ignoring sdk message path=%s", path)

    except WebSocketDisconnect:
        return
    except SdkSynthesisError as exc:
        await _close_safely(websocket, exc.close_code, str(exc))
    except Exception:  # noqa: BLE001
        logger.exception("sdk websocket session failed")
        await _close_safely(websocket, 1011, "internal error")


async def _run_turn(
    websocket: WebSocket,
    request_id: str,
    context: protocol.SynthesisContext,
    fmt: sdk_formats.NegotiatedFormat,
    ssml: str,
) -> None:
    """Execute one synthesis turn: turn.start ... turn.end."""
    boundary: Optional[str] = None
    if context.word_boundary_enabled or context.punctuation_boundary_enabled:
        boundary = "WordBoundary"
    elif context.sentence_boundary_enabled:
        boundary = "SentenceBoundary"

    await websocket.send_text(
        protocol.make_text_message(
            PATH_TURN_START,
            request_id,
            "application/json; charset=utf-8",
            protocol.turn_start_body("edge-tts-api"),
        )
    )

    stream_id = "edge-tts-stream-0001"
    await websocket.send_text(
        protocol.make_text_message(
            PATH_RESPONSE,
            request_id,
            "application/json; charset=utf-8",
            protocol.response_body(stream_id, sdk_formats.content_type_for(fmt)),
        )
    )

    mp3, events = await _synthesize(ssml, None, None, boundary)
    audio_iter, total_ticks = await _transcode_stream(mp3, fmt)

    # Stream audio chunks.
    async for chunk in audio_iter:
        await websocket.send_bytes(
            protocol.make_binary_message(
                PATH_AUDIO,
                request_id,
                sdk_formats.content_type_for(fmt),
                chunk,
            )
        )

    # Boundary metadata (Azure sends boundaries interleaved with audio; the
    # SDK accepts them before turn.end as well).
    if events:
        entries = [
            protocol.boundary_metadata(
                event["type"], event["text"], event["offset"], event["duration"]
            )
            for event in events
        ]
        await websocket.send_text(
            protocol.make_text_message(
                PATH_AUDIO_METADATA,
                request_id,
                "application/json; charset=utf-8",
                json.dumps({"Metadata": entries}),
            )
        )

    if context.session_end_enabled:
        await websocket.send_text(
            protocol.make_text_message(
                PATH_AUDIO_METADATA,
                request_id,
                "application/json; charset=utf-8",
                json.dumps(
                    {"Metadata": [protocol.session_end_metadata(total_ticks)]}
                ),
            )
        )

    await websocket.send_text(
        protocol.make_text_message(PATH_TURN_END, request_id, None, "")
    )


async def _close_safely(websocket: WebSocket, code: int, reason: str) -> None:
    try:
        await websocket.close(code=code, reason=reason)
    except Exception:  # noqa: BLE001
        pass
