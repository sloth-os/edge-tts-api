"""Azure Speech Service text-to-speech websocket protocol.

Implements the wire protocol spoken by the official Speech SDKs (and the
Azure ``/tts/cognitiveservices/websocket/v1`` endpoint):

- Text frames:  ``"Header: value\\r\\n...\\r\\n\\r\\nbody"``
- Binary frames: 2-byte big-endian header length + headers + payload

Client flow (per the SDK's SynthesisAdapterBase):

1. ``speech.config``      — client info (ignored by the service)
2. ``synthesis.context``  — output format + metadata options
3. ``ssml``               — the SSML document to synthesize

Service replies:

- ``turn.start``      — session context
- ``response``        — audio stream description (type + streamId)
- ``audio`` (binary)  — raw audio chunks in the requested format
- ``audio.metadata``  — word/sentence boundary + SessionEnd metadata
- ``turn.end``        — marks the turn complete

Message parsing/serialization mirrors the SDK's WebsocketMessageFormatter.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

# Azure Speech Service websocket message paths.
PATH_SPEECH_CONFIG = "speech.config"
PATH_SYNTHESIS_CONTEXT = "synthesis.context"
PATH_SSML = "ssml"
PATH_TURN_START = "turn.start"
PATH_TURN_END = "turn.end"
PATH_RESPONSE = "response"
PATH_AUDIO = "audio"
PATH_AUDIO_METADATA = "audio.metadata"
PATH_TELEMETRY = "telemetry"
PATH_SYNTHESIS_CONTROL = "synthesis.control"


@dataclass
class SpeechMessage:
    """One decoded websocket message (headers + body)."""

    path: str
    request_id: Optional[str] = None
    content_type: Optional[str] = None
    text: Optional[str] = None  # body of a text message
    data: Optional[bytes] = None  # body of a binary message

    @property
    def headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.request_id:
            headers["X-RequestId"] = self.request_id
        if self.content_type:
            headers["Content-Type"] = self.content_type
        headers["Path"] = self.path
        return headers


def _parse_headers(header_text: str) -> Dict[str, str]:
    """Parse ``Key:Value`` lines (the wire format has no space after colon)."""
    headers: Dict[str, str] = {}
    for line in header_text.split("\r\n"):
        if not line:
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        headers[key.strip().lower()] = value.strip()
    return headers


def parse_text_message(payload: Union[str, bytes]) -> SpeechMessage:
    """Decode a text websocket frame into a SpeechMessage."""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    header_text, sep, body = payload.partition("\r\n\r\n")
    headers = _parse_headers(header_text)
    return SpeechMessage(
        path=headers.get("path", ""),
        request_id=headers.get("x-requestid"),
        content_type=headers.get("content-type"),
        text=body if sep else "",
    )


def parse_binary_message(payload: bytes) -> SpeechMessage:
    """Decode a binary websocket frame into a SpeechMessage.

    Layout: 2-byte big-endian header length, headers, then the payload.
    """
    if len(payload) < 2:
        raise ValueError("binary message too short to contain a header length")
    header_length = int.from_bytes(payload[:2], "big")
    if len(payload) < 2 + header_length:
        raise ValueError("binary message shorter than its declared header")
    header_text = payload[2 : 2 + header_length].decode("utf-8", errors="replace")
    headers = _parse_headers(header_text)
    data = payload[2 + header_length :]
    return SpeechMessage(
        path=headers.get("path", ""),
        request_id=headers.get("x-requestid"),
        content_type=headers.get("content-type"),
        data=data,
    )


def _serialize_headers(
    path: str,
    request_id: Optional[str],
    content_type: Optional[str],
) -> str:
    parts: List[str] = []
    if request_id:
        parts.append(f"X-RequestId:{request_id}\r\n")
    parts.append(f"Path:{path}\r\n")
    if content_type:
        parts.append(f"Content-Type:{content_type}\r\n")
    return "".join(parts)


def make_text_message(
    path: str,
    request_id: Optional[str] = None,
    content_type: Optional[str] = None,
    body: str = "",
) -> str:
    """Serialize a text websocket frame."""
    return (
        _serialize_headers(path, request_id, content_type) + "\r\n" + body
    )


def make_binary_message(
    path: str,
    request_id: Optional[str] = None,
    content_type: Optional[str] = None,
    body: bytes = b"",
) -> bytes:
    """Serialize a binary websocket frame."""
    headers = _serialize_headers(path, request_id, content_type).encode("utf-8")
    return len(headers).to_bytes(2, "big") + headers + body


@dataclass
class SynthesisContext:
    """Parsed ``synthesis.context`` payload."""

    output_format: str = "audio-24khz-48kbitrate-mono-mp3"
    word_boundary_enabled: bool = False
    sentence_boundary_enabled: bool = False
    punctuation_boundary_enabled: bool = False
    bookmark_enabled: bool = False
    viseme_enabled: bool = False
    session_end_enabled: bool = True
    language: str = ""
    raw: Dict = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: str) -> "SynthesisContext":
        try:
            doc = json.loads(payload)
        except ValueError:
            doc = {}
        synthesis = doc.get("synthesis", {}) or {}
        audio = synthesis.get("audio", {}) or {}
        fmt = audio.get("outputFormat", "") or "audio-24khz-48kbitrate-mono-mp3"
        opts = audio.get("metadataOptions", {}) or {}
        language = synthesis.get("language", {}) or {}
        return cls(
            output_format=fmt,
            word_boundary_enabled=bool(opts.get("wordBoundaryEnabled", False)),
            sentence_boundary_enabled=bool(opts.get("sentenceBoundaryEnabled", False)),
            punctuation_boundary_enabled=bool(
                opts.get("punctuationBoundaryEnabled", False)
            ),
            bookmark_enabled=bool(opts.get("bookmarkEnabled", False)),
            viseme_enabled=bool(opts.get("visemeEnabled", False)),
            session_end_enabled=bool(opts.get("sessionEndEnabled", True)),
            language=language.get("locale", "") or "",
            raw=doc,
        )


def boundary_metadata(
    boundary_type: str,
    text: str,
    offset_ticks: int,
    duration_ticks: int,
) -> Dict:
    """Build one audio.metadata entry for a word/sentence boundary.

    ``boundary_type`` is "WordBoundary" or "SentenceBoundary"; the SDK maps
    these onto SpeechSynthesisBoundaryType.Word / Sentence.
    """
    boundary = (
        "Sentence" if boundary_type == "SentenceBoundary" else "Word"
    )
    return {
        "Type": boundary_type,
        "Data": {
            "Offset": int(offset_ticks),
            "Duration": int(duration_ticks),
            "text": {"Text": text, "Length": len(text), "BoundaryType": boundary},
        },
    }


def session_end_metadata(total_ticks: int) -> Dict:
    """Build the SessionEnd metadata entry (audio duration in ticks)."""
    return {
        "Type": "SessionEnd",
        "Data": {"Offset": int(total_ticks)},
    }


def turn_start_body(service_tag: str) -> str:
    return json.dumps({"context": {"serviceTag": service_tag}})


def response_body(stream_id: str, content_type: str) -> str:
    return json.dumps(
        {
            "audio": {"type": f"no,{content_type}", "streamId": stream_id},
            "context": {"serviceTag": "edge-tts-api"},
        }
    )
