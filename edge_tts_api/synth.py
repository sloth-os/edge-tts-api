"""edge-tts synthesis worker for the batch synthesis service.

Bridges the Azure batch synthesis JSON contract to the edge-tts
``Communicate`` API: SSML is passed through (with prosody overrides from
``synthesisConfig`` injected into the first voice element), PlainText is
wrapped in edge-tts's own SSML template, and boundary metadata is
collected in Azure's JSON formats (``[nnnn].word.json`` /
``[nnnn].sentence.json``).
"""

import asyncio
import re
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import edge_tts

from . import audio
from .schemas import BatchSynthesisConfig

Prosody = Dict[str, str]


class SynthesisError(Exception):
    """Raised when a single input fails to synthesize."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> Dict[str, str]:
        return {"code": self.code, "message": self.message}


# Patterns accepted by edge-tts for prosody values (see TTSConfig).
_RATE_RE = re.compile(r"^[+-]\d+%$")
_HZ_RE = re.compile(r"^[+-]\d+Hz$")

# Prosody attributes -> (Azure name, edge-tts default, validator)
_PROSODY_RULES = {
    "rate": ("rate", "+0%", _RATE_RE),
    "pitch": ("pitch", "+0Hz", _HZ_RE),
    "volume": ("volume", "+0%", _RATE_RE),
}


def normalize_prosody(value: str, name: str) -> str:
    """Normalize an Azure prosody string to the edge-tts format.

    Azure accepts both relative (``+10%``, ``-5Hz``) and absolute-ish
    values; edge-tts requires a signed value with an explicit ``+`` for
    positives.
    """
    value = str(value).strip()
    if not value:
        raise SynthesisError(
            "BadRequest", f"The {name} value '{value}' is invalid."
        )
    if value.startswith(("+", "-")):
        return value
    # Bare numbers ("10%", "5Hz") are treated as increases.
    return f"+{value}"


def _prosody_from_config(config_obj: Optional[BatchSynthesisConfig]) -> Prosody:
    """Extract validated prosody overrides from the synthesis config."""
    if config_obj is None:
        return {}
    overrides = config_obj.prosody_overrides()
    prosody: Prosody = {}
    for name, value in overrides.items():
        if value is None:
            continue
        try:
            prosody[name] = normalize_prosody(value, name)
        except SynthesisError:
            raise SynthesisError(
                "BadRequest",
                f"The synthesisConfig.{name} value '{value}' is invalid.",
            ) from None
    return prosody


VOICE_NAME_RE = re.compile(r"<voice\b[^>]*\bname\s*=\s*[\"']([^\"']+)[\"']", re.I)


def inject_prosody(ssml: str, prosody: Prosody) -> str:
    """Insert a prosody element carrying synthesisConfig overrides.

    The overrides are wrapped around the *content* of the first voice
    element.  Any existing prosody on that voice would remain nested and
    take precedence in edge-tts, which is the same precedence Azure
    applies for explicit SSML over synthesisConfig.
    """
    if not prosody:
        return ssml
    attrs = " ".join(f'{k}="{v}"' for k, v in prosody.items())
    open_tag = f"<prosody {attrs}>"
    close_tag = "</prosody>"
    return re.sub(
        r"(<voice\b[^>]*>)(.*?)(</voice>)",
        lambda m: m.group(1) + open_tag + m.group(2) + close_tag + m.group(3),
        ssml,
        count=1,
        flags=re.I | re.S,
    )


def extract_voice_from_ssml(ssml: str) -> Optional[str]:
    """Return the first voice name declared in an SSML document."""
    match = VOICE_NAME_RE.search(ssml)
    return match.group(1) if match else None


def extract_text_from_ssml(ssml: str) -> str:
    """Extract the spoken text content from an SSML document."""
    try:
        root = ET.fromstring(ssml)
    except ET.ParseError:
        # Not valid XML: treat the whole string as text (edge-tts will
        # escape it inside its own SSML template).
        return ssml
    return "".join(root.itertext())


async def synthesize_one(
    content: str,
    *,
    input_kind: str,
    voice: str,
    prosody: Prosody,
    word_boundary: bool,
    sentence_boundary: bool,
) -> Tuple[bytes, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Synthesize one input and return (mp3_bytes, word_json, sentence_json).

    Boundary entries use the Azure format:
    ``{"Text": str, "AudioOffset": int_ms, "Duration": int_ms}``.
    """
    boundary: Optional[str] = None
    if word_boundary:
        boundary = "WordBoundary"
    elif sentence_boundary:
        boundary = "SentenceBoundary"

    if input_kind == "SSML":
        ssml_text = extract_text_from_ssml(content)
        text = ssml_text if ssml_text.strip() else content
        # Prosody overrides for SSML input are injected into the document.
        if prosody:
            content = inject_prosody(content, prosody)
        effective_voice = voice or extract_voice_from_ssml(content) or ""
    else:
        text = content
        effective_voice = voice

    text = text.strip() or " "
    if not effective_voice:
        raise SynthesisError(
            "BadRequest", "The synthesisConfig.voice property is required."
        )

    kwargs: Dict[str, Any] = {}
    if prosody and input_kind != "SSML":
        kwargs.update(prosody)
    if boundary is not None:
        kwargs["boundary"] = boundary

    tts = edge_tts.Communicate(text, effective_voice, **kwargs)

    audio_chunks: List[bytes] = []
    boundaries: List[Dict[str, Any]] = []
    try:
        async for chunk in tts.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
            elif boundary is not None and chunk["type"] == boundary:
                boundaries.append(
                    {
                        "Text": chunk["text"],
                        "AudioOffset": audio.ticks_to_ms(chunk["offset"]),
                        "Duration": audio.ticks_to_ms(chunk["duration"]),
                    }
                )
    except SynthesisError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize edge-tts failures
        raise SynthesisError(
            "InternalServerError", f"edge-tts synthesis failed: {exc}"
        ) from exc

    if not audio_chunks:
        raise SynthesisError(
            "InternalServerError", "No audio was received from edge-tts."
        )

    mp3 = b"".join(audio_chunks)
    if boundary == "WordBoundary":
        return mp3, boundaries, []
    return mp3, [], boundaries


async def gather_voices() -> List[Dict[str, Any]]:
    """Return the edge-tts voice list (used for voice validation)."""
    return await edge_tts.list_voices()


def connect_id() -> str:
    """Unique id for a synthesis sub-request (debug info)."""
    return uuid.uuid4().hex
