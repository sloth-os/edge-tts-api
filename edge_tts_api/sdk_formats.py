"""Output-format negotiation for the SDK websocket protocol.

The official Speech SDK asks for ``riff-*`` output by requesting the
matching ``raw-*`` format and prepending the RIFF header itself
(see AudioOutputFormatImpl: "for riff format, we will request raw format
and add a header in SDK side").  This module maps the wire format strings
onto the encodings this service can produce and tells the caller whether a
RIFF header must be prepended for the client.

MP3 formats map onto the transcoder's MP3 encodings.  Formats that would
require codecs this service does not carry (opus/webm/silk/g722/amr) fall
back to the closest supported PCM encoding, keeping the SDK's audio
playable instead of failing the synthesis.
"""

import re
from typing import NamedTuple, Optional

from . import audio

# The SDK may request these with or without the client-side RIFF header.
# Wire string -> (transcoder format, has_client_side_header).
_FORMAT_MAP = {
    # PCM (riff = SDK adds the header; raw = headerless)
    "riff-8khz-16bit-mono-pcm": ("raw-8khz-16bit-mono-pcm", True),
    "raw-8khz-16bit-mono-pcm": ("raw-8khz-16bit-mono-pcm", False),
    "riff-16khz-16bit-mono-pcm": ("raw-16khz-16bit-mono-pcm", True),
    "raw-16khz-16bit-mono-pcm": ("raw-16khz-16bit-mono-pcm", False),
    "riff-22khz-16bit-mono-pcm": ("raw-22khz-16bit-mono-pcm", True),
    "raw-22khz-16bit-mono-pcm": ("raw-22khz-16bit-mono-pcm", False),
    "riff-24khz-16bit-mono-pcm": ("raw-24khz-16bit-mono-pcm", True),
    "raw-24khz-16bit-mono-pcm": ("raw-24khz-16bit-mono-pcm", False),
    "riff-44khz-16bit-mono-pcm": ("raw-44khz-16bit-mono-pcm", True),
    "raw-44khz-16bit-mono-pcm": ("raw-44khz-16bit-mono-pcm", False),
    "riff-44100hz-16bit-mono-pcm": ("raw-44100hz-16bit-mono-pcm", True),
    "raw-44100hz-16bit-mono-pcm": ("raw-44100hz-16bit-mono-pcm", False),
    "riff-48khz-16bit-mono-pcm": ("raw-48khz-16bit-mono-pcm", True),
    "raw-48khz-16bit-mono-pcm": ("raw-48khz-16bit-mono-pcm", False),
    # MP3
    "audio-16khz-32kbitrate-mono-mp3": ("audio-16khz-32kbitrate-mono-mp3", False),
    "audio-16khz-64kbitrate-mono-mp3": ("audio-16khz-64kbitrate-mono-mp3", False),
    "audio-16khz-128kbitrate-mono-mp3": ("audio-16khz-128kbitrate-mono-mp3", False),
    "audio-24khz-48kbitrate-mono-mp3": ("audio-24khz-48kbitrate-mono-mp3", False),
    "audio-24khz-96kbitrate-mono-mp3": ("audio-24khz-96kbitrate-mono-mp3", False),
    "audio-24khz-160kbitrate-mono-mp3": ("audio-24khz-160kbitrate-mono-mp3", False),
    "audio-48khz-96kbitrate-mono-mp3": ("audio-48khz-96kbitrate-mono-mp3", False),
    "audio-48khz-192kbitrate-mono-mp3": ("audio-48khz-192kbitrate-mono-mp3", False),
}

# rate extractor for fallback mapping (e.g. "ogg-24khz-16bit-mono-opus" -> 24000)
_RATE_RE = re.compile(r"(\d+(?:\.\d+)?)khz", re.I)

# samplerate -> closest raw PCM wire format
_RATE_TO_RAW = {
    8000: "raw-8khz-16bit-mono-pcm",
    16000: "raw-16khz-16bit-mono-pcm",
    22050: "raw-22khz-16bit-mono-pcm",
    24000: "raw-24khz-16bit-mono-pcm",
    44100: "raw-44100hz-16bit-mono-pcm",
    48000: "raw-48khz-16bit-mono-pcm",
}


class NegotiatedFormat(NamedTuple):
    """The concrete encoding for a negotiated wire format."""

    wire_format: str  # format string to report to the client
    transcode_format: str  # format key for audio.transcode*
    has_client_header: bool  # True when the client expects headerless PCM

    @property
    def is_pcm(self) -> bool:
        return self.transcode_format.startswith("raw-")

    @property
    def is_mp3(self) -> bool:
        return self.transcode_format.startswith("audio-")


def negotiate(wire_format: str) -> NegotiatedFormat:
    """Map a client-requested wire format onto a supported encoding.

    Unknown or unsupported-but-parseable formats fall back to headerless
    24 kHz PCM (the native edge-tts rate), which every Speech SDK can play.
    """
    fmt = (wire_format or "").strip().lower()
    mapped = _FORMAT_MAP.get(fmt)
    if mapped is not None:
        return NegotiatedFormat(fmt, mapped[0], mapped[1])

    # Codec families this service cannot produce (opus/webm/silk/g722/amr/
    # siren/mulaw/alaw): fall back to PCM at the closest rate.
    rate = _closest_rate(fmt)
    raw = _RATE_TO_RAW.get(rate, "raw-24khz-16bit-mono-pcm")
    return NegotiatedFormat(raw, raw, False)


def _closest_rate(fmt: str) -> int:
    match = _RATE_RE.search(fmt)
    if not match:
        return 24000
    khz = float(match.group(1))
    hz = int(khz * 1000)
    return min(_RATE_TO_RAW, key=lambda r: (abs(r - hz), r))


def content_type_for(fmt: NegotiatedFormat) -> str:
    """The Content-Type reported on audio chunks for a negotiated format."""
    if fmt.is_mp3:
        return "audio/mpeg"
    return "audio/pcm"
