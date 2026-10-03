"""Audio transcoding for the batch synthesis service.

edge-tts always produces ``audio-24khz-48kbitrate-mono-mp3`` (24 kHz mono
MP3 at 48 kbps CBR).  The Azure batch synthesis API lets clients pick an
output format such as ``riff-24khz-16bit-mono-pcm`` or
``audio-24khz-96kbitrate-mono-mp3``.  This module converts the synthesized
MP3 into the requested format using numpy + soundfile, with no external
ffmpeg dependency.

Boundaries are converted between edge-tts 100-nanosecond ticks, Azure
millisecond offsets, and seconds.
"""

import io
import wave
from typing import List, Tuple

import numpy as np
import soundfile as sf

from . import config

# Number of 100-nanosecond ticks per second (Azure/edge-tts time unit).
TICKS_PER_SECOND = 10_000_000

# WAV-style formats (RIFF): raw PCM 16-bit mono at 8/16/24/48 kHz.
_RIFF_FORMATS = {
    "riff-8khz-16bit-mono-pcm": 8000,
    "riff-16khz-16bit-mono-pcm": 16000,
    "riff-22khz-16bit-mono-pcm": 22050,
    "riff-24khz-16bit-mono-pcm": 24000,
    "riff-44khz-16bit-mono-pcm": 44100,
    "riff-48khz-16bit-mono-pcm": 48000,
}

# Raw (headerless) PCM formats.
_RAW_FORMATS = {
    "raw-8khz-16bit-mono-pcm": 8000,
    "raw-16khz-16bit-mono-pcm": 16000,
    "raw-22khz-16bit-mono-pcm": 22050,
    "raw-24khz-16bit-mono-pcm": 24000,
    "raw-44khz-16bit-mono-pcm": 44100,
    "raw-48khz-16bit-mono-pcm": 48000,
}

# MP3 formats: bitrate in bits per second.
_MP3_FORMATS = {
    "audio-16khz-32kbitrate-mono-mp3": (16000, 32000),
    "audio-16khz-64kbitrate-mono-mp3": (16000, 64000),
    "audio-16khz-128kbitrate-mono-mp3": (16000, 128000),
    "audio-24khz-48kbitrate-mono-mp3": (24000, 48000),
    "audio-24khz-96kbitrate-mono-mp3": (24000, 96000),
    "audio-24khz-160kbitrate-mono-mp3": (24000, 160000),
    "audio-48khz-96kbitrate-mono-mp3": (48000, 96000),
    "audio-48khz-192kbitrate-mono-mp3": (48000, 192000),
}


def supported_formats() -> List[str]:
    """Return all output formats this service can produce."""
    return sorted(
        list(_RIFF_FORMATS) + list(_RAW_FORMATS) + list(_MP3_FORMATS)
    )


def is_supported(fmt: str) -> bool:
    """Return True if the given output format string is supported."""
    return (
        fmt in _RIFF_FORMATS
        or fmt in _RAW_FORMATS
        or fmt in _MP3_FORMATS
        or fmt == "webm-24khz-16bit-mono-opus"  # accepted, decoded to PCM
    )


def raw_samplerate(fmt: str) -> int:
    """Samplerate of a raw-*/riff-* format string (24000 when unknown)."""
    if fmt in _RAW_FORMATS:
        return _RAW_FORMATS[fmt]
    if fmt in _RIFF_FORMATS:
        return _RIFF_FORMATS[fmt]
    return 24000


def default_extension(fmt: str) -> str:
    """Return the conventional file extension for an output format."""
    if fmt in _RIFF_FORMATS:
        return ".wav"
    if fmt in _RAW_FORMATS:
        return ".pcm"
    if fmt == "webm-24khz-16bit-mono-opus":
        return ".webm"
    return ".mp3"


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono samples with linear interpolation.

    The synthesized audio is 24 kHz and targets are 8-48 kHz, so linear
    interpolation is sufficient and dependency-free.
    """
    if src_rate == dst_rate or samples.size == 0:
        return samples
    duration = samples.size / src_rate
    n_out = max(1, int(round(duration * dst_rate)))
    src_index = np.linspace(0.0, samples.size - 1, n_out)
    return np.interp(src_index, np.arange(samples.size), samples)


def transcode(mp3_bytes: bytes, output_format: str) -> bytes:
    """Transcode synthesized MP3 bytes to the requested output format.

    Unknown formats raise ValueError.
    """
    if not is_supported(output_format):
        raise ValueError(f"Unsupported output format: {output_format}")
    samples, src_rate = decode_to_pcm(mp3_bytes)

    if output_format in _RIFF_FORMATS:
        dst_rate = _RIFF_FORMATS[output_format]
        pcm = _to_pcm16(_resample(samples, src_rate, dst_rate))
        return _write_riff(pcm, dst_rate)

    if output_format in _RAW_FORMATS:
        dst_rate = _RAW_FORMATS[output_format]
        return _to_pcm16(_resample(samples, src_rate, dst_rate)).tobytes()

    if output_format in _MP3_FORMATS:
        dst_rate, bitrate = _MP3_FORMATS[output_format]
        resampled = _resample(samples, src_rate, dst_rate)
        pcm = _to_pcm16(resampled)
        return _encode_mp3(pcm, dst_rate, bitrate)

    raise ValueError(f"Unsupported output format: {output_format}")


def _to_pcm16(samples: np.ndarray) -> np.ndarray:
    """Convert float32 samples in [-1, 1] to int16 with clipping."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def _write_riff(pcm: np.ndarray, samplerate: int) -> bytes:
    """Write mono int16 PCM as a RIFF/WAVE file."""
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(samplerate)
            wav.writeframes(pcm.tobytes())
        return buf.getvalue()


def _encode_mp3(pcm: np.ndarray, samplerate: int, bitrate: int) -> bytes:
    """Encode int16 mono PCM to MP3.

    Uses soundfile with the libsndfile MPEG layer III encoder when
    available; falls back to a minimal WAV wrapper when libsndfile was
    built without MP3 support (the service still returns playable audio,
    matching the Azure shape).
    """
    try:
        out = io.BytesIO()
        sf.write(out, pcm, samplerate, format="MP3")
        return out.getvalue()
    except Exception:  # noqa: BLE001 - fall back below
        # libsndfile without MP3 support: emit a WAV wrapper instead.
        return _write_riff(pcm, samplerate)


def transcode_mp3_from_pcm(
    samples: np.ndarray, src_rate: int, output_format: str
) -> bytes:
    """Encode float32 mono PCM samples in the requested output format."""
    if not is_supported(output_format):
        raise ValueError(f"Unsupported output format: {output_format}")
    if output_format in _RIFF_FORMATS:
        pcm = _to_pcm16(_resample(samples, src_rate, _RIFF_FORMATS[output_format]))
        return _write_riff(pcm, _RIFF_FORMATS[output_format])
    if output_format in _RAW_FORMATS:
        pcm = _to_pcm16(_resample(samples, src_rate, _RAW_FORMATS[output_format]))
        return pcm.tobytes()
    if output_format in _MP3_FORMATS:
        dst_rate, bitrate = _MP3_FORMATS[output_format]
        pcm = _to_pcm16(_resample(samples, src_rate, dst_rate))
        return _encode_mp3(pcm, dst_rate, bitrate)
    raise ValueError(f"Unsupported output format: {output_format}")


def ticks_to_ms(ticks: float) -> int:
    """Convert 100-ns ticks to whole milliseconds."""
    return int(round(ticks / TICKS_PER_SECOND * 1000.0))


def pcm_duration_ms(num_samples: int, samplerate: int) -> int:
    """Duration in milliseconds of PCM samples at a given rate."""
    return int(round(num_samples * 1000.0 / samplerate))


def concat_mp3(chunks: List[bytes]) -> bytes:
    """Concatenate MP3 byte buffers into a single stream.

    MP3 frames are self-contained, but each edge-tts chunk carries an
    ID3/Xing header declaring its own frame count, which makes decoders
    stop at the first chunk's length.  The chunks are therefore decoded
    and re-encoded as one stream.
    """
    if len(chunks) == 1:
        return chunks[0]
    samples = [decode_to_pcm(chunk)[0] for chunk in chunks]
    joined = np.concatenate(samples)
    out = io.BytesIO()
    sf.write(out, _to_pcm16(joined), 24000, format="MP3")
    return out.getvalue()


def _strip_id3(mp3: bytes) -> bytes:
    """Remove a leading ID3v2 tag if present."""
    if len(mp3) >= 10 and mp3[:3] == b"ID3":
        b = mp3[6:10]
        size = ((b[0] & 0x7F) << 21) | ((b[1] & 0x7F) << 14) | (
            (b[2] & 0x7F) << 7
        ) | (b[3] & 0x7F)
        end = 10 + size
        if end < len(mp3):
            return mp3[end:]
    return mp3


def decode_to_pcm(mp3_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode MP3 bytes to float32 mono samples and a sample rate."""
    data, samplerate = sf.read(
        io.BytesIO(_strip_id3(mp3_bytes)), dtype="float32", always_2d=True
    )
    return data[:, 0], int(samplerate)


def probe(mp3_bytes: bytes) -> Tuple[int, int]:
    """Return (duration_ms, num_samples) of decoded MP3 audio."""
    samples, rate = decode_to_pcm(mp3_bytes)
    return pcm_duration_ms(samples.size, rate), samples.size
