"""Unit tests for audio transcoding and SSML/prosody handling."""

import io
import json

import numpy as np
import pytest
import soundfile as sf

from edge_tts_api import audio, synth


def _synth_mp3(seconds: float = 0.5) -> bytes:
    """Build a real MP3 via libsndfile (sine tone)."""
    samples = (
        np.sin(np.linspace(0.0, 440 * 2 * np.pi, int(24000 * seconds))) * 0.5
    ).astype("float32")
    pcm = (samples * 32767).astype("int16")
    buf = io.BytesIO()
    sf.write(buf, pcm, 24000, format="MP3")
    return buf.getvalue()


class TestAudio:
    def test_transcode_riff(self):
        mp3 = _synth_mp3()
        wav = audio.transcode(mp3, "riff-24khz-16bit-mono-pcm")
        assert wav[:4] == b"RIFF"
        data, rate = sf.read(io.BytesIO(wav))
        assert rate == 24000

    def test_transcode_riff_resamples(self):
        mp3 = _synth_mp3()
        for fmt, expected in [
            ("riff-16khz-16bit-mono-pcm", 16000),
            ("riff-48khz-16bit-mono-pcm", 48000),
        ]:
            data, rate = sf.read(io.BytesIO(audio.transcode(mp3, fmt)))
            assert rate == expected

    def test_transcode_raw(self):
        pcm = audio.transcode(_synth_mp3(), "raw-16khz-16bit-mono-pcm")
        assert len(pcm) % 2 == 0
        assert len(pcm) > 0

    def test_transcode_mp3(self):
        out = audio.transcode(_synth_mp3(), "audio-24khz-96kbitrate-mono-mp3")
        data, rate = sf.read(io.BytesIO(out))
        assert rate == 24000

    def test_transcode_unsupported(self):
        with pytest.raises(ValueError):
            audio.transcode(b"", "bogus")

    def test_supported_formats(self):
        formats = audio.supported_formats()
        assert "riff-24khz-16bit-mono-pcm" in formats
        assert "audio-24khz-48kbitrate-mono-mp3" in formats
        for fmt in formats:
            assert audio.is_supported(fmt)

    def test_default_extension(self):
        assert audio.default_extension("riff-24khz-16bit-mono-pcm") == ".wav"
        assert audio.default_extension("raw-8khz-16bit-mono-pcm") == ".pcm"
        assert (
            audio.default_extension("audio-24khz-48kbitrate-mono-mp3") == ".mp3"
        )

    def test_concat_mp3(self):
        one = _synth_mp3(0.25)
        two = audio.concat_mp3([one, one])
        duration_single, _ = audio.probe(one)
        duration_double, _ = audio.probe(two)
        assert duration_double == pytest.approx(duration_single * 2, rel=0.05)

    def test_ticks_to_ms(self):
        assert audio.ticks_to_ms(10_000_000) == 1000
        assert audio.ticks_to_ms(0) == 0
        assert audio.ticks_to_ms(123_456) == 12


class TestProsody:
    def test_normalize_adds_plus(self):
        assert synth.normalize_prosody("10%", "rate") == "+10%"
        assert synth.normalize_prosody("-5%", "rate") == "-5%"
        assert synth.normalize_prosody("+5Hz", "pitch") == "+5Hz"
        assert synth.normalize_prosody("2Hz", "pitch") == "+2Hz"

    def test_normalize_empty_raises(self):
        with pytest.raises(synth.SynthesisError):
            synth.normalize_prosody("", "rate")

    def test_inject_prosody(self):
        ssml = (
            '<speak version="1.0" xml:lang="en-US">'
            '<voice name="en-US-JennyNeural">Hello</voice></speak>'
        )
        out = synth.inject_prosody(ssml, {"rate": "+10%"})
        assert '<prosody rate="+10%">' in out
        assert "Hello" in out
        # voice element still intact
        assert '<voice name="en-US-JennyNeural">' in out

    def test_inject_prosody_noop(self):
        ssml = "<speak><voice name='v'>x</voice></speak>"
        assert synth.inject_prosody(ssml, {}) == ssml

    def test_extract_voice(self):
        ssml = (
            '<speak version="1.0"><voice name="en-US-GuyNeural">Hi</voice></speak>'
        )
        assert synth.extract_voice_from_ssml(ssml) == "en-US-GuyNeural"
        assert synth.extract_voice_from_ssml("<speak/>") is None

    def test_extract_text(self):
        ssml = (
            '<speak version="1.0" xml:lang="en-US">'
            '<voice name="v">The <emphasis>rainbow</emphasis>!</voice></speak>'
        )
        assert synth.extract_text_from_ssml(ssml) == "The rainbow!"

    def test_extract_text_invalid_xml(self):
        assert synth.extract_text_from_ssml("not xml <") == "not xml <"
