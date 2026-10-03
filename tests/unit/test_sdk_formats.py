"""Unit tests for SDK output-format negotiation."""

from edge_tts_api import sdk_formats


class TestNegotiate:
    def test_riff_maps_to_raw(self):
        """The SDK asks for riff-* as raw-* and adds the header itself."""
        fmt = sdk_formats.negotiate("riff-24khz-16bit-mono-pcm")
        assert fmt.wire_format == "riff-24khz-16bit-mono-pcm"
        assert fmt.transcode_format == "raw-24khz-16bit-mono-pcm"
        assert fmt.has_client_header is True
        assert fmt.is_pcm is True
        assert fmt.is_mp3 is False

    def test_raw_stays_raw(self):
        fmt = sdk_formats.negotiate("raw-24khz-16bit-mono-pcm")
        assert fmt.transcode_format == "raw-24khz-16bit-mono-pcm"
        assert fmt.has_client_header is False

    def test_mp3_format(self):
        fmt = sdk_formats.negotiate("audio-24khz-48kbitrate-mono-mp3")
        assert fmt.transcode_format == "audio-24khz-48kbitrate-mono-mp3"
        assert fmt.is_mp3 is True
        assert fmt.is_pcm is False

    def test_all_raw_rates(self):
        for rate in (8, 16, 22, 24, 44, 48):
            fmt = sdk_formats.negotiate(f"riff-{rate}khz-16bit-mono-pcm")
            assert fmt.transcode_format == f"raw-{rate}khz-16bit-mono-pcm"
            assert fmt.has_client_header is True

    def test_44100hz_spelling(self):
        fmt = sdk_formats.negotiate("riff-44100hz-16bit-mono-pcm")
        assert fmt.transcode_format == "raw-44100hz-16bit-mono-pcm"

    def test_case_and_whitespace_tolerated(self):
        fmt = sdk_formats.negotiate("  RIFF-24KHZ-16BIT-MONO-PCM ")
        assert fmt.transcode_format == "raw-24khz-16bit-mono-pcm"

    def test_empty_format_falls_back_to_pcm(self):
        """No synthesis.context yet (or blank): headerless native PCM."""
        fmt = sdk_formats.negotiate("")
        assert fmt.is_pcm is True
        assert fmt.transcode_format == "raw-24khz-16bit-mono-pcm"

    def test_unknown_opus_falls_back_to_pcm(self):
        """webm/opus is not produced; fall back to playable PCM at 24 kHz."""
        fmt = sdk_formats.negotiate("webm-24khz-16bit-mono-opus")
        assert fmt.is_pcm is True
        assert fmt.transcode_format == "raw-24khz-16bit-mono-pcm"

    def test_ogg_16khz_opus_falls_back_to_16khz(self):
        fmt = sdk_formats.negotiate("ogg-16khz-16bit-mono-opus")
        assert fmt.transcode_format == "raw-16khz-16bit-mono-pcm"

    def test_garbage_falls_back(self):
        fmt = sdk_formats.negotiate("no-such-format-at-all")
        assert fmt.is_pcm is True
        assert fmt.transcode_format == "raw-24khz-16bit-mono-pcm"


class TestContentType:
    def test_pcm_content_type(self):
        fmt = sdk_formats.negotiate("riff-24khz-16bit-mono-pcm")
        assert sdk_formats.content_type_for(fmt) == "audio/pcm"

    def test_mp3_content_type(self):
        fmt = sdk_formats.negotiate("audio-24khz-48kbitrate-mono-mp3")
        assert sdk_formats.content_type_for(fmt) == "audio/mpeg"
