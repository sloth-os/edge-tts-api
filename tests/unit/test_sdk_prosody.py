"""Unit tests for SDK-websocket prosody normalization.

Azure SSML accepts several prosody rate forms (multiplier, bare percentage,
signed percentage, named constants) while edge-tts only accepts signed
percentages for rate/volume and signed Hz for pitch. These tests pin the
conversion table the SDK websocket path relies on.
"""

import pytest

from edge_tts_api import sdk_synth


class TestNormalizeProsody:
    def test_signed_percentage_passes_through(self):
        assert sdk_synth._normalize_prosody("+25%") == "+25%"
        assert sdk_synth._normalize_prosody("-50%") == "-50%"
        assert sdk_synth._normalize_prosody("+0%") == "+0%"

    def test_bare_percentage_gains_plus(self):
        assert sdk_synth._normalize_prosody("25%") == "+25%"
        assert sdk_synth._normalize_prosody("0%") == "+0%"

    def test_signed_decimal_percentage(self):
        # The official Speech SDK emits e.g. rate="+30.00%".
        assert sdk_synth._normalize_prosody("+30.00%") == "+30.00%"

    def test_multiplier_becomes_relative_percentage(self):
        # mm-gateway's azure adapter sends rate="1.25" (a plain multiplier,
        # one of Azure's documented rate forms).
        assert sdk_synth._normalize_prosody("1.25") == "+25%"
        assert sdk_synth._normalize_prosody("1") == "+0%"
        assert sdk_synth._normalize_prosody("0.5") == "-50%"
        assert sdk_synth._normalize_prosody("2") == "+100%"

    def test_named_rate_constants_map_to_percentages(self):
        assert sdk_synth._normalize_prosody("x-slow") == "-50%"
        assert sdk_synth._normalize_prosody("slow") == "-36%"
        assert sdk_synth._normalize_prosody("medium") == "+0%"
        assert sdk_synth._normalize_prosody("fast") == "+55%"
        assert sdk_synth._normalize_prosody("x-fast") == "+100%"

    def test_pitch_values_pass_through(self):
        assert sdk_synth._normalize_prosody("+10Hz") == "+10Hz"
        assert sdk_synth._normalize_prosody("-5Hz") == "-5Hz"

    def test_output_always_matches_edge_tts_pattern(self):
        import re

        for value in ("1.25", "0.5", "25%", "-50%", "x-slow", "fast"):
            out = sdk_synth._normalize_prosody(value)
            assert re.fullmatch(r"[+-]\d+(\.\d+)?%", out), out

    def test_extract_request_collects_prosody(self):
        ssml = (
            '<speak version="1.0" xml:lang="en-US">'
            '<voice name="en-US-JennyNeural">'
            '<prosody rate="1.25">fast multiplier</prosody>'
            "</voice></speak>"
        )
        _text, prosody = sdk_synth._extract_request(ssml)
        assert prosody is not None
        assert prosody["rate"] == "1.25"
        # The collected value normalizes to something edge-tts accepts.
        assert sdk_synth._normalize_prosody(prosody["rate"]) == "+25%"

        ssml = (
            '<speak version="1.0" xml:lang="en-US">'
            '<voice name="en-US-JennyNeural">'
            '<prosody rate="-50%">slower</prosody>'
            "</voice></speak>"
        )
        _text, prosody = sdk_synth._extract_request(ssml)
        assert prosody["rate"] == "-50%"
        assert sdk_synth._normalize_prosody(prosody["rate"]) == "-50%"
