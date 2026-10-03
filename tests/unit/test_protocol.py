"""Unit tests for the Azure Speech SDK websocket wire protocol."""

import json

import pytest

from edge_tts_api import protocol


class TestParseTextMessage:
    def test_sdk_python_header_format(self):
        """The Python SDK sends headers with no space after the colon."""
        payload = (
            "X-RequestId:abc123\r\n"
            "Path:synthesis.context\r\n"
            "Content-Type:application/json; charset=utf-8\r\n"
            "\r\n"
            '{"synthesis":{"audio":{}}}'
        )
        msg = protocol.parse_text_message(payload)
        assert msg.path == "synthesis.context"
        assert msg.request_id == "abc123"
        assert msg.content_type == "application/json; charset=utf-8"
        assert msg.text == '{"synthesis":{"audio":{}}}'

    def test_header_with_space_is_tolerated(self):
        """Other SDK variants emit ``Key: value`` with a space."""
        payload = (
            "X-RequestId: def\r\nPath: ssml\r\nContent-Type: application/ssml+xml\r\n\r\n"
            "<speak/>"
        )
        msg = protocol.parse_text_message(payload)
        assert msg.path == "ssml"
        assert msg.request_id == "def"
        assert msg.content_type == "application/ssml+xml"
        assert msg.text == "<speak/>"

    def test_header_keys_are_case_insensitive(self):
        payload = "x-requestid:rid\r\npath:speech.config\r\n\r\n"
        msg = protocol.parse_text_message(payload)
        assert msg.request_id == "rid"
        assert msg.path == "speech.config"

    def test_no_content_type(self):
        """turn.end carries no Content-Type header."""
        payload = "X-RequestId:rid\r\nPath:turn.end\r\n\r\n"
        msg = protocol.parse_text_message(payload)
        assert msg.content_type is None
        assert msg.text == ""

    def test_bytes_payload(self):
        msg = protocol.parse_text_message(b"Path:ssml\r\n\r\nhello")
        assert msg.path == "ssml"
        assert msg.text == "hello"

    def test_malformed_headers_do_not_raise(self):
        msg = protocol.parse_text_message("garbage with no headers at all")
        assert msg.path == ""
        assert msg.request_id is None


class TestMakeTextMessage:
    def test_round_trip(self):
        text = protocol.make_text_message(
            "turn.start", "rid-1", "application/json", '{"a":1}'
        )
        assert "X-RequestId:rid-1" in text
        assert "Path:turn.start" in text
        msg = protocol.parse_text_message(text)
        assert msg.path == "turn.start"
        assert msg.request_id == "rid-1"
        assert msg.content_type == "application/json"
        assert msg.text == '{"a":1}'

    def test_without_content_type(self):
        text = protocol.make_text_message("turn.end", "rid-2")
        assert "Content-Type" not in text
        msg = protocol.parse_text_message(text)
        assert msg.content_type is None


class TestBinaryMessages:
    def test_round_trip(self):
        payload = protocol.make_binary_message(
            "audio", "rid-3", "audio/pcm", b"\x01\x02\x03"
        )
        msg = protocol.parse_binary_message(payload)
        assert msg.path == "audio"
        assert msg.request_id == "rid-3"
        assert msg.content_type == "audio/pcm"
        assert msg.data == b"\x01\x02\x03"

    def test_header_length_is_big_endian(self):
        """The wire format prefixes 2-byte big-endian header length."""
        headers = b"Path:audio\r\n"
        payload = len(headers).to_bytes(2, "big") + headers + b"PAY"
        assert payload[:2] == (len(headers)).to_bytes(2, "big")
        msg = protocol.parse_binary_message(payload)
        assert msg.data == b"PAY"

    def test_empty_body(self):
        payload = protocol.make_binary_message("audio", "rid", "audio/pcm", b"")
        msg = protocol.parse_binary_message(payload)
        assert msg.data == b""

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            protocol.parse_binary_message(b"\x00")

    def test_truncated_header_raises(self):
        payload = b"\xff\xffPath:audio"
        with pytest.raises(ValueError):
            protocol.parse_binary_message(payload)


class TestSynthesisContext:
    def test_full_payload(self):
        payload = json.dumps(
            {
                "synthesis": {
                    "audio": {
                        "outputFormat": "riff-24khz-16bit-mono-pcm",
                        "metadataOptions": {
                            "wordBoundaryEnabled": True,
                            "sentenceBoundaryEnabled": False,
                            "punctuationBoundaryEnabled": True,
                            "bookmarkEnabled": False,
                            "visemeEnabled": False,
                            "sessionEndEnabled": True,
                        },
                    },
                    "language": {"autoDetection": False},
                }
            }
        )
        ctx = protocol.SynthesisContext.from_payload(payload)
        assert ctx.output_format == "riff-24khz-16bit-mono-pcm"
        assert ctx.word_boundary_enabled is True
        assert ctx.punctuation_boundary_enabled is True
        assert ctx.sentence_boundary_enabled is False
        assert ctx.session_end_enabled is True

    def test_missing_sections(self):
        ctx = protocol.SynthesisContext.from_payload("{}")
        assert ctx.output_format == "audio-24khz-48kbitrate-mono-mp3"
        assert ctx.session_end_enabled is True

    def test_invalid_json(self):
        ctx = protocol.SynthesisContext.from_payload("not json")
        assert ctx.output_format == "audio-24khz-48kbitrate-mono-mp3"


class TestMetadataBodies:
    def test_word_boundary(self):
        entry = protocol.boundary_metadata("WordBoundary", "hello", 1_000_000, 250_000)
        assert entry["Type"] == "WordBoundary"
        assert entry["Data"]["Offset"] == 1_000_000
        assert entry["Data"]["Duration"] == 250_000
        text = entry["Data"]["text"]
        assert text["Text"] == "hello"
        assert text["Length"] == 5
        assert text["BoundaryType"] == "Word"

    def test_sentence_boundary(self):
        entry = protocol.boundary_metadata(
            "SentenceBoundary", "Hello there.", 0, 1_000
        )
        assert entry["Data"]["text"]["BoundaryType"] == "Sentence"

    def test_session_end(self):
        entry = protocol.session_end_metadata(26_160_000)
        assert entry["Type"] == "SessionEnd"
        assert entry["Data"]["Offset"] == 26_160_000

    def test_turn_start_body(self):
        body = json.loads(protocol.turn_start_body("edge-tts-api"))
        assert body["context"]["serviceTag"] == "edge-tts-api"

    def test_response_body(self):
        body = json.loads(protocol.response_body("stream-1", "audio/pcm"))
        assert body["audio"]["type"] == "no,audio/pcm"
        assert body["audio"]["streamId"] == "stream-1"
