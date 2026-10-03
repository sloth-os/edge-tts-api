"""E2E tests: the official azure-cognitiveservices-speech SDK against this service.

These drive the real SDK (C++ core, Python wrapper) end-to-end over its
native websocket synthesis protocol:

- ``SpeechConfig(host=...)``   -> ws://host/cognitiveservices/websocket/v1
- ``SpeechConfig(endpoint=...)`` -> the explicit /tts/... path
- speak_text / speak_ssml, RIFF + MP3 output formats, word boundaries,
  voice enumeration, and the failure surface for unknown voices.

The service is started as a real subprocess (uvicorn) on a free port so the
SDK's TCP/websocket stack is exercised exactly as in production.
"""

import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]

speechsdk = pytest.importorskip("azure.cognitiveservices.speech")

SAMPLE_TEXT = "The rainbow has seven colors."
SAMPLE_SSML = (
    '<speak version="1.0" xml:lang="en-US">'
    '<voice name="en-US-JennyNeural">'
    "The rainbow has seven colors.</voice></speak>"
)


def _server_ready(base_url: str, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.3)
    return False


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Launch the service as a subprocess on a free port."""
    import os
    import socket
    import subprocess
    import sys

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    output_dir = tmp_path_factory.mktemp("sdk-e2e-output")
    env = {
        **os.environ,
        "EDGE_TTS_API_OUTPUT_DIR": str(output_dir),
        "EDGE_TTS_API_CONCURRENCY": "2",
    }
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "edge_tts_api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        if not _server_ready(base_url):
            out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            raise RuntimeError(f"server did not start:\n{out}")
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _synth(server, speech_config):
    return speechsdk.SpeechSynthesizer(speech_config=speech_config)


class TestSdkWebsocket:
    def test_host_form_speak_text_riff(self, server):
        """SpeechConfig(host=...) speaks text and returns RIFF audio."""
        cfg = speechsdk.SpeechConfig(
            host=f"ws://{server.removeprefix('http://')}", subscription="dummy"
        )
        cfg.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )
        result = _synth(server, cfg).speak_text(SAMPLE_TEXT)
        assert result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted
        assert result.audio_data[:4] == b"RIFF"
        assert len(result.audio_data) > 24000 * 2  # > 1 s of 16-bit 24 kHz

    def test_endpoint_form_speak_text_riff(self, server):
        """SpeechConfig(endpoint=...) with the explicit /tts/... path."""
        cfg = speechsdk.SpeechConfig(
            endpoint=f"ws://{server.removeprefix('http://')}"
            "/tts/cognitiveservices/websocket/v1",
            subscription="dummy",
        )
        cfg.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )
        result = _synth(server, cfg).speak_text(SAMPLE_TEXT)
        assert result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted
        assert result.audio_data[:4] == b"RIFF"

    def test_speak_ssml_with_prosody(self, server):
        cfg = speechsdk.SpeechConfig(
            host=f"ws://{server.removeprefix('http://')}", subscription="dummy"
        )
        ssml = (
            '<speak version="1.0" xml:lang="en-US">'
            '<voice name="en-US-JennyNeural">'
            '<prosody rate="+20%" pitch="+10Hz">Quick prosody test.</prosody>'
            "</voice></speak>"
        )
        result = _synth(server, cfg).speak_ssml(ssml)
        assert result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted
        assert result.audio_data[:4] == b"RIFF"

    def test_mp3_output_format(self, server):
        cfg = speechsdk.SpeechConfig(
            host=f"ws://{server.removeprefix('http://')}", subscription="dummy"
        )
        cfg.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3
        )
        result = _synth(server, cfg).speak_text(SAMPLE_TEXT)
        assert result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted
        # MP3: ID3 tag or frame sync.
        head = result.audio_data[:2]
        assert result.audio_data[:3] == b"ID3" or head[0] == 0xFF and head[1] & 0xE0

    def test_word_boundaries_and_duration(self, server):
        cfg = speechsdk.SpeechConfig(
            host=f"ws://{server.removeprefix('http://')}", subscription="dummy"
        )
        cfg.set_property(
            speechsdk.PropertyId.SpeechServiceResponse_RequestWordBoundary,
            "true",
        )
        synthesizer = _synth(server, cfg)

        boundaries: list = []
        synthesizer.synthesis_word_boundary.connect(
            lambda args: boundaries.append(args)
        )
        completed: list = []
        synthesizer.synthesis_completed.connect(lambda args: completed.append(args))

        result = synthesizer.speak_text(SAMPLE_TEXT)
        assert result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted
        assert boundaries, "no word boundary events fired"
        texts = [b.text for b in boundaries]
        assert "The" in texts and "rainbow" in texts
        for b in boundaries:
            assert b.boundary_type == speechsdk.SpeechSynthesisBoundaryType.Word
            assert b.audio_offset >= 0
            assert b.duration.total_seconds() >= 0
        assert completed

    def test_sequential_syntheses_on_one_config(self, server):
        cfg = speechsdk.SpeechConfig(
            host=f"ws://{server.removeprefix('http://')}", subscription="dummy"
        )
        synthesizer = _synth(server, cfg)
        for text in ("One sentence.", "Two sentences are here."):
            result = synthesizer.speak_text(text)
            assert result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted
            assert result.audio_data[:4] == b"RIFF"

    def test_unknown_voice_is_canceled_service_error(self, server):
        cfg = speechsdk.SpeechConfig(
            host=f"ws://{server.removeprefix('http://')}", subscription="dummy"
        )
        synthesizer = _synth(server, cfg)
        ssml = SAMPLE_SSML.replace(
            "en-US-JennyNeural", "xx-XX-NoSuchVoiceNeural"
        )
        result = synthesizer.speak_ssml(ssml)
        assert result.reason == speechsdk.ResultReason.Canceled
        details = result.cancellation_details
        assert details.reason == speechsdk.CancellationReason.Error
        assert (
            details.error_code == speechsdk.CancellationErrorCode.ServiceError
        )


class TestSdkVoices:
    def test_get_voices_async(self, server):
        cfg = speechsdk.SpeechConfig(
            host=f"ws://{server.removeprefix('http://')}", subscription="dummy"
        )
        synthesizer = _synth(server, cfg)
        result = synthesizer.get_voices_async().get()
        assert result.reason == speechsdk.ResultReason.VoicesListRetrieved
        assert len(result.voices) > 100
        jenny = next(
            v for v in result.voices if v.short_name == "en-US-JennyNeural"
        )
        assert jenny.locale == "en-US"
        assert jenny.gender == speechsdk.SynthesisVoiceGender.Female

    def test_get_voices_with_locale_filter(self, server):
        cfg = speechsdk.SpeechConfig(
            host=f"ws://{server.removeprefix('http://')}", subscription="dummy"
        )
        synthesizer = _synth(server, cfg)
        result = synthesizer.get_voices_async(locale="en-US").get()
        assert result.reason == speechsdk.ResultReason.VoicesListRetrieved
        assert result.voices
        assert all(v.locale == "en-US" for v in result.voices)
