"""Integration tests for the Azure Text-to-Speech REST surface.

Covers the voices/list endpoints and the classic ``/cognitiveservices/v1``
SSML-in/audio-out endpoint.  Live synthesis tests are skipped when the
edge service is unreachable, mirroring tests/integration/test_api.py.
"""

import io
import urllib.error
import urllib.request

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from edge_tts_api import app as app_module
from edge_tts_api.app import app

SAMPLE_SSML = (
    '<speak version="1.0" xml:lang="en-US">'
    '<voice name="en-US-JennyNeural">'
    "The rainbow has seven colors.</voice></speak>"
)


def _can_reach_edge():
    try:
        urllib.request.urlopen("https://speech.platform.bing.com", timeout=5)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:  # noqa: BLE001
        return False


LIVE = pytest.mark.skipif(not _can_reach_edge(), reason="edge service unreachable")


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def _post_tts(client, ssml, output_format, content_type="application/ssml+xml"):
    return client.post(
        "/cognitiveservices/v1",
        content=ssml,
        headers={
            "Content-Type": content_type,
            "X-Microsoft-OutputFormat": output_format,
            "Ocp-Apim-Subscription-Key": "test-key-0000",
        },
    )


class TestVoicesList:
    @LIVE
    def test_voices_list_azure_shape(self, client):
        response = client.get("/cognitiveservices/voices/list")
        assert response.status_code == 200
        voices = response.json()
        assert len(voices) > 100
        jenny = next(v for v in voices if v["ShortName"] == "en-US-JennyNeural")
        assert jenny["Name"] == "Microsoft Server Speech Text to Speech Voice (en-US, JennyNeural)"
        assert jenny["Gender"] == "Female"
        assert jenny["Locale"] == "en-US"
        assert jenny["VoiceType"] == "Neural"
        assert jenny["SampleRateHertz"] == "24000"
        assert jenny["Status"] in ("GA", "Preview")
        assert "WordsPerMinute" in jenny
        assert "StyleList" in jenny

    @LIVE
    def test_voices_list_tts_alias_path(self, client):
        """Both documented paths return the same list."""
        a = client.get("/cognitiveservices/voices/list").json()
        b = client.get("/tts/cognitiveservices/voices/list").json()
        assert len(a) == len(b)
        assert {v["ShortName"] for v in a} == {v["ShortName"] for v in b}


class TestTextToSpeechRest:
    @LIVE
    def test_riff_output_is_wav(self, client):
        response = _post_tts(client, SAMPLE_SSML, "riff-24khz-16bit-mono-pcm")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/wav")
        body = response.content
        assert body[:4] == b"RIFF"
        data, rate = sf.read(io.BytesIO(body))
        assert rate == 24000
        assert len(data) > 24000  # > 1 second

    @LIVE
    def test_mp3_output(self, client):
        response = _post_tts(
            client, SAMPLE_SSML, "audio-24khz-48kbitrate-mono-mp3"
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/mpeg")
        # MP3 frame sync or ID3 tag.
        assert response.content[:3] in (b"ID3",) or (
            response.content[0] == 0xFF and response.content[1] & 0xE0 == 0xE0
        )

    @LIVE
    def test_raw_pcm_output(self, client):
        response = _post_tts(client, SAMPLE_SSML, "raw-24khz-16bit-mono-pcm")
        assert response.status_code == 200
        body = response.content
        assert body[:4] != b"RIFF"  # headerless
        assert len(body) % 2 == 0  # 16-bit samples

    @LIVE
    def test_ssml_voice_is_honored(self, client):
        ssml = SAMPLE_SSML.replace("en-US-JennyNeural", "en-US-GuyNeural")
        response = _post_tts(client, ssml, "riff-24khz-16bit-mono-pcm")
        assert response.status_code == 200
        assert response.content[:4] == b"RIFF"

    def test_wrong_content_type_is_415(self, client):
        response = _post_tts(
            client, "hello", "riff-24khz-16bit-mono-pcm", content_type="text/plain"
        )
        assert response.status_code == 415
        assert response.json()["error"]["code"] == "UnsupportedMediaType"

    def test_missing_output_format_is_400(self, client):
        response = client.post(
            "/cognitiveservices/v1",
            content=SAMPLE_SSML,
            headers={"Content-Type": "application/ssml+xml"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "BadRequest"

    def test_invalid_output_format_is_400(self, client):
        response = _post_tts(client, SAMPLE_SSML, "no-such-format")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "BadRequest"

    def test_empty_body_is_400(self, client):
        response = _post_tts(client, "", "riff-24khz-16bit-mono-pcm")
        assert response.status_code == 400

    @LIVE
    def test_invalid_voice_is_400(self, client):
        ssml = SAMPLE_SSML.replace("JennyNeural", "xx-XX-NoSuchVoiceNeural")
        response = _post_tts(client, ssml, "riff-24khz-16bit-mono-pcm")
        assert response.status_code == 400
        assert response.json()["error"]["code"]


class TestSdkRoutesRegistered:
    def test_websocket_paths_exist(self):
        """Both the endpoint= and host= SDK paths are served."""
        paths = {route.path for route in app.routes if route.path.endswith("websocket/v1")}
        assert "/tts/cognitiveservices/websocket/v1" in paths
        assert "/cognitiveservices/websocket/v1" in paths
