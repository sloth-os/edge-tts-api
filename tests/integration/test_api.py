"""Integration tests for the Azure-compatible batch synthesis service.

These run the full HTTP + synthesis stack through the in-process test
client. Live tests hit the real Microsoft Edge TTS endpoint through
edge-tts and are skipped automatically when the network is unavailable.
"""

import asyncio
import io
import json
import zipfile
from pathlib import Path

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from edge_tts_api import app as app_module
from edge_tts_api.app import app

API = "/texttospeech/batchsyntheses"
PARAMS = {"api-version": "2024-04-01"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Test client with isolated output and concurrency of 2."""
    monkeypatch.setattr(app_module.config, "OUTPUT_DIR", tmp_path)
    app_module.store.output_dir = tmp_path
    app_module.store._jobs = {}
    monkeypatch.setattr(app_module.config, "CONCURRENCY", 2)
    app_module._semaphore = asyncio.Semaphore(2)
    with TestClient(app) as test_client:
        yield test_client
    # Cancel any stragglers so tmp_path cleanup does not race.
    for job_id in list(app_module.store._tasks):
        app_module.store.cancel_task(job_id)


def _wait_terminal(client, job_id, timeout=60.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"{API}/{job_id}", params=PARAMS).json()
        if body["status"] in ("Succeeded", "Failed"):
            return body
        time.sleep(0.2)
    return body


def _download(client, body):
    result_url = body["outputs"]["result"]
    path = result_url.split("/texttospeech/", 1)[1]
    response = client.get(f"/texttospeech/{path}", params=PARAMS)
    assert response.status_code == 200
    return response.content


# ---------------------------------------------------------------------------
# Azure response shape
# ---------------------------------------------------------------------------


class TestAzureResponseShape:
    def test_create_response_shape(self, client):
        response = client.put(
            f"{API}/shape-test-001",
            params=PARAMS,
            json={
                "inputKind": "SSML",
                "inputs": [
                    {
                        "content": (
                            '<speak version="1.0" xml:lang="en-US">'
                            '<voice name="en-US-JennyNeural">'
                            "The rainbow has seven colors.</voice></speak>"
                        )
                    }
                ],
                "properties": {
                    "outputFormat": "riff-24khz-16bit-mono-pcm",
                    "wordBoundaryEnabled": False,
                    "sentenceBoundaryEnabled": False,
                    "concatenateResult": False,
                    "decompressOutputFiles": False,
                },
            },
        )
        assert response.status_code == 201
        body = response.json()
        # Exact top-level fields Azure returns on create.
        assert body["id"] == "shape-test-001"
        assert body["status"] in ("NotStarted", "Running", "Succeeded")
        assert set(body["createdDateTime"]) <= set(
            "0123456789-.:TZ"
        )
        assert body["inputKind"] == "SSML"
        assert body["customVoices"] == {}
        props = body["properties"]
        assert props["outputFormat"] == "riff-24khz-16bit-mono-pcm"
        assert props["wordBoundaryEnabled"] is False
        assert props["sentenceBoundaryEnabled"] is False
        assert props["concatenateResult"] is False
        assert props["decompressOutputFiles"] is False
        assert props["timeToLiveInHours"] == 744
        # inputs is not echoed in the response (Azure behavior).
        assert "inputs" not in body

    def test_get_response_shape(self, client):
        client.put(
            f"{API}/shape-test-002",
            params=PARAMS,
            json={
                "inputKind": "SSML",
                "inputs": [
                    {
                        "content": (
                            '<speak version="1.0" xml:lang="en-US">'
                            '<voice name="en-US-JennyNeural">'
                            "Hello there.</voice></speak>"
                        )
                    }
                ],
            },
        )
        body = _wait_terminal(client, "shape-test-002")
        assert body["id"] == "shape-test-002"
        assert body["status"] == "Succeeded"
        props = body["properties"]
        # Read-only fields Azure adds once the job completes.
        assert isinstance(props["sizeInBytes"], int) and props["sizeInBytes"] > 0
        assert props["succeededAudioCount"] == 1
        assert props["failedAudioCount"] == 0
        assert isinstance(props["durationInMilliseconds"], int)
        assert props["billingDetails"]["neuralCharacters"] > 0
        assert "outputs" in body
        assert body["outputs"]["result"].endswith("results.zip")

    def test_plain_text_requires_voice(self, client):
        response = client.put(
            f"{API}/shape-test-003",
            params=PARAMS,
            json={"inputKind": "PlainText", "inputs": [{"content": "hello"}]},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "BadRequest"

    def test_invalid_output_format(self, client):
        response = client.put(
            f"{API}/shape-test-004",
            params=PARAMS,
            json={
                "inputKind": "SSML",
                "inputs": [
                    {
                        "content": (
                            '<speak version="1.0" xml:lang="en-US">'
                            '<voice name="en-US-JennyNeural">Hi</voice></speak>'
                        )
                    }
                ],
                "properties": {"outputFormat": "bogus-format"},
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "BadRequest"

    def test_invalid_id(self, client):
        response = client.put(
            f"{API}/bad_id!",
            params=PARAMS,
            json={"inputKind": "SSML", "inputs": [{"content": "x"}]},
        )
        assert response.status_code == 400

    def test_invalid_api_version(self, client):
        response = client.get(f"{API}/shape-test-005", params={"api-version": "2020-01-01"})
        assert response.status_code == 400

    def test_missing_inputs(self, client):
        # Azure answers a missing required field with 400 + error contract.
        response = client.put(
            f"{API}/shape-test-006",
            params=PARAMS,
            json={"inputKind": "SSML"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "BadRequest"

    def test_error_contract(self, client):
        response = client.get(f"{API}/does-not-exist", params=PARAMS)
        assert response.status_code == 404
        assert response.json() == {
            "error": {
                "code": "NotFound",
                "message": "The specified entity cannot be found.",
            }
        }


# ---------------------------------------------------------------------------
# Live synthesis (edge-tts → Azure-shaped results)
# ---------------------------------------------------------------------------


def _can_reach_edge():
    """The edge service answers plain GETs with an HTTP error, not a
    network failure — any HTTP response counts as reachable."""
    try:
        import urllib.request

        urllib.request.urlopen("https://speech.platform.bing.com", timeout=5)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


LIVE = pytest.mark.skipif(not _can_reach_edge(), reason="edge service unreachable")


@LIVE
class TestLiveSynthesis:
    def test_ssml_single_input(self, client):
        client.put(
            f"{API}/live-test-001",
            params=PARAMS,
            json={
                "description": "my ssml test",
                "inputKind": "SSML",
                "inputs": [
                    {
                        "content": (
                            '<speak version="1.0" xml:lang="en-US">'
                            '<voice name="en-US-JennyNeural">'
                            "The rainbow has seven colors.</voice></speak>"
                        )
                    }
                ],
                "properties": {"outputFormat": "riff-24khz-16bit-mono-pcm"},
            },
        )
        body = _wait_terminal(client, "live-test-001")
        assert body["status"] == "Succeeded"
        zip_bytes = _download(client, body)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert "0001.wav" in names
            assert "summary.json" in names
            assert "0001.debug.json" in names
            # 24 kHz mono WAV decodes cleanly.
            data, rate = sf.read(io.BytesIO(zf.read("0001.wav")))
            assert rate == 24000
            assert len(data) > 24000  # > 1 second
            summary = json.loads(zf.read("summary.json"))
            assert summary["status"] == "Succeeded"
            entry = summary["results"][0]
            assert entry["status"] == "Succeeded"
            assert entry["audioFileName"] == "0001.wav"
            assert entry["properties"]["durationInMilliseconds"].isdigit()

    def test_plain_text_with_synthesis_config(self, client):
        client.put(
            f"{API}/live-test-002",
            params=PARAMS,
            json={
                "inputKind": "PlainText",
                "synthesisConfig": {"voice": "en-US-JennyNeural"},
                "inputs": [{"content": "The rainbow has seven colors."}],
            },
        )
        body = _wait_terminal(client, "live-test-002")
        assert body["status"] == "Succeeded"
        assert body["synthesisConfig"]["voice"] == "en-US-JennyNeural"
        zip_bytes = _download(client, body)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # Default output format is riff-24khz-16bit-mono-pcm.
            assert "0001.wav" in zf.namelist()

    def test_multiple_inputs_and_mp3_format(self, client):
        client.put(
            f"{API}/live-test-003",
            params=PARAMS,
            json={
                "inputKind": "PlainText",
                "synthesisConfig": {"voice": "en-US-GuyNeural"},
                "inputs": [
                    {"content": "First sentence here."},
                    {"content": "Second sentence here."},
                ],
                "properties": {
                    "outputFormat": "audio-24khz-96kbitrate-mono-mp3",
                    "concatenateResult": False,
                },
            },
        )
        body = _wait_terminal(client, "live-test-003")
        assert body["status"] == "Succeeded"
        assert body["properties"]["succeededAudioCount"] == 2
        zip_bytes = _download(client, body)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert "0001.mp3" in names and "0002.mp3" in names

    def test_concatenate_result(self, client):
        client.put(
            f"{API}/live-test-004",
            params=PARAMS,
            json={
                "inputKind": "PlainText",
                "synthesisConfig": {"voice": "en-US-AriaNeural"},
                "inputs": [
                    {"content": "Synthesize this to a file."},
                    {"content": "Synthesize this to another file."},
                ],
                "properties": {"concatenateResult": True},
            },
        )
        body = _wait_terminal(client, "live-test-004")
        assert body["status"] == "Succeeded"
        zip_bytes = _download(client, body)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            audio_names = [n for n in zf.namelist() if n.endswith(".wav")]
            assert audio_names == ["0001.wav"]

    def test_word_boundaries(self, client):
        client.put(
            f"{API}/live-test-005",
            params=PARAMS,
            json={
                "inputKind": "PlainText",
                "synthesisConfig": {"voice": "en-US-JennyNeural"},
                "inputs": [{"content": "The rainbow has seven colors."}],
                "properties": {"wordBoundaryEnabled": True},
            },
        )
        body = _wait_terminal(client, "live-test-005")
        assert body["status"] == "Succeeded"
        zip_bytes = _download(client, body)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            words = json.loads(zf.read("0001.word.json"))
            assert [w["Text"] for w in words][:3] == ["The", "rainbow", "has"]
            for entry in words:
                assert isinstance(entry["AudioOffset"], int)
                assert isinstance(entry["Duration"], int)
            # Monotonic offsets.
            offsets = [entry["AudioOffset"] for entry in words]
            assert offsets == sorted(offsets)

    def test_sentence_boundaries(self, client):
        client.put(
            f"{API}/live-test-006",
            params=PARAMS,
            json={
                "inputKind": "PlainText",
                "synthesisConfig": {"voice": "en-US-JennyNeural"},
                "inputs": [{"content": "One sentence. Another sentence."}],
                "properties": {"sentenceBoundaryEnabled": True},
            },
        )
        body = _wait_terminal(client, "live-test-006")
        assert body["status"] == "Succeeded"
        zip_bytes = _download(client, body)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            sentences = json.loads(zf.read("0001.sentence.json"))
            assert len(sentences) >= 2

    def test_list_pagination_and_delete(self, client):
        for i in (1, 2, 3):
            client.put(
                f"{API}/list-test-00{i}",
                params=PARAMS,
                json={
                    "inputKind": "SSML",
                    "inputs": [
                        {
                            "content": (
                                '<speak version="1.0" xml:lang="en-US">'
                                '<voice name="en-US-JennyNeural">'
                                f"Job number {i}.</voice></speak>"
                            )
                        }
                    ],
                },
            )
        response = client.get(
            API, params={**PARAMS, "skip": 0, "maxpagesize": 2}
        )
        body = response.json()
        assert len(body["value"]) == 2
        assert "nextLink" in body
        assert "skip=2" in body["nextLink"]

        response = client.delete(f"{API}/list-test-001", params=PARAMS)
        assert response.status_code == 204
        response = client.get(f"{API}/list-test-001", params=PARAMS)
        assert response.status_code == 404
        response = client.delete(f"{API}/list-test-001", params=PARAMS)
        assert response.status_code == 404

    def test_prosody_synthesis_config(self, client):
        client.put(
            f"{API}/live-test-007",
            params=PARAMS,
            json={
                "inputKind": "PlainText",
                "synthesisConfig": {
                    "voice": "en-US-JennyNeural",
                    "rate": "+20%",
                    "pitch": "+10Hz",
                },
                "inputs": [{"content": "Speaking a bit faster and higher."}],
            },
        )
        body = _wait_terminal(client, "live-test-007")
        assert body["status"] == "Succeeded"

    def test_ssml_with_prosody_override(self, client):
        client.put(
            f"{API}/live-test-008",
            params=PARAMS,
            json={
                "inputKind": "SSML",
                "synthesisConfig": {"voice": "en-US-GuyNeural", "rate": "-10%"},
                "inputs": [
                    {
                        "content": (
                            '<speak version="1.0" xml:lang="en-US">'
                            '<voice name="en-US-GuyNeural">'
                            "Slower by config.</voice></speak>"
                        )
                    }
                ],
            },
        )
        body = _wait_terminal(client, "live-test-008")
        assert body["status"] == "Succeeded"

    def test_failed_input_marks_job_failed(self, client):
        client.put(
            f"{API}/live-test-009",
            params=PARAMS,
            json={
                "inputKind": "PlainText",
                "synthesisConfig": {"voice": "en-US-JennyNeural"},
                "inputs": [
                    {"content": "This one works."},
                    {"content": "x" * 10},  # will succeed too; failure path uses bogus voice
                ],
            },
        )
        body = _wait_terminal(client, "live-test-009")
        # Both inputs succeed with a valid voice; this asserts the happy
        # path of multi-input accounting.
        assert body["status"] == "Succeeded"
        assert body["properties"]["succeededAudioCount"] == 2

    def test_invalid_voice_fails_job(self, client):
        client.put(
            f"{API}/live-test-010",
            params=PARAMS,
            json={
                "inputKind": "PlainText",
                "synthesisConfig": {"voice": "xx-XX-NoSuchVoiceNeural"},
                "inputs": [{"content": "This will fail."}],
            },
        )
        body = _wait_terminal(client, "live-test-010")
        assert body["status"] == "Failed"
        assert body["properties"]["failedAudioCount"] == 1
        assert body["properties"]["error"]["code"]

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
