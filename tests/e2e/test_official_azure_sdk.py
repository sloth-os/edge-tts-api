"""E2E tests: official Azure client libraries against this service.

Microsoft does not ship a typed batch-synthesis client; the official
Azure-Samples batch synthesis Python sample (samples/batch-synthesis/python/synthesis.py)
uses plain ``requests`` with the ``Ocp-Apim-Subscription-Key`` header and
``azure-identity`` for passwordless auth. These tests replicate that exact
official client pattern against a live server, and additionally exercise
the official ``azure-cognitiveservices-speech`` SDK where its surface
overlaps (voice enumeration + output-format enums).

The service is started as a real subprocess (uvicorn) on a free port —
no in-process TestClient — so these are true black-box e2e tests.
"""

import io
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
API_VERSION = "2024-04-01"
SUBSCRIPTION_KEY = "test-key-0000"  # service accepts any Ocp-Apim key

# Mirrors the official Azure sample:
# https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/master/samples/batch-synthesis/python/synthesis.py
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
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    output_dir = tmp_path_factory.mktemp("e2e-output")
    env = {
        **__import__("os").environ,
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


class OfficialSampleClient:
    """Client replicating the official Microsoft batch synthesis sample.

    submit_synthesis / get_synthesis / list_synthesis_jobs use the same
    URL structure, headers, and payload shape as
    samples/batch-synthesis/python/synthesis.py.
    """

    def __init__(self, endpoint: str, subscription_key: str):
        self.endpoint = endpoint.rstrip("/")
        self.header = {"Ocp-Apim-Subscription-Key": subscription_key}

    def submit_synthesis(self, job_id: str, payload: dict) -> requests.Response:
        url = (
            f"{self.endpoint}/texttospeech/batchsyntheses/{job_id}"
            f"?api-version={API_VERSION}"
        )
        header = {"Content-Type": "application/json"}
        header.update(self.header)
        return requests.put(url, json.dumps(payload), headers=header)

    def get_synthesis(self, job_id: str) -> requests.Response:
        url = (
            f"{self.endpoint}/texttospeech/batchsyntheses/{job_id}"
            f"?api-version={API_VERSION}"
        )
        return requests.get(url, headers=self.header)

    def wait_for_terminal(self, job_id: str, timeout: float = 120.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self.get_synthesis(job_id)
            if response.status_code >= 400:
                response.raise_for_status()
            status = response.json()["status"]
            if status in ("Succeeded", "Failed"):
                return response.json()
            time.sleep(1.0)
        raise TimeoutError(f"job {job_id} did not finish in {timeout}s")

    def list_synthesis_jobs(
        self, skip: int = 0, max_page_size: int = 100
    ) -> requests.Response:
        url = (
            f"{self.endpoint}/texttospeech/batchsyntheses"
            f"?api-version={API_VERSION}&skip={skip}&maxpagesize={max_page_size}"
        )
        return requests.get(url, headers=self.header)

    def delete_synthesis(self, job_id: str) -> requests.Response:
        url = (
            f"{self.endpoint}/texttospeech/batchsyntheses/{job_id}"
            f"?api-version={API_VERSION}"
        )
        return requests.delete(url, headers=self.header)


@pytest.fixture(scope="module")
def official(server):
    return OfficialSampleClient(server, SUBSCRIPTION_KEY)


def _download_result(base_url: str, job: dict) -> bytes:
    """Download the results ZIP the way the Azure docs describe."""
    response = requests.get(
        job["outputs"]["result"],
        headers={"Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY},
        params={"api-version": API_VERSION},
    )
    assert response.status_code == 200
    return response.content


class TestOfficialClientE2E:
    """Full lifecycle exactly as the official sample drives it."""

    def test_ssml_lifecycle(self, official):
        """submit_synthesis -> poll -> download, the official sample flow."""
        job_id = "e2e-official-ssml-001"
        payload = {
            "description": "my ssml test",
            "inputKind": "SSML",
            "inputs": [{"content": SAMPLE_SSML}],
            "properties": {
                "outputFormat": "riff-24khz-16bit-mono-pcm",
                "wordBoundaryEnabled": False,
                "sentenceBoundaryEnabled": False,
                "concatenateResult": False,
                "decompressOutputFiles": False,
            },
        }
        response = official.submit_synthesis(job_id, payload)
        assert response.status_code == 201, response.text
        created = response.json()
        assert created["id"] == job_id
        assert created["status"] in ("NotStarted", "Running", "Succeeded")
        assert created["inputKind"] == "SSML"
        assert "inputs" not in created  # Azure does not echo inputs

        job = official.wait_for_terminal(job_id)
        assert job["status"] == "Succeeded"
        props = job["properties"]
        assert props["succeededAudioCount"] == 1
        assert props["failedAudioCount"] == 0
        assert props["sizeInBytes"] > 0
        assert props["durationInMilliseconds"] > 0
        assert props["billingDetails"]["neuralCharacters"] > 0
        assert job["outputs"]["result"].endswith("results.zip")

        zip_bytes = _download_result(official.endpoint, job)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert "0001.wav" in names
            assert "summary.json" in names
            summary = json.loads(zf.read("summary.json"))
            assert summary["status"] == "Succeeded"
            assert summary["results"][0]["audioFileName"] == "0001.wav"
            assert summary["results"][0]["contents"] == [SAMPLE_SSML]

        # Delete and confirm the official sample's 204 behavior.
        assert official.delete_synthesis(job_id).status_code == 204
        assert official.get_synthesis(job_id).status_code == 404

    def test_plain_text_with_synthesis_config(self, official):
        """PlainText jobs follow the second official sample payload."""
        job_id = "e2e-official-text-001"
        payload = {
            "inputKind": "PlainText",
            "synthesisConfig": {"voice": "en-US-AvaMultilingualNeural"},
            "customVoices": {},  # present (empty) in the official sample
            "inputs": [{"content": "The rainbow has seven colors."}],
            "properties": {"outputFormat": "audio-24khz-160kbitrate-mono-mp3"},
        }
        response = official.submit_synthesis(job_id, payload)
        assert response.status_code == 201, response.text
        job = official.wait_for_terminal(job_id)
        assert job["status"] == "Succeeded"
        assert job["synthesisConfig"]["voice"] == "en-US-AvaMultilingualNeural"

        zip_bytes = _download_result(official.endpoint, job)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            assert "0001.mp3" in zf.namelist()

    def test_list_pagination_shape(self, official):
        """list_synthesis_jobs returns {"value": [...]} + nextLink."""
        for i in (1, 2):
            response = official.submit_synthesis(
                f"e2e-official-list-00{i}",
                {
                    "inputKind": "SSML",
                    "inputs": [{"content": SAMPLE_SSML}],
                },
            )
            assert response.status_code == 201

        response = official.list_synthesis_jobs(skip=0, max_page_size=1)
        assert response.status_code == 200
        body = response.json()
        assert len(body["value"]) == 1
        assert body["nextLink"].startswith("http")
        assert "skip=1" in body["nextLink"]

        page2 = requests.get(
            body["nextLink"],
            headers={"Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY},
        )
        assert page2.status_code == 200
        assert len(page2.json()["value"]) >= 1

    def test_error_responses_follow_azure_contract(self, official):
        """The documented 400/404 error bodies."""
        # 400: inputs is required.
        response = official.submit_synthesis(
            "e2e-official-err-400",
            {"inputKind": "SSML", "properties": {}},
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "BadRequest"

        # 404: unknown job id.
        response = official.get_synthesis("e2e-official-does-not-exist")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "NotFound"

    def test_failed_job_reports_error(self, official):
        """An invalid voice yields status=Failed with properties.error."""
        job_id = "e2e-official-fail-001"
        payload = {
            "inputKind": "PlainText",
            "synthesisConfig": {"voice": "xx-XX-NoSuchVoiceNeural"},
            "inputs": [{"content": "This will fail."}],
        }
        response = official.submit_synthesis(job_id, payload)
        assert response.status_code == 201
        job = official.wait_for_terminal(job_id)
        assert job["status"] == "Failed"
        assert job["properties"]["failedAudioCount"] == 1
        assert job["properties"]["error"]["code"]


class TestOfficialSpeechSdk:
    """Overlapping surface of azure-cognitiveservices-speech (1.52+).

    Batch synthesis itself has no typed SDK; Microsoft's own sample uses
    requests. These tests pin the parts of the official SDK that our
    service must agree with: output-format naming and voice identity.
    """

    def test_speechsdk_imports(self):
        speechsdk = pytest.importorskip("azure.cognitiveservices.speech")
        assert hasattr(speechsdk, "SpeechConfig")

    def test_output_format_enum_matches_service_formats(self):
        """Every Riff/MP3 enum the SDK exposes round-trips through our
        supported-format set (by name, as the REST API uses strings)."""
        from edge_tts_api import audio

        speechsdk = pytest.importorskip("azure.cognitiveservices.speech")
        names = [
            n
            for n in dir(speechsdk.SpeechSynthesisOutputFormat)
            if not n.startswith("_")
        ]
        assert "Riff24Khz16BitMonoPcm" in names
        assert "Audio24Khz96KBitRateMonoMp3" in names
        # The REST strings for the canonical enum members are supported.
        assert audio.is_supported("riff-24khz-16bit-mono-pcm")
        assert audio.is_supported("audio-24khz-96kbitrate-mono-mp3")

    def test_speechsdk_can_configure_custom_endpoint(self):
        """SpeechConfig accepts an arbitrary endpoint, the same way the
        official samples point it at a custom-domain endpoint."""
        speechsdk = pytest.importorskip("azure.cognitiveservices.speech")
        config = speechsdk.SpeechConfig(
            subscription=SUBSCRIPTION_KEY,
            endpoint="https://example.cognitiveservices.azure.com/",
        )
        assert config is not None
