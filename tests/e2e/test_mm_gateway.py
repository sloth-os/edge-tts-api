"""E2E tests: the real ``sloth-os/mm-gateway`` routing through this service.

mm-gateway (https://github.com/sloth-os/mm-gateway) is a provider-neutral
media gateway whose ``azure`` speech backend drives the official
``azure-cognitiveservices-speech`` SDK — the same SDK this service speaks
to over its websocket synthesis protocol. These tests prove the full chain:

    gateway REST (/v1/audio) -> Azure SDK -> ws://this-service -> edge-tts

Both containers run on a shared Docker network: this service runs from an
image of the repo under test (built by the fixture, or reused from
``EDGE_TTS_API_IMAGE`` when CI has already built one), and mm-gateway is
pulled from the registry (``ghcr.io/sloth-os/mm-gateway:latest``, or the
``sloth-os/mm-gateway:latest`` Docker Hub mirror). mm-gateway's azure
backend is configured purely through its documented env layout:

    AZURE_AUDIO_API_KEY=<any value; this service ignores it>
    AZURE_AUDIO_BASE_URL=ws://<tts-host>:<port>/tts/cognitiveservices/websocket/v1

so the tests also pin the "drop-in Azure replacement" claim end to end.

Skipped (not failed) when Docker is unavailable, so the rest of the e2e
suite still runs on machines without a daemon.
"""

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]

# The gateway image, in preference order: GHCR hosts the multi-arch build
# the mm-gateway CI publishes; docker.io mirrors it for plain `docker pull`.
GATEWAY_IMAGES = [
    "ghcr.io/sloth-os/mm-gateway:latest",
    "sloth-os/mm-gateway:latest",
]

GATEWAY_API_KEY = "e2e-gateway-key"
# This service performs no authentication; the SDK requires *some* key.
DUMMY_AZURE_KEY = "e2e-dummy-key"

NETWORK = "edge-tts-e2e"
TTS_CONTAINER = "edge-tts-api"  # doubles as the docker-network DNS name
GATEWAY_CONTAINER = "mm-e2e-gateway"


def _docker_available() -> bool:
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _run(cmd: list, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )


def _container_ready(container: str, url: str, timeout: float = 90.0) -> bool:
    """Poll a container-internal URL via `docker exec`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = _run(
            ["docker", "exec", container, "python", "-c",
             f"import urllib.request,sys;"
             f"sys.exit(0 if urllib.request.urlopen('{url}', timeout=3).status==200 else 1)"],
            timeout=60,
        )
        if probe.returncode == 0:
            return True
        time.sleep(1.0)
    return False


def _wait_http(url: str, timeout: float = 90.0, headers: dict | None = None) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=3) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    return False


def _resolve_gateway_image() -> str | None:
    """The first mm-gateway image that exists locally or on a registry.

    A local image short-circuits the registry round-trips: pulling once
    before the suite (or on a runner with the image pre-cached) makes the
    test independent of registry flakiness. Registry checks fall back to
    `docker manifest inspect`, trying each mirror in turn.
    """
    candidates = [os.environ.get("MM_GATEWAY_IMAGE")] if os.environ.get("MM_GATEWAY_IMAGE") else []
    candidates += GATEWAY_IMAGES
    for image in candidates:
        if _run(["docker", "image", "inspect", image], timeout=60).returncode == 0:
            return image
    for image in candidates:
        if _run(["docker", "manifest", "inspect", image], timeout=120).returncode == 0:
            return image
    return None


def _gateway_port() -> int:
    """Pick a free host port for the published gateway endpoint."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _cleanup() -> None:
    _run(["docker", "rm", "-f", GATEWAY_CONTAINER, TTS_CONTAINER], timeout=120)


@pytest.fixture(scope="module")
def gateway():
    """Start this service + a real mm-gateway, both in containers.

    Yields the gateway's published base URL (``http://127.0.0.1:<port>``).
    """
    if not _docker_available():
        pytest.skip("docker is not available")

    image = _resolve_gateway_image()
    if image is None:
        pytest.skip("no mm-gateway image reachable on any registry")

    _cleanup()
    try:
        # A dedicated network gives the gateway a stable DNS name for this
        # service (the container name doubles as the hostname).
        _run(["docker", "network", "rm", NETWORK])
        assert _run(["docker", "network", "create", NETWORK]).returncode == 0

        # This service, from an image of the repo under test. CI passes a
        # ready-built image via EDGE_TTS_API_IMAGE; locally the fixture
        # builds one (PIP_INDEX_URL can point pip at a reachable mirror).
        tts_image = os.environ.get("EDGE_TTS_API_IMAGE") or "edge-tts-api:e2e"
        if _run(["docker", "image", "inspect", tts_image]).returncode != 0:
            build = _run(["docker", "build", "-t", tts_image, str(ROOT)], timeout=1500)
            assert build.returncode == 0, build.stderr[-2000:]
        assert _run(
            ["docker", "run", "-d", "--name", TTS_CONTAINER,
             "--network", NETWORK, "--restart", "no",
             "-v", "edge-tts-e2e-data:/data",
             tts_image]
        ).returncode == 0

        # The real mm-gateway with its azure backend aimed at this service.
        # AZURE_AUDIO_BASE_URL is mm-gateway's documented custom-endpoint
        # knob; the SDK turns it into SpeechConfig(endpoint=...).
        port = _gateway_port()
        azure_url = f"ws://{TTS_CONTAINER}:8000/tts/cognitiveservices/websocket/v1"
        run = _run(
            ["docker", "run", "-d", "--name", GATEWAY_CONTAINER,
             "--network", NETWORK, "--restart", "no",
             "-p", f"127.0.0.1:{port}:8000",
             "-e", f"AZURE_AUDIO_API_KEY={DUMMY_AZURE_KEY}",
             "-e", f"AZURE_AUDIO_BASE_URL={azure_url}",
             "-e", f"GATEWAY_API_KEY={GATEWAY_API_KEY}",
             image]
        )
        assert run.returncode == 0, run.stderr[-2000:]

        assert _container_ready(
            GATEWAY_CONTAINER, "http://127.0.0.1:8000/health"
        ), "mm-gateway did not become healthy"
        # The published port is host-only; the gateway reaches this service
        # over the shared Docker network by container name.
        base = f"http://127.0.0.1:{port}"
        assert _wait_http(f"{base}/health"), (
            "gateway did not answer /health on the published port"
        )
        yield base
    finally:
        _cleanup()
        _run(["docker", "network", "rm", NETWORK])


def _auth() -> dict:
    return {"Authorization": f"Bearer {GATEWAY_API_KEY}"}


def _request(method: str, url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {**_auth()}
    if data is not None:
        # urllib defaults to x-www-form-urlencoded for bytes bodies, which
        # the gateway rejects; the API is JSON-only.
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _synthesize(base: str, parameters: dict, text: str = "The rainbow has seven colors.") -> dict:
    body = {
        "model": "azure-tts",
        "input": [{"type": "text", "text": text}],
        "parameters": {"voice": "default", **parameters},
    }
    created = _request("POST", f"{base}/v1/audio", body)
    task_url = created.get("links", {}).get("self") or f"{base}/v1/audio/{created['id']}"
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        task = _request("GET", task_url)
        if task["status"] in ("succeeded", "failed", "cancelled", "expired"):
            return task
        time.sleep(2.0)
    raise TimeoutError(f"gateway audio task did not finish: {task_url}")


def _audio_bytes(task: dict) -> tuple[bytes, dict]:
    import base64

    output = (task.get("outputs") or [{}])[0]
    uri = output.get("uri", "")
    assert uri.startswith("data:"), f"expected inline audio, got {uri[:40]}"
    return base64.b64decode(uri.split(",", 1)[1]), output


class TestMmGatewaySpeech:
    """mm-gateway's /v1/audio, routed to this service by its azure backend."""

    def test_mp3_synthesis_succeeds(self, gateway):
        task = _synthesize(
            gateway, {"file_format": "mp3", "delivery": "inline"},
            "The speech gateway is ready for the real end to end test.",
        )
        assert task["status"] == "succeeded", task
        data, output = _audio_bytes(task)
        assert output["mime_type"] == "audio/mpeg"
        assert output["sample_rate_hz"] == 24000
        assert output["duration_seconds"] > 1.0
        assert data[:3] == b"ID3" or data[0] == 0xFF
        assert len(data) > 1000
        assert task["usage"]["input_characters"] > 0
        assert task["usage"]["duration_seconds"] == output["duration_seconds"]

    def test_wav_synthesis_succeeds(self, gateway):
        task = _synthesize(gateway, {"file_format": "wav", "delivery": "inline"})
        assert task["status"] == "succeeded", task
        data, output = _audio_bytes(task)
        assert output["mime_type"] == "audio/wav"
        assert data[:4] == b"RIFF"

    def test_pcm_synthesis_succeeds(self, gateway):
        task = _synthesize(gateway, {"file_format": "pcm", "delivery": "inline"})
        assert task["status"] == "succeeded", task
        data, output = _audio_bytes(task)
        assert output["mime_type"] == "audio/pcm"
        # 16-bit mono: an even byte count at the reported duration/rate.
        rate = output["sample_rate_hz"]
        expected = int(output["duration_seconds"] * rate) * 2
        assert abs(len(data) - expected) <= rate  # ~1 s of scheduling slack
        assert len(data) % 2 == 0

    def test_speed_prosody_is_honored(self, gateway):
        """The gateway's speed control (SSML prosody rate) synthesizes."""
        task = _synthesize(
            gateway,
            {"file_format": "mp3", "delivery": "inline", "speed": 1.25},
            "Speaking a little faster now.",
        )
        assert task["status"] == "succeeded", task
        data, _ = _audio_bytes(task)
        assert len(data) > 500

    def test_language_locale_is_honored(self, gateway):
        """Multilingual voice + language wraps text in <lang> SSML."""
        task = _synthesize(
            gateway,
            {"file_format": "mp3", "delivery": "inline", "language": "fr-FR"},
            "Bonjour, ceci est un test.",
        )
        assert task["status"] == "succeeded", task
        data, _ = _audio_bytes(task)
        assert len(data) > 500

    def test_models_and_voices_reflect_azure_backend(self, gateway):
        models = _request("GET", f"{gateway}/v1/models?modality=audio")
        ids = {m["id"] for m in models["data"]}
        assert "azure-tts" in ids
        assert "gateway-audio-azure" in ids

        limits = _request("GET", f"{gateway}/v1/models/limits?modality=audio")
        entry = next(m for m in limits["data"] if m["id"] == "azure-tts")
        assert entry["limits"]["supported_file_formats"] == [
            "mp3", "wav", "pcm", "opus",
        ]
        assert entry["limits"]["supports_voice_cloning"] is False

        voices = _request("GET", f"{gateway}/v1/voices")
        assert any(v["id"] == "default" for v in voices["data"])

    def test_estimate_is_admissible(self, gateway):
        estimate = _request(
            "POST", f"{gateway}/v1/audio/estimate",
            {
                "model": "azure-tts",
                "input": [{"type": "text", "text": "Estimate me."}],
                "parameters": {"voice": "default", "file_format": "mp3"},
            },
        )
        assert estimate["model"] == "azure-tts"
        assert estimate["candidates"][0]["admissible"] is True

    def test_unknown_preset_is_rejected(self, gateway):
        """A preset the azure backend does not serve fails validation."""
        import urllib.error

        body = {
            "model": "azure-tts",
            "input": [{"type": "text", "text": "Hello."}],
            "parameters": {"voice": "nonexistent", "file_format": "mp3"},
        }
        req = urllib.request.Request(
            f"{gateway}/v1/audio",
            data=json.dumps(body).encode(),
            method="POST",
            headers={**_auth(), "Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(req, timeout=30)
        assert excinfo.value.code in (400, 422)

    def test_voices_endpoint_rejects_cloning(self, gateway):
        """The azure backend advertises no voice cloning."""
        import urllib.error

        body = {
            "model": "azure-tts",
            "input": [{"type": "audio", "uri": "data:audio/wav;base64,AAAA"}],
            "parameters": {"name": "Speaker"},
            "consent": {"granted": True},
        }
        req = urllib.request.Request(
            f"{gateway}/v1/voices",
            data=json.dumps(body).encode(),
            method="POST",
            headers={**_auth(), "Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(req, timeout=30)
        assert excinfo.value.code in (400, 422)

    def test_authentication_is_enforced_by_gateway(self, gateway):
        """Anonymous requests are rejected by the gateway front end."""
        import urllib.error

        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(f"{gateway}/v1/models", timeout=30)
        assert excinfo.value.code == 401
