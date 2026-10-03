"""Shared configuration for the edge-tts-api service."""

import os
import tempfile
from pathlib import Path

# Root directory of the repository.
ROOT_DIR = Path(__file__).resolve().parent.parent

# Directory where finished jobs (and their result ZIPs) are stored.
OUTPUT_DIR = Path(os.environ.get("EDGE_TTS_API_OUTPUT_DIR", ROOT_DIR / "output"))

# Scratch directory for per-job synthesis artifacts.
TMP_DIR = Path(
    os.environ.get("EDGE_TTS_API_TMP_DIR", Path(tempfile.gettempdir()) / "edge-tts-api")
)

# Maximum JSON payload size accepted by the Azure batch synthesis API.
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024

# The Azure API supports up to 10000 inputs per job.
MAX_INPUTS = 10000

# Number of concurrent edge-tts synthesis workers.
CONCURRENCY = int(os.environ.get("EDGE_TTS_API_CONCURRENCY", "4"))

# Host/port to bind to (used by `python -m edge_tts_api` and run.sh).
HOST = os.environ.get("EDGE_TTS_API_HOST", "127.0.0.1")
PORT = int(os.environ.get("EDGE_TTS_API_PORT", "8000"))

# Base URL used when building absolute result URLs. Derived from the request
# when unset (falls back to http://HOST:PORT for non-HTTP contexts).
BASE_URL = os.environ.get("EDGE_TTS_API_BASE_URL")

# Supported api-version values. The service implements the 2024-04-01
# (GA) Azure batch synthesis API shape.
SUPPORTED_API_VERSIONS = ("2024-04-01",)

# Default output format used when the requested one is unsupported.
# edge-tts always synthesizes audio-24khz-48kbitrate-mono-mp3; the service
# transcodes to the requested format afterwards.
EDGE_NATIVE_FORMAT = "audio-24khz-48kbitrate-mono-mp3"
DEFAULT_OUTPUT_FORMAT = "riff-24khz-16bit-mono-pcm"


def ensure_dirs() -> None:
    """Create the output and scratch directories if they do not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
