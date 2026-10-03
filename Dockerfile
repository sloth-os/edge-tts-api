FROM python:3.12-slim AS base

# libsndfile is the only native dependency (soundfile needs it for MP3/WAV).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# The PyPI CDN can be slow; the tuna mirror is the fallback used locally.
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir \
      --timeout 120 --retries 5 \
      --index-url ${PIP_INDEX_URL} \
      -r requirements.txt

COPY edge_tts_api ./edge_tts_api

# Run tests during build; a broken image should fail to build.
COPY tests ./tests
COPY pytest.ini .
RUN pip install --no-cache-dir \
      --timeout 120 --retries 5 \
      --index-url ${PIP_INDEX_URL} \
      pytest requests httpx \
    && python -m pytest tests/unit -q --no-header -p no:cacheprovider \
    && rm -rf tests pytest.ini

ENV EDGE_TTS_API_HOST=0.0.0.0 \
    EDGE_TTS_API_PORT=8000 \
    EDGE_TTS_API_OUTPUT_DIR=/data \
    PYTHONUNBUFFERED=1

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["python", "-m", "edge_tts_api"]
