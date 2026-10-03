# edge-tts-api

An Azure Speech **Batch Synthesis API** compatible text-to-speech service backed by
[`edge-tts`](https://pypi.org/project/edge-tts/) (Microsoft Edge's online TTS service).

If you have client code written for the Azure Speech
[batch synthesis API](https://learn.microsoft.com/azure/ai-services/speech-service/batch-synthesis)
(`PUT/GET/DELETE /texttospeech/batchsyntheses/{id}?api-version=2024-04-01`), you can
point it at this service — same endpoints, same JSON request/response contract — and get
synthesized speech from Edge's neural voices for free, without an Azure Speech resource.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m edge_tts_api            # serves on http://127.0.0.1:8000
```

Configuration via environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `EDGE_TTS_API_HOST` | `127.0.0.1` | Bind host |
| `EDGE_TTS_API_PORT` | `8000` | Bind port |
| `EDGE_TTS_API_CONCURRENCY` | `4` | Parallel edge-tts synthesis workers |
| `EDGE_TTS_API_OUTPUT_DIR` | `./output` | Job store + result ZIPs |
| `EDGE_TTS_API_BASE_URL` | *(request base URL)* | Public base URL used in `outputs.result` |

## API

All four Azure batch synthesis operations are implemented. `api-version=2024-04-01`
is the supported version.

### Create a job — `PUT /texttospeech/batchsyntheses/{id}`

```bash
curl -X PUT "http://127.0.0.1:8000/texttospeech/batchsyntheses/my-job-001?api-version=2024-04-01" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "my ssml test",
    "inputKind": "SSML",
    "inputs": [
      {"content": "<speak version=\"1.0\" xml:lang=\"en-US\"><voice name=\"en-US-JennyNeural\">The rainbow has seven colors.</voice></speak>"}
    ],
    "properties": {
      "outputFormat": "riff-24khz-16bit-mono-pcm",
      "wordBoundaryEnabled": false,
      "sentenceBoundaryEnabled": false,
      "concatenateResult": false,
      "decompressOutputFiles": false
    }
  }'
```

Response `201 Created`:

```json
{
  "id": "my-job-001",
  "status": "NotStarted",
  "createdDateTime": "2026-10-02T12:00:00.0000000Z",
  "lastActionDateTime": "2026-10-02T12:00:00.0000000Z",
  "inputKind": "SSML",
  "customVoices": {},
  "description": "my ssml test",
  "properties": {
    "timeToLiveInHours": 744,
    "outputFormat": "riff-24khz-16bit-mono-pcm",
    "concatenateResult": false,
    "decompressOutputFiles": false,
    "wordBoundaryEnabled": false,
    "sentenceBoundaryEnabled": false
  }
}
```

Plain text input uses `synthesisConfig` (voice required):

```json
{
  "inputKind": "PlainText",
  "synthesisConfig": {"voice": "en-US-JennyNeural", "rate": "+10%"},
  "inputs": [{"content": "The rainbow has seven colors."}]
}
```

`synthesisConfig` supports `voice`, `rate`, `pitch`, and `volume` (mapped onto
edge-tts prosody). `style`, `backgroundAudio`, and `customVoices` are accepted but
ignored, since Edge's free tier has no custom/styled voices.

### Get a job — `GET /texttospeech/batchsyntheses/{id}`

Poll until `status` is `Succeeded` or `Failed`. On success the response gains
read-only accounting fields and `outputs`:

```json
{
  "id": "my-job-001",
  "status": "Succeeded",
  "createdDateTime": "2026-10-02T12:00:00.0000000Z",
  "lastActionDateTime": "2026-10-02T12:00:01.2500000Z",
  "inputKind": "SSML",
  "customVoices": {},
  "description": "my ssml test",
  "properties": {
    "timeToLiveInHours": 744,
    "outputFormat": "riff-24khz-16bit-mono-pcm",
    "concatenateResult": false,
    "decompressOutputFiles": false,
    "wordBoundaryEnabled": false,
    "sentenceBoundaryEnabled": false,
    "sizeInBytes": 125612,
    "succeededAudioCount": 1,
    "failedAudioCount": 0,
    "durationInMilliseconds": 2616,
    "billingDetails": {"neuralCharacters": 115}
  },
  "outputs": {
    "result": "http://127.0.0.1:8000/texttospeech/batchsyntheses/my-job-001/files/results.zip"
  }
}
```

Download the ZIP from `outputs.result` (add `?api-version=2024-04-01`). It contains:

- `0001.wav` — numbered audio files, one per input (extension follows `outputFormat`)
- `0001.word.json` / `0001.sentence.json` — boundary data when requested, in the
  Azure format (`Text`, `AudioOffset` ms, `Duration` ms)
- `0001.debug.json` — per-input synthesis debug info
- `summary.json` — the Azure-format job summary

### List jobs — `GET /texttospeech/batchsyntheses?skip=0&maxpagesize=100`

Returns `{"value": [...]}` plus `nextLink` when more pages exist.

### Delete a job — `DELETE /texttospeech/batchsyntheses/{id}`

Returns `204 No Content`. Jobs are also swept automatically once
`timeToLiveInHours` (default 744) elapses after completion.

## Output formats

edge-tts synthesizes `audio-24khz-48kbitrate-mono-mp3` natively; the service
transcodes to the requested Azure format (numpy + soundfile, no ffmpeg needed):

- `riff-{8,16,22,24,44,48}khz-16bit-mono-pcm` (`.wav`)
- `raw-{...}khz-16bit-mono-pcm` (`.pcm`)
- `audio-{16,24,48}khz-{32..192}kbitrate-mono-mp3` (`.mp3`)

Unrecognized formats are rejected with the Azure-style 400 error.

## Errors

Errors use the Azure contract:

```json
{"error": {"code": "BadRequest", "message": "The inputs is required."}}
```

## Differences from Azure

- No authentication is required (set a reverse proxy with a key check if you
  expose it publicly).
- `destinationContainerUrl` is accepted but ignored — results are served from
  this service at `outputs.result` instead of Azure Blob Storage.
- `billingDetails.neuralCharacters` counts input characters (no billing happens).
- Synthesis runs through Microsoft Edge's free read-aloud endpoint, so it is
  subject to that service's limits and availability.

## Development

```bash
.venv/bin/python -m pytest tests/ -q   # 36 tests; live ones hit the Edge service
```
