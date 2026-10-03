# edge-tts-api

An Azure Speech **Text-to-Speech API** compatible service backed by
[`edge-tts`](https://pypi.org/project/edge-tts/) (Microsoft Edge's online TTS service).

The primary surface is the Azure Text-to-Speech API, and the **official
`azure-cognitiveservices-speech` SDK works against this service unmodified** —
point `SpeechConfig` at it and `speak_text` / `speak_ssml` / `get_voices_async`
just work:

```python
import azure.cognitiveservices.speech as speechsdk

config = speechsdk.SpeechConfig(host="ws://127.0.0.1:8000", subscription="anything")
config.set_speech_synthesis_output_format(
    speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
)
synthesizer = speechsdk.SpeechSynthesizer(speech_config=config)
result = synthesizer.speak_text("The rainbow has seven colors.")
```

(Or with the explicit endpoint form:
`SpeechConfig(endpoint="ws://127.0.0.1:8000/tts/cognitiveservices/websocket/v1", ...)`.)

The REST TTS API is served too (`GET /cognitiveservices/voices/list`,
`POST /cognitiveservices/v1`), and the legacy Azure **batch synthesis API**
(`PUT/GET/DELETE /texttospeech/batchsyntheses/{id}?api-version=2024-04-01`)
remains available — point existing batch clients at this service the same way,
same endpoints, same JSON request/response contract — and get synthesized
speech from Edge's neural voices for free, without an Azure Speech resource.

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

### Official Speech SDK (websocket)

The SDK's native synthesis protocol is served on both paths the SDK can
reach:

- `/tts/cognitiveservices/websocket/v1` — used by `SpeechConfig(endpoint=...)`
- `/cognitiveservices/websocket/v1` — used by `SpeechConfig(host=...)`

Supported: `speak_text` / `speak_ssml`, all Riff/Raw PCM and MP3 output
formats (negotiated through `set_speech_synthesis_output_format`),
word/sentence boundary events, `synthesis_started` / `synthesis_completed`
events with audio duration, and `get_voices_async` (with optional locale
filter). Unknown voices surface as `CancellationReason.Error` /
`CancellationErrorCode.ServiceError`, as they do against Azure.

```python
import azure.cognitiveservices.speech as speechsdk

config = speechsdk.SpeechConfig(host="ws://127.0.0.1:8000", subscription="anything")
synthesizer = speechsdk.SpeechSynthesizer(speech_config=config)
synthesizer.synthesis_word_boundary.connect(
    lambda args: print(args.text, args.audio_offset, args.duration)
)
result = synthesizer.speak_ssml(
    '<speak version="1.0" xml:lang="en-US">'
    '<voice name="en-US-JennyNeural">'
    '<prosody rate="+20%">Quick prosody test.</prosody>'
    "</voice></speak>"
)

voices = synthesizer.get_voices_async().get()          # 300+ voices
en_us = synthesizer.get_voices_async(locale="en-US").get()
```

### REST — list voices: `GET /cognitiveservices/voices/list`

(Also served at `/tts/cognitiveservices/voices/list`.) Returns the Azure
voice-list JSON:

```json
[
  {
    "Name": "Microsoft Server Speech Text to Speech Voice (en-US, JennyNeural)",
    "DisplayName": "Jenny",
    "LocalName": "Jenny",
    "ShortName": "en-US-JennyNeural",
    "Gender": "Female",
    "Locale": "en-US",
    "LocaleName": "English (United States)",
    "SampleRateHertz": "24000",
    "VoiceType": "Neural",
    "Status": "GA",
    "WordsPerMinute": "150",
    "StyleList": [],
    "VoiceTag": {"ContentCategories": [...], "VoicePersonalities": [...]}
  }
]
```

### REST — synthesize: `POST /cognitiveservices/v1`

SSML in, audio bytes out — the classic Azure REST contract:

```bash
curl -X POST "http://127.0.0.1:8000/cognitiveservices/v1" \
  -H "Content-Type: application/ssml+xml" \
  -H "X-Microsoft-OutputFormat: riff-24khz-16bit-mono-pcm" \
  -H "Ocp-Apim-Subscription-Key: anything" \
  -d '<speak version="1.0" xml:lang="en-US"><voice name="en-US-JennyNeural">The rainbow has seven colors.</voice></speak>' \
  --output rainbow.wav
```

- `Content-Type` must contain `application/ssml+xml` (else `415`)
- `X-Microsoft-OutputFormat` is required (else `400`); values are the same
  format strings as below
- The `<voice name="...">` in the SSML selects the Edge voice

### Legacy batch synthesis API

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

Over the SDK websocket the Speech SDK requests Riff formats as the matching
raw format and prepends the RIFF header itself; formats needing codecs the
service does not carry (opus/webm/silk/g722/amr) fall back to the closest
playable PCM rate.

Unrecognized formats are rejected with the Azure-style 400 error (REST).

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

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR to `main`:

1. **Test matrix** — unit, integration, and e2e suites in parallel jobs. The
   e2e job drives the service (as a real uvicorn subprocess) using the
   **official `azure-cognitiveservices-speech` SDK** end-to-end over the
   websocket synthesis protocol (`speak_text`/`speak_ssml`, output formats,
   word boundaries, voice enumeration), plus the official Microsoft
   batch-synthesis client pattern from
   [Azure-Samples/cognitive-services-speech-sdk](https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/master/samples/batch-synthesis/python/synthesis.py)
   (`requests` + `Ocp-Apim-Subscription-Key`) for the legacy surface.
2. **Docker** — builds the image (tests run inside the build), smoke-tests it,
   then builds and publishes a **multi-arch image (linux/amd64 + linux/arm64)**
   to GHCR at `ghcr.io/sloth-os/edge-tts-api`, tagged by branch, SHA, semver,
   and `latest` on the default branch.

Run the Docker image locally:

```bash
docker build -t edge-tts-api .
docker run -p 8000:8000 -v edge-tts-data:/data edge-tts-api
curl http://127.0.0.1:8000/health
```

## Development

```bash
.venv/bin/pip install -r requirements.txt pytest requests azure-cognitiveservices-speech
.venv/bin/python -m pytest tests/unit tests/integration -q   # offline suites
.venv/bin/python -m pytest tests/e2e -m e2e -q              # live server + Edge TTS
```
