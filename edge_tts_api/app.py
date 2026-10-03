"""Azure Speech-compatible TTS service backed by edge-tts.

Two API surfaces are exposed:

1. The REST **text-to-speech API** (what the official Speech SDKs call
   "the REST API"): POST an SSML document, receive audio bytes, plus the
   voices/list route.
2. The **SDK websocket protocol** at ``/tts/cognitiveservices/websocket/v1``
   — the same protocol the official ``azure-cognitiveservices-speech``
   SpeechSynthesizer speaks, so the SDK works against this service with
   ``SpeechConfig(endpoint="ws://host:port/tts/cognitiveservices/websocket/v1")``
   (or ``SpeechConfig(host="ws://host:port", ...)``).

The earlier Azure Batch Synthesis API surface (PUT/GET/DELETE
/texttospeech/batchsyntheses) is retained for compatibility.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response as FastAPIResponse

from . import (
    audio,
    config,
    results as results_mod,
    sdk_synth,
    store as store_mod,
    synth,
)
from .schemas import (
    BatchSynthesisConfig,
    BatchSynthesisInput,
    BatchSynthesisRequest,
    SYNTHESIS_ID_PATTERN,
)

logger = logging.getLogger("edge_tts_api")

app = FastAPI(
    title="edge-tts Azure-compatible TTS service",
    description=(
        "Azure Speech text-to-speech compatible service backed by edge-tts "
        "(Microsoft Edge online TTS): REST TTS + voices/list, and the "
        "official Speech SDK websocket protocol."
    ),
    version="2.0.0",
)

store = store_mod.JobStore()

_semaphore: Optional[asyncio.Semaphore] = None


@app.on_event("startup")
async def _startup() -> None:
    config.ensure_dirs()
    store.load()
    global _semaphore
    _semaphore = asyncio.Semaphore(config.CONCURRENCY)

    async def sweeper() -> None:
        while True:
            await asyncio.sleep(3600)
            try:
                removed = store.sweep_expired()
                if removed:
                    logger.info("TTL sweep removed %d expired jobs", removed)
            except Exception:  # noqa: BLE001
                logger.exception("TTL sweep failed")

    asyncio.create_task(sweeper())


# ---------------------------------------------------------------------------
# Errors follow the Azure contract: {"error": {"code": ..., "message": ...}}
# ---------------------------------------------------------------------------


def azure_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"code": code, "message": message}}
    )


@app.exception_handler(HTTPException)
async def _http_error_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail:
        return azure_error(exc.status_code, detail["code"], detail["message"])
    return azure_error(exc.status_code, "HTTPException", str(detail))


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Map pydantic validation failures onto the Azure error contract.

    Azure answers a body missing required fields (e.g. inputs) with
    HTTP 400 + {"error": {"code": "BadRequest", ...}}, not FastAPI's
    default 422 detail list.
    """
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(part) for part in first.get("loc", []) if part != "body")
    return azure_error(
        400,
        "BadRequest",
        f"The {field or 'request body'} is invalid. {first.get('msg', '')}".strip(),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYNTHESIS_ID_RE = re.compile(SYNTHESIS_ID_PATTERN)


def _check_id(synthesis_id: str) -> None:
    if not _SYNTHESIS_ID_RE.match(synthesis_id):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "BadRequest",
                "message": (
                    "The id is invalid. It must be 3-64 characters long, contain "
                    "only numbers, letters, hyphens, underscores, and dots, and "
                    "start and end with a letter or number."
                ),
            },
        )


def _check_api_version(api_version: str) -> None:
    if api_version not in config.SUPPORTED_API_VERSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "BadRequest",
                "message": (
                    f"The api-version '{api_version}' is invalid. Supported "
                    f"versions: {', '.join(config.SUPPORTED_API_VERSIONS)}."
                ),
            },
        )


def _base_url(request: Request) -> str:
    if config.BASE_URL:
        return config.BASE_URL.rstrip("/")
    return str(request.base_url).rstrip("/")


def _result_url(request: Request, job_id: str) -> str:
    return (
        f"{_base_url(request)}/texttospeech/batchsyntheses/{job_id}"
        "/files/results.zip"
    )


def _summary_url(request: Request, job_id: str) -> Optional[str]:
    record = store.get(job_id)
    if record is None:
        return None
    if record["properties"].get("decompressOutputFiles"):
        return (
            f"{_base_url(request)}/texttospeech/batchsyntheses/{job_id}"
            "/files/summary.json"
        )
    return None


def _api_model(record: Dict[str, Any], request: Request) -> Dict[str, Any]:
    """Serialize an internal job record into the Azure response shape."""
    response: Dict[str, Any] = {
        "id": record["id"],
        "status": record["status"],
        "createdDateTime": record["createdDateTime"],
        "lastActionDateTime": record["lastActionDateTime"],
        "inputKind": record["inputKind"],
        "customVoices": record.get("customVoices") or {},
        "properties": record["properties"],
    }
    if record.get("description") is not None:
        response["description"] = record["description"]
    if record.get("synthesisConfig") is not None:
        response["synthesisConfig"] = record["synthesisConfig"]
    if record["status"] == store_mod.STATUS_SUCCEEDED:
        outputs: Dict[str, Any] = {"result": _result_url(request, record["id"])}
        summary_url = _summary_url(request, record["id"])
        if summary_url:
            outputs["summary"] = summary_url
        response["outputs"] = outputs
    return response


# ---------------------------------------------------------------------------
# Background synthesis
# ---------------------------------------------------------------------------


async def _run_job(job_id: str) -> None:
    """Synthesize all inputs of a job, then assemble the results ZIP."""
    record = store.get(job_id)
    if record is None:
        return

    assert _semaphore is not None
    inputs: List[BatchSynthesisInput] = record["_inputs"]
    props = record["_properties"]
    output_format = props.outputFormat

    prosody = synth._prosody_from_config(
        BatchSynthesisConfig(**record["synthesisConfig"])
        if record.get("synthesisConfig")
        else None
    )
    contents = [item.content for item in inputs]

    store.update(
        job_id, status=store_mod.STATUS_RUNNING, lastActionDateTime=store_mod.utc_now_iso()
    )

    # index -> artifacts of a succeeded input
    artifacts: Dict[int, Dict[str, Any]] = {}
    failures: Dict[int, Dict[str, Any]] = {}

    async def run_one(index: int, content: str) -> None:
        try:
            async with _semaphore:
                mp3, word_json, sentence_json = await synth.synthesize_one(
                    content,
                    input_kind=record["inputKind"],
                    voice=record["_voice"],
                    prosody=prosody,
                    word_boundary=props.wordBoundaryEnabled,
                    sentence_boundary=props.sentenceBoundaryEnabled,
                )
                encoded = audio.transcode(mp3, output_format)
                duration_ms, _ = audio.probe(mp3)
            artifacts[index] = {
                "audio": encoded,
                "sizeInBytes": len(encoded),
                "durationInMilliseconds": duration_ms,
                "word": json.dumps(word_json, indent=2) if word_json else "",
                "sentence": json.dumps(sentence_json, indent=2) if sentence_json else "",
            }
        except synth.SynthesisError as exc:
            failures[index] = exc.to_dict()
        except Exception as exc:  # noqa: BLE001 - never kill the whole job
            failures[index] = {
                "code": "InternalServerError",
                "message": f"Synthesis failed: {exc}",
            }

    await asyncio.gather(
        *(asyncio.create_task(run_one(i, c)) for i, c in enumerate(contents))
    )

    succeeded_indices = sorted(artifacts)
    succeeded = len(succeeded_indices)
    failed = len(failures)
    total_size = sum(artifacts[i]["sizeInBytes"] for i in succeeded_indices)
    total_duration_ms = sum(
        artifacts[i]["durationInMilliseconds"] for i in succeeded_indices
    )

    if props.concatenateResult and succeeded:
        audio_files = [
            _concat_encoded(
                [artifacts[i]["audio"] for i in succeeded_indices], output_format
            )
        ]
        word_texts = [artifacts[i]["word"] for i in succeeded_indices]
        sentence_texts = [artifacts[i]["sentence"] for i in succeeded_indices]
        word_files = [_merge_boundary_json(word_texts)]
        sentence_files = [_merge_boundary_json(sentence_texts)]
        total_size = len(audio_files[0])
    else:
        audio_files = [artifacts[i]["audio"] for i in succeeded_indices]
        word_files = [artifacts[i]["word"] for i in succeeded_indices]
        sentence_files = [artifacts[i]["sentence"] for i in succeeded_indices]

    per_input: List[Dict[str, Any]] = []
    for index, content in enumerate(contents):
        if index in artifacts:
            per_input.append(
                {
                    "index": index,
                    "status": store_mod.STATUS_SUCCEEDED,
                    "sizeInBytes": artifacts[index]["sizeInBytes"],
                    "durationInMilliseconds": artifacts[index]["durationInMilliseconds"],
                }
            )
        else:
            per_input.append(
                {"index": index, "status": store_mod.STATUS_FAILED, "error": failures.get(index)}
            )

    # Azure marks the job failed when any input fails.
    status = (
        store_mod.STATUS_SUCCEEDED if failed == 0 else store_mod.STATUS_FAILED
    )
    internal_id = store_mod.new_internal_id()

    summary = results_mod.build_summary(
        job_id=job_id,
        internal_id=internal_id,
        status=status,
        input_contents=contents,
        results=per_input,
        concatenate_result=props.concatenateResult,
        extension=audio.default_extension(output_format),
    )

    zip_bytes = results_mod.build_zip(
        output_format=output_format,
        concatenate_result=props.concatenateResult,
        audio_files=audio_files,
        word_files=word_files,
        sentence_files=sentence_files,
        summary=summary,
        debug_info=per_input,
        syntheses=[
            {
                "id": synth.connect_id(),
                "status": per_input[index]["status"],
                **(
                    {"error": per_input[index]["error"]}
                    if per_input[index]["status"] == store_mod.STATUS_FAILED
                    else {}
                ),
            }
            for index in range(len(contents))
        ],
    )

    job_dir = store.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "results.zip").write_bytes(zip_bytes)
    (job_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    properties = {
        "timeToLiveInHours": props.timeToLiveInHours,
        "outputFormat": output_format,
        "concatenateResult": props.concatenateResult,
        "decompressOutputFiles": props.decompressOutputFiles,
        "wordBoundaryEnabled": props.wordBoundaryEnabled,
        "sentenceBoundaryEnabled": props.sentenceBoundaryEnabled,
        "sizeInBytes": total_size,
        "succeededAudioCount": succeeded,
        "failedAudioCount": failed,
        "durationInMilliseconds": total_duration_ms,
        "billingDetails": {"neuralCharacters": sum(len(c) for c in contents)},
    }
    if failures:
        first_failure = failures[min(failures)]
        properties["error"] = first_failure

    store.update(
        job_id,
        status=status,
        lastActionDateTime=store_mod.utc_now_iso(),
        properties=properties,
    )


def _concat_encoded(chunks: List[bytes], output_format: str) -> bytes:
    """Concatenate already-encoded audio chunks into one buffer.

    Chunks are decoded to PCM, joined sample-accurately, and re-encoded in
    the target format (works uniformly for WAV, raw PCM, and MP3).
    """
    if len(chunks) == 1:
        return chunks[0]
    samples = [audio.decode_to_pcm(chunk)[0] for chunk in chunks]
    rate = audio.decode_to_pcm(chunks[0])[1]
    joined = np.concatenate(samples)
    return audio.transcode_mp3_from_pcm(joined, rate, output_format)


def _merge_boundary_json(json_texts: List[str]) -> List[str]:
    """Merge per-input boundary JSON arrays into one concatenated text."""
    if not any(json_texts):
        return [""]
    merged: List[Any] = []
    for text in json_texts:
        if text:
            merged.extend(json.loads(text))
    return [json.dumps(merged, indent=2)] if merged else [""]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.put("/texttospeech/batchsyntheses/{synthesis_id}")
async def create_batch_synthesis(
    synthesis_id: str,
    request_body: BatchSynthesisRequest,
    request: Request,
    response: Response,
    api_version: str = Query(..., alias="api-version"),
    content_length: Optional[str] = Header(None, alias="content-length"),
) -> Dict[str, Any]:
    _check_id(synthesis_id)
    _check_api_version(api_version)

    if content_length is not None:
        try:
            if int(content_length) > config.MAX_PAYLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "code": "BadRequest",
                        "message": (
                            "The request payload is too large. The maximum "
                            "JSON payload size is 2 megabytes."
                        ),
                    },
                )
        except ValueError:
            pass

    if len(request_body.inputs) > config.MAX_INPUTS:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "BadRequest",
                "message": (
                    "The number of requested text inputs exceeded the limit "
                    "of 10,000."
                ),
            },
        )

    voice: Optional[str] = None
    if request_body.inputKind == "PlainText":
        if (
            request_body.synthesisConfig is None
            or not request_body.synthesisConfig.voice
        ):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "BadRequest",
                    "message": (
                        "The synthesisConfig voice property is required for "
                        "PlainText input."
                    ),
                },
            )
        voice = request_body.synthesisConfig.voice
    elif request_body.inputKind == "SSML" and request_body.synthesisConfig is not None:
        voice = request_body.synthesisConfig.voice

    props = request_body.properties
    if props.outputFormat and not audio.is_supported(props.outputFormat):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "BadRequest",
                "message": (
                    f"The outputFormat '{props.outputFormat}' is unsupported or "
                    "invalid. Provide a valid format value, or leave outputFormat "
                    "empty to use the default setting."
                ),
            },
        )

    existing = store.get(synthesis_id)
    if existing is not None:
        if existing["status"] not in store_mod.TERMINAL_STATUSES:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "Conflict",
                    "message": (
                        f"A batch synthesis job with id '{synthesis_id}' already "
                        "exists and is not finished."
                    ),
                },
            )
        store.delete(synthesis_id)

    now = store_mod.utc_now_iso()
    record: Dict[str, Any] = {
        "id": synthesis_id,
        "status": store_mod.STATUS_NOT_STARTED,
        "createdDateTime": now,
        "lastActionDateTime": now,
        "inputKind": request_body.inputKind,
        "customVoices": request_body.customVoices or {},
        "properties": {
            "timeToLiveInHours": props.timeToLiveInHours,
            "outputFormat": props.outputFormat,
            "concatenateResult": props.concatenateResult,
            "decompressOutputFiles": props.decompressOutputFiles,
            "wordBoundaryEnabled": props.wordBoundaryEnabled,
            "sentenceBoundaryEnabled": props.sentenceBoundaryEnabled,
            "destinationContainerUrl": props.destinationContainerUrl,
            "destinationPath": props.destinationPath,
        },
        "_inputs": request_body.inputs,
        "_properties": props,
        "_voice": voice,
    }
    if request_body.description is not None:
        record["description"] = request_body.description
    if request_body.synthesisConfig is not None:
        record["synthesisConfig"] = request_body.synthesisConfig.model_dump(
            exclude_none=True
        )

    store.create(record)

    task = asyncio.create_task(_run_job(synthesis_id))
    store.attach_task(synthesis_id, task)

    response.status_code = 201
    response.headers["Operation-Location"] = _result_url(request, synthesis_id)
    return _api_model(record, request)


@app.get("/texttospeech/batchsyntheses/{synthesis_id}")
async def get_batch_synthesis(
    synthesis_id: str, request: Request, api_version: str = Query(..., alias="api-version")
) -> Dict[str, Any]:
    _check_id(synthesis_id)
    _check_api_version(api_version)
    record = store.get(synthesis_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "NotFound",
                "message": "The specified entity cannot be found.",
            },
        )
    return _api_model(record, request)


@app.get("/texttospeech/batchsyntheses")
async def list_batch_syntheses(
    request: Request,
    api_version: str = Query(..., alias="api-version"),
    skip: int = Query(0, ge=0),
    maxpagesize: int = Query(100, ge=1, le=100),
) -> Dict[str, Any]:
    _check_api_version(api_version)
    jobs = sorted(
        store.all_jobs(), key=lambda record: record["createdDateTime"], reverse=True
    )
    page = jobs[skip : skip + maxpagesize]
    body: Dict[str, Any] = {"value": [_api_model(record, request) for record in page]}
    if skip + maxpagesize < len(jobs):
        body["nextLink"] = (
            f"{_base_url(request)}/texttospeech/batchsyntheses"
            f"?skip={skip + maxpagesize}&maxpagesize={maxpagesize}"
            f"&api-version={api_version}"
        )
    return body


@app.delete("/texttospeech/batchsyntheses/{synthesis_id}")
async def delete_batch_synthesis(
    synthesis_id: str, api_version: str = Query(..., alias="api-version")
) -> Response:
    _check_id(synthesis_id)
    _check_api_version(api_version)
    if store.get(synthesis_id) is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "NotFound",
                "message": "The specified entity cannot be found.",
            },
        )
    store.delete(synthesis_id)
    return Response(status_code=204)


@app.get("/texttospeech/batchsyntheses/{synthesis_id}/files/{file_name}")
async def get_result_file(
    synthesis_id: str, file_name: str, api_version: str = Query(..., alias="api-version")
) -> Response:
    """Serve a job's result artifacts (outputs.result / outputs.summary)."""
    _check_id(synthesis_id)
    _check_api_version(api_version)
    record = store.get(synthesis_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "NotFound",
                "message": "The specified entity cannot be found.",
            },
        )
    if record["status"] != store_mod.STATUS_SUCCEEDED:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "NotFound",
                "message": "The synthesis job has not succeeded.",
            },
        )
    if file_name not in ("results.zip", "summary.json"):
        raise HTTPException(
            status_code=404,
            detail={
                "code": "NotFound",
                "message": "The specified file cannot be found.",
            },
        )
    path = store.job_dir(synthesis_id) / file_name
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "NotFound",
                "message": "The specified file cannot be found.",
            },
        )
    media_type = "application/zip" if file_name == "results.zip" else "application/json"
    return FileResponse(path, media_type=media_type, filename=file_name)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Azure text-to-speech REST API (api-version-less, like the real service)
# ---------------------------------------------------------------------------

# X-Microsoft-OutputFormat values the REST endpoint accepts (same set the
# websocket path negotiates, plus their riff equivalents which the REST API
# must produce itself since there is no client to add headers).


@app.get("/tts/cognitiveservices/voices/list")
@app.get("/cognitiveservices/voices/list")
async def list_voices_rest() -> FastAPIResponse:
    """List voices in the Azure voices/list response shape."""
    try:
        voices = await synth.gather_voices()
    except Exception as exc:  # noqa: BLE001 - upstream edge-tts failure
        raise HTTPException(
            status_code=502,
            detail={
                "code": "ServiceUnavailable",
                "message": f"Failed to fetch the voice list: {exc}",
            },
        ) from exc
    return JSONResponse(content=_azure_voices(voices))


def _azure_voices(edge_voices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map the edge-tts voice list onto the Azure voices/list shape."""
    out: List[Dict[str, Any]] = []
    for voice in edge_voices:
        friendly = voice.get("FriendlyName", "")
        display = friendly.split(" Online")[0].replace("Microsoft ", "") or voice.get(
            "ShortName", ""
        )
        entry = {
            "Name": voice.get("Name", ""),
            "DisplayName": display,
            "LocalName": display,
            "ShortName": voice.get("ShortName", ""),
            "Gender": voice.get("Gender", "Unknown"),
            "Locale": voice.get("Locale", ""),
            "LocaleName": voice.get("LocaleName", ""),
            "SampleRateHertz": "24000",
            "VoiceType": "Neural",
            "Status": voice.get("Status", "GA"),
            "WordsPerMinute": "150",
        }
        # Optional detail fields when edge-tts exposes them.
        voice_tag = voice.get("VoiceTag") or {}
        content_categories = voice_tag.get("ContentCategories") or []
        if content_categories:
            entry["StyleList"] = []
            entry["VoiceTag"] = {
                "ContentCategories": content_categories,
                "VoicePersonalities": voice_tag.get(
                    "VoicePersonalities", []
                ),
            }
        out.append(entry)
    return out


@app.post("/cognitiveservices/v1")
async def text_to_speech_rest(
    request: Request,
    output_format: Optional[str] = Header(None, alias="X-Microsoft-OutputFormat"),
    content_type: Optional[str] = Header(None),
) -> FastAPIResponse:
    """The classic REST text-to-speech endpoint: SSML in, audio out."""
    if content_type and "ssml+xml" not in content_type:
        raise HTTPException(
            status_code=415,
            detail={
                "code": "UnsupportedMediaType",
                "message": (
                    "The Content-Type must be application/ssml+xml. "
                    f"Received: '{content_type}'."
                ),
            },
        )
    if not output_format:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "BadRequest",
                "message": "The X-Microsoft-OutputFormat header is required.",
            },
        )
    if output_format and not audio.is_supported(output_format):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "BadRequest",
                "message": (
                    f"The output format '{output_format}' is invalid or "
                    "unsupported."
                ),
            },
        )
    ssml = (await request.body()).decode("utf-8", errors="replace")
    if not ssml.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "code": "BadRequest",
                "message": "The request body must contain SSML.",
            },
        )

    voice = synth.extract_voice_from_ssml(ssml)
    try:
        mp3, _, _ = await synth.synthesize_one(
            ssml,
            input_kind="SSML",
            voice=voice or "",
            prosody={},
            word_boundary=False,
            sentence_boundary=False,
        )
    except synth.SynthesisError as exc:
        raise HTTPException(
            status_code=400, detail={"code": exc.code, "message": exc.message}
        ) from exc

    encoded = audio.transcode(mp3, output_format)
    media = "audio/wav" if output_format.startswith("riff-") else "audio/mpeg"
    if output_format.startswith("raw-"):
        media = "application/octet-stream"
    return FastAPIResponse(content=encoded, media_type=media)


# ---------------------------------------------------------------------------
# Official Speech SDK websocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/tts/cognitiveservices/websocket/v1")
@app.websocket("/cognitiveservices/websocket/v1")
async def sdk_websocket_endpoint(websocket: WebSocket) -> None:
    """Serve the official Speech SDK synthesis protocol.

    The ``/tts`` prefix form is what the SDK requests when configured with
    ``endpoint=...``; the shorter path is what it derives from
    ``host="ws://host:port"``.
    """
    await sdk_synth.handle_sdk_websocket(websocket)
