"""Result ZIP builder for the batch synthesis service.

Produces the ZIP layout of the Azure batch synthesis API: numbered audio
files (``0001.wav``), a ``summary.json``, and optional boundary files
(``0001.word.json``, ``0001.sentence.json``) plus ``0001.debug.json``.
"""

import io
import json
import zipfile
from typing import Any, Dict, List, Optional, Union

from . import audio
from .schemas import BatchSynthesisProperties


def _as_text(value: Union[str, List, Dict]) -> str:
    """Coerce a boundary payload to JSON text (str passthrough)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2)


def build_summary(
    job_id: str,
    internal_id: str,
    status: str,
    input_contents: List[str],
    results: List[Dict[str, Any]],
    concatenate_result: bool,
    extension: str,
) -> Dict[str, Any]:
    """Build the Azure-format summary.json object."""
    formatted = []
    for index, result in enumerate(results):
        if concatenate_result:
            audio_file_name = f"0001{extension}"
        else:
            audio_file_name = f"{index + 1:04d}{extension}"
        entry: Dict[str, Any] = {
            "contents": [input_contents[index]] if index < len(input_contents) else [],
            "status": result["status"],
            "audioFileName": audio_file_name,
        }
        if result["status"] == "Succeeded":
            entry["properties"] = {
                "sizeInBytes": str(result["sizeInBytes"]),
                "durationInMilliseconds": str(result["durationInMilliseconds"]),
            }
        else:
            entry["properties"] = {
                "error": result.get("error", {}),
            }
        formatted.append(entry)
    return {
        "jobID": internal_id,
        "status": status,
        "results": formatted,
    }


def build_zip(
    *,
    output_format: str,
    concatenate_result: bool,
    audio_files: List[bytes],
    word_files: List[str],
    sentence_files: List[str],
    summary: Dict[str, Any],
    debug_info: List[Dict[str, Any]],
    syntheses: List[Dict[str, Any]],
) -> bytes:
    """Assemble the results ZIP archive.

    audio_files: one encoded audio buffer per input (or a single buffer
    when concatenate_result is true).
    word_files/sentence_files: raw JSON text per input ("" when absent).
    """
    extension = audio.default_extension(output_format)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:

        def entry_name(index: int, suffix: str) -> str:
            if concatenate_result:
                number = 1
            else:
                number = index + 1
            prefix = f"{number:04d}"
            return f"{prefix}{suffix}"

        for index, audio_bytes in enumerate(audio_files):
            zf.writestr(entry_name(index, extension), audio_bytes)

        for index, text in enumerate(word_files):
            if text:
                zf.writestr(entry_name(index, ".word.json"), _as_text(text))

        for index, text in enumerate(sentence_files):
            if text:
                zf.writestr(entry_name(index, ".sentence.json"), _as_text(text))

        for index, synthesis in enumerate(syntheses):
            zf.writestr(
                entry_name(index, ".debug.json"), json.dumps(synthesis, indent=2)
            )

        zf.writestr("summary.json", json.dumps(summary, indent=2))
        # Duplicate the summary at the root name Azure also exposes.
        zf.writestr("debug.json", json.dumps(debug_info, indent=2))
    return buffer.getvalue()
