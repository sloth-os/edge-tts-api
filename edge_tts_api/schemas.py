"""Schemas for the Azure Batch Synthesis API compatible service.

Field names, shapes, and default values mirror the Azure Speech service
"Text to Speech Batch Synthesis API" (api-version 2024-04-01) so that
clients written against the Azure API work against this service unchanged.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

InputKind = Literal["PlainText", "SSML"]

# The input id can be used as the zip file entry name and must be safe
# for a filesystem path component in the results ZIP.
SYNTHESIS_ID_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,62}[a-zA-Z0-9]$"


class BatchSynthesisInput(BaseModel):
    """A single text or SSML input."""

    # minLength: 1 — the Azure API rejects empty contents.
    content: str = Field(min_length=1)
    description: Optional[str] = None
    id: Optional[str] = Field(default=None, pattern=r"^[^/\\]+$")


class BatchSynthesisBackgroundAudioDefinition(BaseModel):
    """Background audio setting. Kept for Azure parity; not used by edge-tts."""

    src: str
    fadein: Optional[int] = None
    fadeout: Optional[int] = None
    volume: Optional[float] = None


class BatchSynthesisConfig(BaseModel):
    """Text-to-speech configuration for plain text input."""

    voice: str = Field(min_length=1)
    style: Optional[str] = None
    rate: Optional[str] = None
    pitch: Optional[str] = None
    volume: Optional[str] = None
    backgroundAudio: Optional[BatchSynthesisBackgroundAudioDefinition] = None

    def prosody_overrides(
        self,
    ) -> Dict[str, Optional[str]]:
        """Return only the prosody fields that were set."""
        return {
            "rate": self.rate,
            "pitch": self.pitch,
            "volume": self.volume,
        }


class BatchSynthesisProperties(BaseModel):
    """Detailed properties of a batch synthesis task.

    Read-only fields (sizeInBytes, succeededAudioCount, ...) are accepted
    in requests for Azure parity but recomputed by the service when the
    job completes.
    """

    timeToLiveInHours: int = Field(
        default=744, ge=1, le=744,
        description="Hours to keep a completed job; default and max is 744 (31 days).",
    )
    outputFormat: str = Field(
        default="riff-24khz-16bit-mono-pcm",
        description="Audio output format. Unrecognized formats fall back to the default.",
    )
    concatenateResult: bool = False
    decompressOutputFiles: bool = False
    wordBoundaryEnabled: bool = False
    sentenceBoundaryEnabled: bool = False
    destinationContainerUrl: Optional[str] = None
    destinationPath: Optional[str] = None
    # Read-only outputs; populated on completion.
    sizeInBytes: Optional[int] = Field(default=None, exclude=True)
    succeededAudioCount: Optional[int] = Field(default=None, exclude=True)
    failedAudioCount: Optional[int] = Field(default=None, exclude=True)
    durationInMilliseconds: Optional[int] = Field(default=None, exclude=True)
    billingDetails: Optional[Dict[str, int]] = Field(default=None, exclude=True)
    error: Optional["BatchSynthesisError"] = Field(default=None, exclude=True)


class BatchSynthesisError(BaseModel):
    """Batch Synthesis Error."""

    code: str
    message: str


class BatchSynthesisRequest(BaseModel):
    """Request body for creating a batch synthesis job."""

    inputKind: InputKind
    inputs: List[BatchSynthesisInput] = Field(min_length=1, max_length=10000)
    synthesisConfig: Optional[BatchSynthesisConfig] = None
    customVoices: Optional[Dict[str, str]] = None
    description: Optional[str] = None
    properties: BatchSynthesisProperties = Field(
        default_factory=BatchSynthesisProperties
    )


BatchSynthesisProperties.model_rebuild()
