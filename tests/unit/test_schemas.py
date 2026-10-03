"""Unit tests for the schemas module (Azure request validation)."""

import pytest
from pydantic import ValidationError

from edge_tts_api.schemas import (
    BatchSynthesisProperties,
    BatchSynthesisRequest,
)


class TestBatchSynthesisRequest:
    def test_minimal_ssml_request(self):
        request = BatchSynthesisRequest(
            inputKind="SSML",
            inputs=[{"content": "<speak><voice name='v'>Hi</voice></speak>"}],
        )
        assert request.properties.outputFormat == "riff-24khz-16bit-mono-pcm"
        assert request.properties.timeToLiveInHours == 744

    def test_input_kind_is_case_insensitive_value(self):
        # Literal types accept the two documented values only.
        request = BatchSynthesisRequest(
            inputKind="PlainText", inputs=[{"content": "hi"}]
        )
        assert request.inputKind == "PlainText"

    def test_invalid_input_kind_rejected(self):
        with pytest.raises(ValidationError):
            BatchSynthesisRequest(inputKind="Bogus", inputs=[{"content": "x"}])

    def test_inputs_required(self):
        with pytest.raises(ValidationError):
            BatchSynthesisRequest(inputKind="SSML")

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            BatchSynthesisRequest(inputKind="SSML", inputs=[{"content": ""}])

    def test_inputs_capped_at_10000(self):
        with pytest.raises(ValidationError):
            BatchSynthesisRequest(
                inputKind="SSML", inputs=[{"content": "x"}] * 10001
            )

    def test_properties_defaults_match_azure(self):
        props = BatchSynthesisProperties()
        assert props.outputFormat == "riff-24khz-16bit-mono-pcm"
        assert props.concatenateResult is False
        assert props.decompressOutputFiles is False
        assert props.wordBoundaryEnabled is False
        assert props.sentenceBoundaryEnabled is False
        assert props.timeToLiveInHours == 744

    def test_time_to_live_bounds(self):
        with pytest.raises(ValidationError):
            BatchSynthesisProperties(timeToLiveInHours=745)
        with pytest.raises(ValidationError):
            BatchSynthesisProperties(timeToLiveInHours=0)

    def test_readonly_fields_not_serialized(self):
        # Read-only outputs are excluded from the request echo.
        props = BatchSynthesisProperties(sizeInBytes=999)
        dumped = props.model_dump(exclude_none=True)
        assert "sizeInBytes" not in dumped
        assert "billingDetails" not in dumped

    def test_synthesis_config_optional_fields(self):
        request = BatchSynthesisRequest(
            inputKind="PlainText",
            synthesisConfig={
                "voice": "en-US-JennyNeural",
                "rate": "+10%",
                "pitch": "+5Hz",
                "volume": "-10%",
                "style": "cheerful",
            },
            inputs=[{"content": "hi"}],
        )
        assert request.synthesisConfig.voice == "en-US-JennyNeural"
        assert request.synthesisConfig.prosody_overrides()["rate"] == "+10%"

    def test_synthesis_config_voice_min_length(self):
        with pytest.raises(ValidationError):
            BatchSynthesisRequest(
                inputKind="PlainText",
                synthesisConfig={"voice": ""},
                inputs=[{"content": "hi"}],
            )
