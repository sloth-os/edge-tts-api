"""Unit tests for the results ZIP builder."""

import io
import json
import zipfile

from edge_tts_api.results import build_summary, build_zip


class TestBuildSummary:
    def test_azure_summary_shape(self):
        summary = build_summary(
            job_id="job-1",
            internal_id="guid-1234",
            status="Succeeded",
            input_contents=["<speak>one</speak>", "<speak>two</speak>"],
            results=[
                {"index": 0, "status": "Succeeded", "sizeInBytes": 100,
                 "durationInMilliseconds": 1000},
                {"index": 1, "status": "Failed", "error": {
                    "code": "InternalServerError", "message": "boom"}},
            ],
            concatenate_result=False,
            extension=".wav",
        )
        assert summary["jobID"] == "guid-1234"
        assert summary["status"] == "Succeeded"
        first, second = summary["results"]
        assert first["audioFileName"] == "0001.wav"
        assert first["properties"]["sizeInBytes"] == "100"  # string per Azure
        assert first["properties"]["durationInMilliseconds"] == "1000"
        assert second["status"] == "Failed"
        assert second["properties"]["error"]["code"] == "InternalServerError"

    def test_concatenated_uses_single_file(self):
        summary = build_summary(
            job_id="job-1",
            internal_id="g",
            status="Succeeded",
            input_contents=["a", "b"],
            results=[
                {"index": 0, "status": "Succeeded", "sizeInBytes": 1,
                 "durationInMilliseconds": 1},
                {"index": 1, "status": "Succeeded", "sizeInBytes": 2,
                 "durationInMilliseconds": 2},
            ],
            concatenate_result=True,
            extension=".mp3",
        )
        assert all(r["audioFileName"] == "0001.mp3" for r in summary["results"])


class TestBuildZip:
    def test_zip_layout(self):
        zip_bytes = build_zip(
            output_format="riff-24khz-16bit-mono-pcm",
            concatenate_result=False,
            audio_files=[b"RIFF-audio"],
            word_files=['[{"Text": "hi"}]'],
            sentence_files=[""],
            summary={"jobID": "g", "status": "Succeeded", "results": []},
            debug_info=[{"index": 0}],
            syntheses=[{"id": "abc", "status": "Succeeded"}],
        )
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = set(zf.namelist())
            assert "0001.wav" in names
            assert "0001.word.json" in names
            # Empty boundary strings produce no entry.
            assert "0001.sentence.json" not in names
            assert "0001.debug.json" in names
            assert "summary.json" in names
            assert "debug.json" in names
            words = json.loads(zf.read("0001.word.json"))
            assert words[0]["Text"] == "hi"

    def test_zip_concatenated_numbering(self):
        zip_bytes = build_zip(
            output_format="audio-24khz-48kbitrate-mono-mp3",
            concatenate_result=True,
            audio_files=[b"joined"],
            word_files=[""],
            sentence_files=[""],
            summary={"jobID": "g", "status": "Succeeded", "results": []},
            debug_info=[],
            syntheses=[{"id": "a"}, {"id": "b"}],
        )
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert names.count("0001.mp3") == 1
            # All inputs map onto the same 0001 prefix.
            assert "0001.debug.json" in names

    def test_multiple_audio_files_numbered_sequentially(self):
        zip_bytes = build_zip(
            output_format="raw-16khz-16bit-mono-pcm",
            concatenate_result=False,
            audio_files=[b"a", b"b", b"c"],
            word_files=["", "", ""],
            sentence_files=["", "", ""],
            summary={"jobID": "g", "status": "Succeeded", "results": []},
            debug_info=[],
            syntheses=[{"id": "1"}, {"id": "2"}, {"id": "3"}],
        )
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            assert zf.namelist()[:3] == ["0001.pcm", "0002.pcm", "0003.pcm"]
