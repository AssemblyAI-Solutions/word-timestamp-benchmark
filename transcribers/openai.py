"""OpenAI transcriber.

Defaults to `whisper-1` because it is the only OpenAI transcription
model that currently supports word-level timestamps. `gpt-4o-transcribe`
and `gpt-4o-mini-transcribe` only support `response_format=json` and
therefore cannot return word timestamps via the public API at this
time. If a user passes one of those, we raise with a clear message.

Word timestamps require `response_format=verbose_json` and
`timestamp_granularities=["word"]`. Response times are in seconds.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from .base import Word, TranscriptionError, register


# Models that DO support word timestamps via the public API.
_WORD_TS_SUPPORTED = {"whisper-1"}

# Models that exist but do NOT currently support word timestamps.
_WORD_TS_UNSUPPORTED = {
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe-diarize",
}


@register("openai")
def transcribe(
    audio_path: Path,
    model: str = "whisper-1",
    api_key: Optional[str] = None,
    timeout_s: float = 60.0,
) -> List[Word]:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise TranscriptionError(
            "OpenAI API key missing (set OPENAI_API_KEY in .env)"
        )

    if model in _WORD_TS_UNSUPPORTED:
        raise TranscriptionError(
            f"OpenAI model {model!r} does not support word-level timestamps "
            f"via the public API (only response_format=json is allowed). "
            f"Use whisper-1 for word-timestamp evals."
        )

    client = OpenAI(api_key=key, timeout=float(timeout_s), max_retries=1)

    try:
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                file=f,
                model=model,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
    except Exception as e:
        raise TranscriptionError(f"OpenAI: {e}") from e

    words_raw = getattr(resp, "words", None) or []
    if not words_raw and model in _WORD_TS_SUPPORTED:
        # If we got an empty words array on a supported model, surface that.
        # (Could indicate an empty audio file or upstream issue.)
        return []

    return [
        Word(
            text=getattr(w, "word", ""),
            start_ms=int(round(float(getattr(w, "start", 0.0)) * 1000)),
            end_ms=int(round(float(getattr(w, "end", 0.0)) * 1000)),
        )
        for w in words_raw
    ]
