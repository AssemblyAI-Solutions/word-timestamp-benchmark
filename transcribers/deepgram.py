"""Deepgram transcriber.

Defaults to `nova-3`. Word-level timestamps are returned in the
`words` array under each channel's alternatives. Times are seconds
(float) in the response; we convert to int ms.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from deepgram import DeepgramClient

from .base import Word, TranscriptionError, register


@register("deepgram")
def transcribe(
    audio_path: Path,
    model: str = "nova-3",
    api_key: Optional[str] = None,
    timeout_s: float = 60.0,
) -> List[Word]:
    key = api_key or os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise TranscriptionError(
            "Deepgram API key missing (set DEEPGRAM_API_KEY in .env)"
        )
    client = DeepgramClient(api_key=key)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    try:
        response = client.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model=model,
            punctuate=False,
            smart_format=False,
            request_options={"timeout_in_seconds": int(timeout_s), "max_retries": 1},
        )
    except Exception as e:
        raise TranscriptionError(f"Deepgram: {e}") from e

    # Response shape: results.channels[0].alternatives[0].words[]
    try:
        words_raw = response.results.channels[0].alternatives[0].words
    except (AttributeError, IndexError, TypeError) as e:
        raise TranscriptionError(
            f"Deepgram: unexpected response shape ({e})"
        ) from e

    if not words_raw:
        return []

    return [
        Word(
            text=getattr(w, "word", ""),
            start_ms=int(round(float(getattr(w, "start", 0.0)) * 1000)),
            end_ms=int(round(float(getattr(w, "end", 0.0)) * 1000)),
        )
        for w in words_raw
    ]
