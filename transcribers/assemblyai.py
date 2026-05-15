"""AssemblyAI transcriber.

Defaults to `universal-3-pro`. Word-level timestamps are requested via
the `speech_models` array parameter, which is the current AAI API
(the legacy `speech_model` enum is also accepted for backward compat).

Note: `format_text=False` is incompatible with `universal-3-pro`, so we
leave format_text at its default for current-API model identifiers and
let the alignment step handle ITN-induced substitutions.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import assemblyai as aai

from .base import Word, TranscriptionError, register


# Legacy SDK enum values (deprecated `speech_model` parameter).
_LEGACY_ENUM_VALUES = {m.value for m in aai.SpeechModel}


def _build_config(model: str) -> aai.TranscriptionConfig:
    config_kwargs: dict = dict(punctuate=False)
    if model in _LEGACY_ENUM_VALUES:
        config_kwargs["speech_model"] = aai.SpeechModel(model)
        config_kwargs["format_text"] = False
    else:
        # Current-API identifier (e.g. universal-3-pro, universal-2).
        # Pass via the speech_models array. Comma-separated values become
        # a priority-ordered fallback chain.
        if "," in model:
            config_kwargs["speech_models"] = [
                s.strip() for s in model.split(",") if s.strip()
            ]
        else:
            config_kwargs["speech_models"] = [model]
    return aai.TranscriptionConfig(**config_kwargs)


@register("assemblyai")
def transcribe(
    audio_path: Path,
    model: str = "universal-3-pro",
    api_key: Optional[str] = None,
    timeout_s: float = 60.0,
) -> List[Word]:
    key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")
    if not key:
        raise TranscriptionError(
            "AssemblyAI API key missing (set ASSEMBLYAI_API_KEY in .env)"
        )
    aai.settings.api_key = key
    # Tighten HTTP timeout. Default 30 s; bump for slow networks. This is what
    # makes individual hung calls bounded rather than indefinite (the SDK's
    # shared httpx client can otherwise wedge on a single bad upload).
    aai.settings.http_timeout = float(timeout_s)

    config = _build_config(model)
    transcript = aai.Transcriber(config=config).transcribe(str(audio_path))
    if transcript.error:
        raise TranscriptionError(f"AssemblyAI: {transcript.error}")

    return [
        Word(text=w.text, start_ms=int(w.start), end_ms=int(w.end))
        for w in (transcript.words or [])
    ]
