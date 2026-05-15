"""Provider-specific transcriber modules.

Each module exposes a `transcribe(audio_path, model, api_key=None) -> List[Word]`
function that returns word-level timestamps in milliseconds.
"""
from .base import Word, TranscriptionError, REGISTRY, register, get_transcriber

# Importing the provider modules registers them in REGISTRY.
from . import assemblyai as _aai  # noqa: F401
from . import deepgram as _dg  # noqa: F401
from . import openai as _oai  # noqa: F401

__all__ = ["Word", "TranscriptionError", "REGISTRY", "register", "get_transcriber"]
