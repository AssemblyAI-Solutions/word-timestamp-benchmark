"""Shared types for all transcriber providers.

A transcriber is a callable: `(audio_path, model, api_key) -> List[Word]`.
It must return word-level timestamps in milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


@dataclass
class Word:
    """Single token with start/end times in integer milliseconds.

    `word_count` records how many original tokens this Word represents on
    its side of the alignment. Defaults to 1 for native (un-merged) tokens
    from a TextGrid or a provider response; gets set to N when a substitution
    merge collapses N tokens into a single virtual span. The match-rate
    metric uses the ref-side `word_count` so a merged span correctly
    accounts for all the reference words it covers.
    """
    text: str
    start_ms: int
    end_ms: int
    word_count: int = 1


class TranscriptionError(RuntimeError):
    """Raised when a provider call fails for any reason (API error, auth, etc)."""


# Type alias for a transcriber callable.
Transcriber = Callable[[Path, str, Optional[str]], List[Word]]

# Provider registry: name -> callable.
REGISTRY: Dict[str, Transcriber] = {}


def register(name: str) -> Callable[[Transcriber], Transcriber]:
    """Decorator: register a transcriber under the given provider name."""
    def _wrap(fn: Transcriber) -> Transcriber:
        REGISTRY[name] = fn
        return fn
    return _wrap


def get_transcriber(name: str) -> Transcriber:
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown provider {name!r}. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]
