"""MIDI exporter.

Writes a multi-track MIDI file (melody track + voicing track + chord track
as text markers) derived from a :class:`ml.types.TranscriptionResult`.
"""

from __future__ import annotations

from pathlib import Path

from ml.types import TranscriptionResult


def build_midi(result: TranscriptionResult, dest_path: Path) -> Path:
    """Write a Standard MIDI File for the transcription. Returns ``dest_path``."""
    raise NotImplementedError("Implemented in PR-12 (MIDI writer).")
