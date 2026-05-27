"""End-to-end transcription pipeline.

PR-0 ships the signature only. Real wiring lands in PR-12 once all model
heads (melody, chord, key, voicing) and the quantization engine (PR-11) are
trained and merged. This file's signature is the contract that Agent B
relies on for the FastAPI ``/transcribe`` endpoint - changing it requires
filing a ticket against Agent B.
"""

from __future__ import annotations

from ml.types import TranscriptionResult


def transcribe(audio_path: str, *, options: dict | None = None) -> TranscriptionResult:
    """Transcribe an audio file end-to-end.

    Parameters
    ----------
    audio_path:
        Local path or URL to the input audio.
    options:
        Optional dictionary of pipeline overrides (model variants, beat-grid
        resolution, voicing density). Concrete keys are documented in PR-12.

    Returns
    -------
    TranscriptionResult
        Full lead-sheet payload: beats, melody notes, chord segments, key,
        voicings, and (when assembled) MusicXML / MIDI / LilyPond artifacts.
    """
    raise NotImplementedError("Wired up in PR-12 once model heads are trained.")
