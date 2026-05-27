"""LilyPond PDF renderer.

Consumes the MusicXML emitted by :mod:`ml.scoring.musicxml` and shells out
to the ``lilypond`` binary to produce a print-quality PDF. The MusicXML
remains the canonical artifact; LilyPond output is a convenience export.
"""

from __future__ import annotations

from pathlib import Path

from ml.types import TranscriptionResult


def build_lilypond_pdf(result: TranscriptionResult, dest_dir: Path) -> Path:
    """Render a transcription to a LilyPond PDF and return the output path."""
    raise NotImplementedError("Implemented in PR-12 (LilyPond writer).")
