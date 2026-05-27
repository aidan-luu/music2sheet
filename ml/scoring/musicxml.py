"""MusicXML score builder.

MusicXML is the **canonical output format** of Sheet Sage 2. Every other
score artifact (LilyPond PDF, MIDI) is derived from the MusicXML emitted
here so that the frontend (Agent A, via OpenSheetMusicDisplay or Verovio)
and the LilyPond/MIDI exporters share a single source of truth.
"""

from __future__ import annotations

from ml.types import TranscriptionResult


def build_musicxml(result: TranscriptionResult) -> str:
    """Render a :class:`TranscriptionResult` to a MusicXML 3.1 string.

    The returned string is a complete ``score-partwise`` document with two
    staves: lead-sheet melody+chords on top, voicing piano-roll on bottom.
    """
    raise NotImplementedError("Implemented in PR-12 (MusicXML writer).")
