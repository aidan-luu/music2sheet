"""Shared dataclasses for the music2sheet ML pipeline.

These types form the contract between Agent C (ML), Agent B (API), and Agent A
(frontend). Once a model head lands in a later PR, it must emit values matching
these schemas verbatim. The downstream score writers (PR-12) and the JSON
serializer used by the API (Agent B) both depend on field names and types
declared here.

Conventions
-----------
* Times are absolute seconds from the start of the audio file (float).
* MIDI pitches are integers in [0, 127]; pitch-class tonics are in [0, 11]
  where 0 = C, 1 = C#/Db, ... 11 = B.
* Chord labels use the Harte syntax (see Harte et al., 2005), e.g. ``"C:maj"``,
  ``"A:min7"``, ``"G:7/B"``, with the literal string ``"N"`` reserved for the
  no-chord segment.
* ``confidence`` values are model-reported scores in [0.0, 1.0]; consumers must
  not assume calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Beat:
    """A single beat event detected by the beat tracker (PR-2)."""

    time: float
    downbeat: bool
    confidence: float


@dataclass(slots=True)
class Note:
    """A monophonic-or-polyphonic note event (melody head PR-4/5, voicing PR-9)."""

    pitch: int  # MIDI pitch in [0, 127]
    onset: float  # seconds
    duration: float  # seconds
    velocity: int = 80  # MIDI velocity in [0, 127]


@dataclass(slots=True)
class Chord:
    """A chord segment emitted by the chord head (PR-6/7).

    ``label`` follows Harte syntax. ``"N"`` denotes a no-chord region.
    """

    label: str
    onset: float
    duration: float
    confidence: float


@dataclass(slots=True)
class Key:
    """A global (or windowed) key estimate from the key head (PR-8)."""

    tonic: int  # pitch class in [0, 11]
    mode: str  # "major" | "minor"
    confidence: float


@dataclass(slots=True)
class Voicing:
    """A piano-roll voicing block (PR-9/10).

    The voicing layer emits realised polyphonic chord realisations on the
    lower stave. Each ``Voicing`` covers ``[onset, onset + duration)`` and
    carries the simultaneous notes voiced during that span.
    """

    notes: list[Note]
    onset: float
    duration: float


@dataclass(slots=True)
class TranscriptionResult:
    """Top-level transcription artifact returned by :func:`ml.inference.transcribe`.

    This is the canonical hand-off object between Agent C's pipeline (PR-12)
    and Agent B's API layer. The three path fields are optional because the
    Python API may be invoked without writing artifacts to disk; the API
    layer fills them in when serving.
    """

    audio_path: str
    beats: list[Beat]
    notes: list[Note]
    chords: list[Chord]
    key: Key
    voicings: list[Voicing] = field(default_factory=list)
    musicxml: str | None = None
    midi_path: str | None = None
    lilypond_pdf_path: str | None = None
