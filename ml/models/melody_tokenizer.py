"""Framewise melody tokenizer (PR-4a).

The melody head emits one token per MERT encoder frame (75 Hz native).
This keeps the decoder length equal to the encoder length, mirroring the
SheetSage "one note per frame" formulation and avoiding the alignment
machinery that a note-event tokenizer (e.g. NOTE_ON / NOTE_OFF) would
require.

Token layout
------------
::

    0   PAD       padding for batched training (not emitted by encode)
    1   BOS       sequence start (not emitted by encode; data pipeline adds it)
    2   EOS       sequence end   (not emitted by encode; data pipeline adds it)
    3   REST      no melody active in this frame
    4-131         MIDI pitch p mapped to ``4 + p`` (128 pitches)
    132+          reserved for future duration/onset tokens

The vocabulary therefore has at minimum 132 entries; callers should
construct the decoder with ``vocab_size >= MelodyTokenizer.vocab_size``.

Polyphony / overlapping notes
-----------------------------
The melody head is monophonic by design, but real annotations sometimes
contain overlaps (e.g. ornament + sustain). When notes overlap on the
same frame, the later note in the input list wins. Callers that care
should pre-flatten to a monophonic line before calling :meth:`encode`.
"""

from __future__ import annotations

import numpy as np

from ml.types import Note


class MelodyTokenizer:
    """Encode / decode a framewise pitch stream against the melody vocab."""

    PAD: int = 0
    BOS: int = 1
    EOS: int = 2
    REST: int = 3
    PITCH_OFFSET: int = 4
    NUM_MIDI_PITCHES: int = 128

    def __init__(self, frame_rate_hz: float = 75.0) -> None:
        if frame_rate_hz <= 0:
            raise ValueError(f"frame_rate_hz must be > 0, got {frame_rate_hz}")
        self.frame_rate_hz: float = float(frame_rate_hz)

    # ------------------------------------------------------------------ #
    # Vocab helpers
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        """Minimum vocab_size required by this tokenizer (= 132)."""
        return self.PITCH_OFFSET + self.NUM_MIDI_PITCHES

    def pitch_to_token(self, midi_pitch: int) -> int:
        if not 0 <= midi_pitch < self.NUM_MIDI_PITCHES:
            raise ValueError(
                f"midi_pitch must be in [0, {self.NUM_MIDI_PITCHES}), got {midi_pitch}"
            )
        return self.PITCH_OFFSET + midi_pitch

    def token_to_pitch(self, token_id: int) -> int | None:
        """Inverse of :meth:`pitch_to_token`. Returns ``None`` for special tokens."""
        if token_id < self.PITCH_OFFSET:
            return None
        pitch = token_id - self.PITCH_OFFSET
        if pitch >= self.NUM_MIDI_PITCHES:
            return None
        return pitch

    # ------------------------------------------------------------------ #
    # Encode / decode
    # ------------------------------------------------------------------ #
    def encode(self, notes: list[Note], audio_duration_s: float) -> np.ndarray:
        """Convert a list of :class:`Note` into one framewise token per frame.

        Frames are produced at ``self.frame_rate_hz``; the array length is
        ``int(audio_duration_s * frame_rate_hz)``. Frames not covered by
        any input note are filled with :attr:`REST`. Overlaps resolve by
        last-write-wins (see the module docstring).
        """
        if audio_duration_s < 0:
            raise ValueError(f"audio_duration_s must be >= 0, got {audio_duration_s}")

        n_frames = int(audio_duration_s * self.frame_rate_hz)
        tokens = np.full(n_frames, self.REST, dtype=np.int64)

        for note in notes:
            start_frame = int(note.onset * self.frame_rate_hz)
            end_frame = int((note.onset + note.duration) * self.frame_rate_hz)
            start_frame = max(0, start_frame)
            end_frame = min(n_frames, end_frame)
            if end_frame <= start_frame:
                continue
            tokens[start_frame:end_frame] = self.pitch_to_token(note.pitch)

        return tokens

    def decode(
        self,
        token_ids: np.ndarray,
        frame_rate_hz: float | None = None,
    ) -> list[Note]:
        """Inverse of :meth:`encode` — group runs of equal pitch tokens into Notes.

        Non-pitch tokens (PAD/BOS/EOS/REST and any out-of-range IDs) are
        silently skipped. ``frame_rate_hz`` overrides ``self.frame_rate_hz``
        for the conversion; useful when a model decoded at a different
        sample rate than the tokenizer was constructed with.
        """
        rate = float(frame_rate_hz) if frame_rate_hz is not None else self.frame_rate_hz
        if rate <= 0:
            raise ValueError(f"frame_rate_hz must be > 0, got {rate}")

        notes: list[Note] = []
        if token_ids.size == 0:
            return notes

        n = int(token_ids.shape[0])
        i = 0
        while i < n:
            tok = int(token_ids[i])
            j = i + 1
            while j < n and int(token_ids[j]) == tok:
                j += 1
            pitch = self.token_to_pitch(tok)
            if pitch is not None:
                onset = i / rate
                duration = (j - i) / rate
                notes.append(Note(pitch=pitch, onset=onset, duration=duration))
            i = j
        return notes
