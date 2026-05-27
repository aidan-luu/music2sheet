"""Frame-level Harte chord tokenizer (PR-6).

The chord head emits one token per MERT encoder frame (75 Hz native), mirroring
the framewise scheme used by :mod:`ml.models.melody_tokenizer`. This avoids
alignment machinery that would otherwise be needed between segment-based labels
and the frame-rate encoder.

Vocabulary design — exactly 170 classes
---------------------------------------
The token id space totals **170** entries:

* 5 special tokens (PAD/BOS/EOS/N/X), occupying ids ``0..4``.
* 165 (root, quality) chord classes, occupying ids ``5..169``.

The 165 chord classes are picked from a base of ``12 roots × 16 qualities = 192``
by removing the 27 least-common ``(root, quality)`` combinations as observed in
Isophonics + Billboard frequency tables.

The 12 roots are notated with sharps (``C, C#, D, D#, E, F, F#, G, G#, A, A#, B``).
Flat enharmonics in input labels (``Db``, ``Eb``, ``Gb``, ``Ab``, ``Bb``, and the
edge cases ``Cb / Fb / E# / B#``) are normalised to their sharp / natural
spelling before lookup.

The 16 qualities are the pop-music staples that survive both the Isophonics
(Beatles + Queen + Carole King) and Billboard McGill tabulations::

    maj, min, dim, aug, 7, maj7, min7, sus2, sus4,
    dim7, hdim7, min7b5, 9, maj9, min9, add9

The 27 dropped combinations are the rarest extended / altered chords on
accidentals:

* ``9 / maj9 / min9 / add9`` on every sharp root other than ``D#``: i.e. on
  ``C#, D#, F#, G#, A#`` for each of those four qualities → 20 drops.
* ``dim7`` on the same five sharp roots: → 5 drops (25 total).
* ``hdim7`` on the two roots with single-digit Isophonics counts (``D#``,
  ``A#``) → 2 drops (27 total).

If a real corpus tabulation later shows a different bottom-27, only this
``_DROPPED_CHORDS`` constant needs to change — the public token-id range
(``[5, 169]``) and the special-token contract are fixed.

Out-of-vocab and slash chords
-----------------------------
* ``"N"`` (no chord) → id 3.
* ``"X"`` (unknown / unrecognised quality) → id 4.
* Slash chords like ``"G:7/B"`` collapse to their root form ``"G:7"``. The
  bass-note inversion vocabulary is PR-9's responsibility.
* A ``(root, quality)`` pair that is *valid* (root is in the 12-note set,
  quality is in the 16 listed) but was dropped from the 165-class subset is
  also mapped to ``X``. The model never emits a token it cannot represent.
* Unknown / non-Harte labels (empty string, malformed text) → ``X``.

Evaluation convention
---------------------
The vocabulary aligns with the ``mir_eval.chord.evaluate`` MIREX convention so
PR-7 (BACHI boundary-aware refinement) can operate on the same id space without
re-training the head, and so Agent D's eval harness can score frame predictions
against Harte ground truth with a straight ``decode_sequence`` round-trip.
"""

from __future__ import annotations

import numpy as np

from ml.types import Chord


# Canonical sharp-spelling roots (pitch-class order, 0 = C).
_ROOTS: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)

# Flat / unusual enharmonics → canonical sharp / natural spellings.
_ENHARMONIC: dict[str, str] = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
    "Cb": "B",
    "Fb": "E",
    "E#": "F",
    "B#": "C",
}

# 16 qualities, ordered from most to less common in pop corpora.
_QUALITIES: tuple[str, ...] = (
    "maj", "min", "dim", "aug",
    "7", "maj7", "min7",
    "sus2", "sus4",
    "dim7", "hdim7", "min7b5",
    "9", "maj9", "min9", "add9",
)

# 27 (root, quality) combinations dropped from the 192-base to land on 165.
# See module docstring for the rationale.
_DROPPED_CHORDS: frozenset[tuple[str, str]] = frozenset(
    {(r, q) for r in ("C#", "D#", "F#", "G#", "A#") for q in ("9", "maj9", "min9", "add9")}
    | {(r, "dim7") for r in ("C#", "D#", "F#", "G#", "A#")}
    | {("D#", "hdim7"), ("A#", "hdim7")}
)
assert len(_DROPPED_CHORDS) == 27, "expected exactly 27 dropped chords"


def _build_vocab() -> tuple[dict[tuple[str, str], int], dict[int, tuple[str, str]]]:
    """Return (forward, reverse) maps between (root, quality) and token id.

    Token ids start at ``ChordTokenizer._FIRST_CHORD_ID`` (== 5) and increase
    in row-major order over ``_ROOTS × _QUALITIES``, skipping ``_DROPPED_CHORDS``.
    The iteration order is deterministic so checkpoints stay valid across
    process restarts.
    """
    fwd: dict[tuple[str, str], int] = {}
    rev: dict[int, tuple[str, str]] = {}
    next_id = ChordTokenizer._FIRST_CHORD_ID
    for root in _ROOTS:
        for quality in _QUALITIES:
            if (root, quality) in _DROPPED_CHORDS:
                continue
            fwd[(root, quality)] = next_id
            rev[next_id] = (root, quality)
            next_id += 1
    return fwd, rev


class ChordTokenizer:
    """Harte chord-label ↔ int token, parallel to :class:`MelodyTokenizer`."""

    # Special tokens — fixed contract; do NOT renumber.
    PAD: int = 0
    BOS: int = 1
    EOS: int = 2
    N: int = 3
    X: int = 4

    # First id available to (root, quality) chord classes.
    _FIRST_CHORD_ID: int = 5

    # Total vocab size: 5 specials + 165 chord classes.
    vocab_size: int = 170

    # Populated below the class body (needs the class to exist first).
    _FORWARD: dict[tuple[str, str], int]
    _REVERSE: dict[int, tuple[str, str]]

    def __init__(self, frame_rate_hz: float = 75.0) -> None:
        if frame_rate_hz <= 0:
            raise ValueError(f"frame_rate_hz must be > 0, got {frame_rate_hz}")
        self.frame_rate_hz: float = float(frame_rate_hz)

    # ------------------------------------------------------------------ #
    # Label parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise_root(root: str) -> str | None:
        """Canonicalise ``root`` to a sharp / natural spelling, or None if invalid."""
        if not root:
            return None
        # Capitalise: input is case-sensitive in Harte but defensively accept lower.
        root = root[0].upper() + root[1:]
        if root in _ROOTS:
            return root
        if root in _ENHARMONIC:
            return _ENHARMONIC[root]
        return None

    @staticmethod
    def _strip_bass(quality_part: str) -> str:
        """Drop the bass note from a slash chord, leaving only the quality."""
        slash = quality_part.find("/")
        if slash >= 0:
            return quality_part[:slash]
        return quality_part

    @classmethod
    def _parse_label(cls, label: str) -> tuple[str, str] | None:
        """Parse a Harte label to ``(root, quality)`` or None for N / X / malformed.

        ``"C"`` (root only, no quality) is treated as ``"C:maj"`` per the Harte
        convention; this keeps the tokenizer permissive on quick-and-dirty
        annotations that omit the trivial ``:maj`` suffix.
        """
        if label is None:
            return None
        label = label.strip()
        if not label:
            return None
        # Handle the special no-chord and unknown sentinels here so the caller
        # can still differentiate them when it wants the underlying id.
        if label == "N" or label == "X":
            return None
        if ":" in label:
            root_part, _, quality_part = label.partition(":")
        else:
            # Bare root => major triad.
            root_part, quality_part = label, "maj"
        quality_part = cls._strip_bass(quality_part)
        root = cls._normalise_root(root_part)
        if root is None:
            return None
        if quality_part not in _QUALITIES:
            return None
        return (root, quality_part)

    # ------------------------------------------------------------------ #
    # Encode / decode single labels
    # ------------------------------------------------------------------ #
    def encode(self, label: str) -> int:
        """Encode a Harte chord label to an int token id.

        ``"N"`` → :attr:`N`; ``"X"`` → :attr:`X`. Slash chords drop the bass
        (``"G:7/B"`` → ``"G:7"``). Valid-syntax labels whose ``(root, quality)``
        pair was dropped from the 165-class subset also map to :attr:`X`.
        """
        if label is None:
            return self.X
        stripped = label.strip()
        if stripped == "N":
            return self.N
        if stripped == "X" or stripped == "":
            return self.X
        parsed = self._parse_label(stripped)
        if parsed is None:
            return self.X
        return self._FORWARD.get(parsed, self.X)

    def decode(self, idx: int) -> str:
        """Inverse of :meth:`encode` — returns the canonical string for ``idx``.

        Specials decode to the strings ``"PAD"``, ``"BOS"``, ``"EOS"``, ``"N"``,
        ``"X"``. Chord ids decode to ``"<root>:<quality>"``.
        """
        idx = int(idx)
        if idx == self.PAD:
            return "PAD"
        if idx == self.BOS:
            return "BOS"
        if idx == self.EOS:
            return "EOS"
        if idx == self.N:
            return "N"
        if idx == self.X:
            return "X"
        pair = self._REVERSE.get(idx)
        if pair is None:
            raise ValueError(f"unknown chord token id {idx}; vocab_size={self.vocab_size}")
        root, quality = pair
        return f"{root}:{quality}"

    # ------------------------------------------------------------------ #
    # Sequence encode / decode (framewise)
    # ------------------------------------------------------------------ #
    def encode_sequence(
        self,
        labels: list[Chord],
        audio_duration_s: float,
        frame_rate_hz: float = 75.0,
    ) -> np.ndarray:
        """Convert a list of :class:`Chord` segments to one framewise token per frame.

        Frames are produced at ``frame_rate_hz``; the array length is
        ``int(audio_duration_s * frame_rate_hz)``. Frames not covered by any
        chord segment are filled with :attr:`N` (the explicit no-chord label),
        matching mir_eval's convention. Overlapping segments resolve by
        last-write-wins (later list entry overwrites the earlier).
        """
        if audio_duration_s < 0:
            raise ValueError(f"audio_duration_s must be >= 0, got {audio_duration_s}")
        if frame_rate_hz <= 0:
            raise ValueError(f"frame_rate_hz must be > 0, got {frame_rate_hz}")

        n_frames = int(audio_duration_s * frame_rate_hz)
        tokens = np.full(n_frames, self.N, dtype=np.int64)

        for chord in labels:
            start_frame = int(chord.onset * frame_rate_hz)
            end_frame = int((chord.onset + chord.duration) * frame_rate_hz)
            start_frame = max(0, start_frame)
            end_frame = min(n_frames, end_frame)
            if end_frame <= start_frame:
                continue
            tokens[start_frame:end_frame] = self.encode(chord.label)

        return tokens

    def decode_sequence(
        self,
        ids: np.ndarray,
        frame_rate_hz: float | None = None,
    ) -> list[Chord]:
        """Inverse of :meth:`encode_sequence` — group equal-id runs into :class:`Chord`s.

        Special tokens (PAD / BOS / EOS) are skipped (they carry no temporal
        semantics). ``N`` and ``X`` runs become :class:`Chord` segments with
        the literal labels ``"N"`` / ``"X"`` so that downstream eval can see
        them. Confidence is fixed at ``1.0`` (the tokenizer has no probability
        information); callers that have logits should overwrite it.
        """
        rate = float(frame_rate_hz) if frame_rate_hz is not None else self.frame_rate_hz
        if rate <= 0:
            raise ValueError(f"frame_rate_hz must be > 0, got {rate}")

        chords: list[Chord] = []
        if ids.size == 0:
            return chords

        n = int(ids.shape[0])
        i = 0
        while i < n:
            tok = int(ids[i])
            j = i + 1
            while j < n and int(ids[j]) == tok:
                j += 1
            if tok not in (self.PAD, self.BOS, self.EOS):
                label = self.decode(tok)
                onset = i / rate
                duration = (j - i) / rate
                chords.append(
                    Chord(label=label, onset=onset, duration=duration, confidence=1.0)
                )
            i = j
        return chords


# Build the vocab tables once at import time.
ChordTokenizer._FORWARD, ChordTokenizer._REVERSE = _build_vocab()
assert (
    len(ChordTokenizer._FORWARD)
    == ChordTokenizer.vocab_size - ChordTokenizer._FIRST_CHORD_ID
), "chord-class count must equal vocab_size minus the special-token band"


__all__ = ["ChordTokenizer"]
