"""Fast CPU tests for the PR-6 chord head + Harte tokenizer.

All tests in this file are pure-Python / pure-torch CPU work and combined
should finish well under the 30 s envelope the PR sets.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.models.chord_head import ChordHead, ChordHeadConfig
from ml.models.chord_tokenizer import ChordTokenizer
from ml.types import Chord


# ---------------------------------------------------------------------- #
# ChordTokenizer
# ---------------------------------------------------------------------- #
def test_tokenizer_vocab_size() -> None:
    """The locked vocab size must be exactly 170 (5 specials + 165 chord classes)."""
    tok = ChordTokenizer()
    assert tok.vocab_size == 170
    # Specials are immutable contract.
    assert (tok.PAD, tok.BOS, tok.EOS, tok.N, tok.X) == (0, 1, 2, 3, 4)
    # Forward / reverse maps cover the 165 (root, quality) chord ids.
    assert len(ChordTokenizer._FORWARD) == 165
    assert len(ChordTokenizer._REVERSE) == 165
    # IDs land exactly in [5, 169].
    chord_ids = set(ChordTokenizer._REVERSE.keys())
    assert min(chord_ids) == 5
    assert max(chord_ids) == 169


def test_tokenizer_roundtrip() -> None:
    """encode → decode is identity on simple labels; slash chord drops the bass."""
    tok = ChordTokenizer()

    # Plain (root, quality) cases — exact roundtrip.
    for label in ("C:maj", "A:min7", "F:7", "B:dim", "G:sus4"):
        idx = tok.encode(label)
        assert idx >= ChordTokenizer._FIRST_CHORD_ID, f"{label} should be a chord id"
        assert tok.decode(idx) == label

    # No-chord and unknown sentinels.
    assert tok.encode("N") == tok.N
    assert tok.decode(tok.N) == "N"
    assert tok.encode("X") == tok.X
    assert tok.decode(tok.X) == "X"

    # Slash chord collapses to the root form.
    slash_idx = tok.encode("G:7/B")
    root_idx = tok.encode("G:7")
    assert slash_idx == root_idx
    assert tok.decode(slash_idx) == "G:7"

    # Flat enharmonic spellings normalise to sharps.
    assert tok.encode("Bb:maj") == tok.encode("A#:maj")
    assert tok.encode("Eb:min7") == tok.encode("D#:min7")


def test_tokenizer_sequence_alignment() -> None:
    """3 chords across 2 s @ 75 Hz → shape (150,) with N filling the gap."""
    tok = ChordTokenizer()
    chords = [
        Chord(label="C:maj", onset=0.0, duration=0.8, confidence=1.0),
        Chord(label="A:min", onset=0.8, duration=0.4, confidence=1.0),
        # gap [1.2, 1.6) → N
        Chord(label="G:7", onset=1.6, duration=0.4, confidence=1.0),
    ]

    frames = tok.encode_sequence(chords, audio_duration_s=2.0, frame_rate_hz=75.0)
    assert frames.shape == (150,)

    c_id = tok.encode("C:maj")
    am_id = tok.encode("A:min")
    g7_id = tok.encode("G:7")

    # Per-frame spot checks bracketing each boundary.
    assert frames[0] == c_id
    assert frames[59] == c_id           # 0.8 s * 75 = 60 → last C frame is 59
    assert frames[60] == am_id
    assert frames[89] == am_id          # (0.8 + 0.4) * 75 = 90 → last Am frame is 89
    # Gap [90, 120) is N (no chord).
    assert np.all(frames[90:120] == tok.N)
    assert frames[120] == g7_id
    assert frames[149] == g7_id


# ---------------------------------------------------------------------- #
# ChordHead
# ---------------------------------------------------------------------- #
def test_chord_head_forward_shape() -> None:
    """(2, 100, 1024) MERT features → (2, 100, 170) logits."""
    head = ChordHead(ChordHeadConfig())
    head.eval()  # disable dropout so the test is deterministic.

    feats = torch.randn(2, 100, 1024)
    logits = head(feats)

    assert logits.shape == (2, 100, 170)
    assert logits.dtype == feats.dtype
    assert torch.isfinite(logits).all(), "head produced NaN/Inf for finite input"


def test_chord_head_param_count() -> None:
    """Parameter count stays in the [0.5 M, 5 M] target envelope."""
    head = ChordHead(ChordHeadConfig())
    n_params = sum(p.numel() for p in head.parameters())
    assert 500_000 <= n_params <= 5_000_000, (
        f"ChordHead has {n_params:,} params; expected between 0.5 M and 5 M. "
        "Adjust ChordHeadConfig defaults if the architecture changed."
    )
