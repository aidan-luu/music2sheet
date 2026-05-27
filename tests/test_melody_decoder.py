"""Unit tests for the PR-4a melody decoder + tokenizer.

These tests are CPU-only, deterministic, and avoid all model downloads
or audio I/O so they run by default (no ``RUN_SLOW=1`` required).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.models.melody_decoder import MelodyDecoder, MelodyDecoderConfig
from ml.models.melody_tokenizer import MelodyTokenizer
from ml.types import Note


# ---------------------------------------------------------------------- #
# Decoder tests
# ---------------------------------------------------------------------- #
def test_config_defaults() -> None:
    cfg = MelodyDecoderConfig()
    assert cfg.d_model == 512
    assert cfg.n_heads == 8
    assert cfg.n_layers == 12
    assert cfg.d_ff == 2048
    assert cfg.dropout == pytest.approx(0.1)
    assert cfg.max_seq_len == 2048
    assert cfg.vocab_size == 132
    assert cfg.encoder_feat_dim == 1024


def test_model_param_count() -> None:
    cfg = MelodyDecoderConfig()
    model = MelodyDecoder(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    assert 40_000_000 <= n_params <= 60_000_000, (
        f"expected ~50M params, got {n_params:,}"
    )


def test_forward_shape() -> None:
    torch.manual_seed(0)
    cfg = MelodyDecoderConfig()
    model = MelodyDecoder(cfg).eval()

    encoder_feats = torch.randn(2, 100, cfg.encoder_feat_dim)
    target_tokens = torch.randint(0, cfg.vocab_size, (2, 200))

    with torch.no_grad():
        logits = model(encoder_feats, target_tokens)

    assert logits.shape == (2, 200, cfg.vocab_size)


def test_generate_shape() -> None:
    torch.manual_seed(0)
    cfg = MelodyDecoderConfig()
    model = MelodyDecoder(cfg).eval()

    encoder_feats = torch.randn(1, 50, cfg.encoder_feat_dim)

    with torch.no_grad():
        out = model.generate(encoder_feats, max_new_tokens=128)

    assert out.dim() == 2
    assert out.shape[0] == 1
    assert 1 <= out.shape[1] <= 128


# ---------------------------------------------------------------------- #
# Tokenizer tests
# ---------------------------------------------------------------------- #
def test_tokenizer_roundtrip() -> None:
    tokenizer = MelodyTokenizer(frame_rate_hz=75.0)
    notes = [
        Note(pitch=60, onset=0.0, duration=0.5),
        Note(pitch=62, onset=0.5, duration=0.25),
        Note(pitch=64, onset=0.8, duration=0.4),
        Note(pitch=65, onset=1.3, duration=0.2),
        Note(pitch=67, onset=1.6, duration=0.6),
    ]
    audio_duration_s = 2.5

    token_ids = tokenizer.encode(notes, audio_duration_s)
    decoded = tokenizer.decode(token_ids)

    assert len(decoded) == len(notes), (
        f"expected {len(notes)} notes after round-trip, got {len(decoded)}"
    )

    # 1-frame tolerance on onsets / durations; exact match on pitch.
    frame_s = 1.0 / tokenizer.frame_rate_hz
    for orig, got in zip(notes, decoded):
        assert orig.pitch == got.pitch
        assert abs(orig.onset - got.onset) <= frame_s + 1e-9
        assert abs(orig.duration - got.duration) <= frame_s + 1e-9


def test_tokenizer_rests() -> None:
    tokenizer = MelodyTokenizer(frame_rate_hz=75.0)
    token_ids = tokenizer.encode([], audio_duration_s=2.0)

    assert token_ids.shape == (int(2.0 * 75),)
    assert token_ids.shape == (150,)
    assert np.all(token_ids == tokenizer.REST)
