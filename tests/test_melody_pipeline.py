"""Unit tests for the PR-4b melody training data pipeline.

Only the alignment-invariant smoke test runs the real wrappers (which
need Demucs + MERT checkpoints). Collation, caching, and length-filter
tests use lightweight fakes so they pass on CPU-only CI without
``RUN_SLOW=1``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ml.datasets.hooktheory import HookTheoryDataset
from ml.models.melody_tokenizer import MelodyTokenizer
from ml.training.collate import melody_collate
from ml.training.datasets import MelodyTrainingDataset
from ml.training.melody_pipeline import (
    MelodyDataPipeline,
    MelodyTrainingExample,
)
from ml.types import Note


# ---------------------------------------------------------------------- #
# In-memory fakes
# ---------------------------------------------------------------------- #
class _FakeHookTheoryDataset(HookTheoryDataset):
    """HookTheoryDataset that yields entries from an in-memory list.

    Skips the on-disk manifest read so tests don't need a JSONL file.
    """

    def __init__(self, entries: list[dict[str, Any]]) -> None:  # noqa: D401 - test helper
        # Intentionally bypass the parent constructor — it requires a manifest
        # file on disk. We satisfy the base class API by populating the same
        # internal attributes the parent uses.
        self.manifest_path = Path("/dev/null")
        self.split = "train"
        self._entries = entries


class _RecordingPipeline:
    """Pipeline stand-in that records every ``prepare`` call.

    Returns a deterministic :class:`MelodyTrainingExample` so the dataset
    can serialise / deserialise it via the cache layer.
    """

    def __init__(self, t_enc: int = 4, feat_dim: int = 1024) -> None:
        self.t_enc = t_enc
        self.feat_dim = feat_dim
        self.calls: list[tuple[str, int]] = []

    def prepare(
        self, audio_path: str | Path, notes: list[Note]
    ) -> MelodyTrainingExample:
        self.calls.append((str(audio_path), len(notes)))
        return MelodyTrainingExample(
            audio_id=Path(str(audio_path)).stem,
            encoder_feats=np.ones((self.t_enc, self.feat_dim), dtype=np.float32),
            target_tokens=np.full((self.t_enc,), 7, dtype=np.int64),
            audio_duration_s=float(self.t_enc) / 75.0,
        )


# ---------------------------------------------------------------------- #
# Lazy-construction
# ---------------------------------------------------------------------- #
def test_pipeline_init_is_lazy() -> None:
    """Constructing the pipeline must not load Demucs or MERT weights."""
    pipeline = MelodyDataPipeline()

    assert pipeline.demucs._model is None
    assert pipeline.mert._model is None
    # Tokenizer is pure-Python; just sanity-check it was wired.
    assert isinstance(pipeline.tokenizer, MelodyTokenizer)
    assert pipeline.tokenizer.frame_rate_hz == pytest.approx(75.0)
    assert pipeline.use_vocals_stem is True


# ---------------------------------------------------------------------- #
# Alignment invariant on real models (slow)
# ---------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="Slow test: downloads Demucs + MERT checkpoints. Set RUN_SLOW=1 to enable.",
)
def test_prepare_alignment_invariant(tmp_path: Path) -> None:
    """End-to-end smoke: encoder_feats and target_tokens share length T_enc."""
    import soundfile as sf

    from ml.models.demucs import DEMUCS_SR

    sr = DEMUCS_SR
    duration_s = 3.0
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    mono = (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1)
    wav_path = tmp_path / "synthetic.wav"
    sf.write(str(wav_path), stereo, sr, subtype="FLOAT")

    pipeline = MelodyDataPipeline()
    notes = [Note(pitch=69, onset=0.5, duration=1.0)]
    example = pipeline.prepare(wav_path, notes)

    assert example.encoder_feats.ndim == 2
    assert example.encoder_feats.shape[1] == 1024
    assert example.target_tokens.ndim == 1
    assert example.encoder_feats.shape[0] == example.target_tokens.shape[0]
    assert example.audio_duration_s == pytest.approx(duration_s, rel=0.05)


# ---------------------------------------------------------------------- #
# Collate
# ---------------------------------------------------------------------- #
def test_collate_padding() -> None:
    """Pads to longest length; encoder pad=0.0, token pad=PAD (==0)."""
    feat_dim = 1024
    items = [
        {
            "encoder_feats": torch.full((3, feat_dim), 1.5),
            "target_tokens": torch.tensor([10, 11, 12], dtype=torch.int64),
        },
        {
            "encoder_feats": torch.full((5, feat_dim), 2.5),
            "target_tokens": torch.tensor([20, 21, 22, 23, 24], dtype=torch.int64),
        },
        {
            "encoder_feats": torch.full((2, feat_dim), 3.5),
            "target_tokens": torch.tensor([30, 31], dtype=torch.int64),
        },
    ]

    out = melody_collate(items)

    assert out["encoder_feats"].shape == (3, 5, feat_dim)
    assert out["target_tokens"].shape == (3, 5)
    assert out["lengths"].tolist() == [3, 5, 2]

    # Token PAD region must be exactly MelodyTokenizer.PAD (== 0).
    assert MelodyTokenizer.PAD == 0
    # Item 0: valid frames [0:3] keep their values; [3:5] are PAD.
    assert torch.equal(out["target_tokens"][0, :3], torch.tensor([10, 11, 12]))
    assert torch.all(out["target_tokens"][0, 3:] == MelodyTokenizer.PAD)
    # Item 2: only first 2 frames are valid.
    assert torch.equal(out["target_tokens"][2, :2], torch.tensor([30, 31]))
    assert torch.all(out["target_tokens"][2, 2:] == MelodyTokenizer.PAD)

    # Encoder pad region must be exactly 0.0.
    assert torch.all(out["encoder_feats"][0, 3:] == 0.0)
    assert torch.all(out["encoder_feats"][2, 2:] == 0.0)
    # And valid regions still carry the original fill values.
    assert torch.all(out["encoder_feats"][0, :3] == 1.5)
    assert torch.all(out["encoder_feats"][1] == 2.5)


# ---------------------------------------------------------------------- #
# Dataset caching + length filter
# ---------------------------------------------------------------------- #
def test_dataset_cache_roundtrip(tmp_path: Path) -> None:
    """Second-construction with the same cache_dir must short-circuit prepare()."""
    underlying = _FakeHookTheoryDataset(
        [
            {
                "id": "song-001",
                "audio_path": "/synthetic/song-001.wav",
                "audio_duration_s": 2.0,
                "notes": [Note(pitch=60, onset=0.0, duration=1.0)],
            }
        ]
    )

    first_pipeline = _RecordingPipeline(t_enc=6, feat_dim=8)
    first_ds = MelodyTrainingDataset(
        underlying=underlying,
        pipeline=first_pipeline,  # type: ignore[arg-type]
        cache_dir=tmp_path,
        max_duration_s=30.0,
    )
    item0 = first_ds[0]
    assert item0["encoder_feats"].shape == (6, 8)
    assert item0["target_tokens"].shape == (6,)
    assert torch.all(item0["target_tokens"] == 7)
    assert len(first_pipeline.calls) == 1
    cache_file = tmp_path / "song-001.melody_train.npz"
    assert cache_file.exists()

    class _ExplodingPipeline:
        def prepare(self, *args: object, **kwargs: object) -> MelodyTrainingExample:
            raise AssertionError(
                "pipeline.prepare must not be called on cache hit"
            )

    second_ds = MelodyTrainingDataset(
        underlying=underlying,
        pipeline=_ExplodingPipeline(),  # type: ignore[arg-type]
        cache_dir=tmp_path,
        max_duration_s=30.0,
    )
    item1 = second_ds[0]
    assert item1["encoder_feats"].shape == (6, 8)
    assert torch.allclose(item1["encoder_feats"], item0["encoder_feats"])
    assert torch.equal(item1["target_tokens"], item0["target_tokens"])


def test_skip_long_examples() -> None:
    """Entries longer than max_duration_s are excluded from __len__/__getitem__."""
    underlying = _FakeHookTheoryDataset(
        [
            {
                "id": "short",
                "audio_path": "/synthetic/short.wav",
                "audio_duration_s": 5.0,
                "notes": [],
            },
            {
                "id": "too-long",
                "audio_path": "/synthetic/too-long.wav",
                "audio_duration_s": 90.0,
                "notes": [],
            },
            {
                "id": "borderline",
                "audio_path": "/synthetic/borderline.wav",
                "audio_duration_s": 30.0,
                "notes": [],
            },
        ]
    )

    pipeline = _RecordingPipeline(t_enc=2, feat_dim=4)
    ds = MelodyTrainingDataset(
        underlying=underlying,
        pipeline=pipeline,  # type: ignore[arg-type]
        cache_dir=None,
        max_duration_s=30.0,
    )

    assert len(ds) == 2
    seen_audio_paths = {ds[i] is not None for i in range(len(ds))}
    assert all(seen_audio_paths)
    # Both calls land on the two surviving entries (short + borderline);
    # never on the 90 s one.
    called_paths = [p for p, _ in pipeline.calls]
    assert "/synthetic/too-long.wav" not in called_paths
    assert "/synthetic/short.wav" in called_paths
    assert "/synthetic/borderline.wav" in called_paths
