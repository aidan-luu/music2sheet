"""Fast CPU tests for the PR-6 chord training data pipeline.

Mirrors the structure of ``tests/test_melody_pipeline.py``: only the
lazy-init check and the cache short-circuit are exercised here. The
end-to-end audio → MERT features path is PR-7 / PR-D-FUTURE territory and
needs ``RUN_SLOW=1`` to bring in real model weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ml.datasets.isophonics import IsophonicsDataset
from ml.models.chord_tokenizer import ChordTokenizer
from ml.training.chord_data import (
    ChordDataPipeline,
    ChordTrainingDataset,
    ChordTrainingExample,
)
from ml.types import Chord


# ---------------------------------------------------------------------- #
# In-memory fakes
# ---------------------------------------------------------------------- #
class _FakeIsophonicsDataset(IsophonicsDataset):
    """IsophonicsDataset that yields entries from an in-memory list.

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
    """Pipeline stand-in that records every ``prepare`` call and returns
    a deterministic :class:`ChordTrainingExample`.
    """

    def __init__(self, t_enc: int = 4, feat_dim: int = 1024) -> None:
        self.t_enc = t_enc
        self.feat_dim = feat_dim
        self.calls: list[tuple[str, int]] = []

    def prepare(
        self, audio_path: str | Path, chords: list[Chord]
    ) -> ChordTrainingExample:
        self.calls.append((str(audio_path), len(chords)))
        return ChordTrainingExample(
            audio_id=Path(str(audio_path)).stem,
            encoder_feats=np.ones((self.t_enc, self.feat_dim), dtype=np.float32),
            # Use a non-special id so cache hits can be distinguished from
            # accidental zeros.
            target_tokens=np.full((self.t_enc,), 11, dtype=np.int64),
            audio_duration_s=float(self.t_enc) / 75.0,
        )


# ---------------------------------------------------------------------- #
# Lazy-construction
# ---------------------------------------------------------------------- #
def test_chord_pipeline_init_is_lazy() -> None:
    """Constructing the pipeline must not load MERT weights."""
    pipeline = ChordDataPipeline()

    # MERT wrapper holds the model lazily; verify nothing was loaded.
    assert pipeline.mert._model is None
    assert pipeline.mert._processor is None
    # Tokenizer is pure-Python; just sanity-check it was wired.
    assert isinstance(pipeline.tokenizer, ChordTokenizer)
    assert pipeline.tokenizer.frame_rate_hz == pytest.approx(75.0)
    assert pipeline.tokenizer.vocab_size == 170


# ---------------------------------------------------------------------- #
# Dataset caching + length filter
# ---------------------------------------------------------------------- #
def test_chord_dataset_cache_roundtrip(tmp_path: Path) -> None:
    """Second-construction with the same cache_dir must short-circuit prepare()."""
    underlying = _FakeIsophonicsDataset(
        [
            {
                "id": "track-001",
                "audio_path": "/synthetic/track-001.wav",
                "audio_duration_s": 2.0,
                "chords": [
                    Chord(label="C:maj", onset=0.0, duration=1.0, confidence=1.0),
                ],
            }
        ]
    )

    first_pipeline = _RecordingPipeline(t_enc=6, feat_dim=8)
    first_ds = ChordTrainingDataset(
        underlying=underlying,
        pipeline=first_pipeline,  # type: ignore[arg-type]
        cache_dir=tmp_path,
        max_duration_s=30.0,
    )
    item0 = first_ds[0]
    assert item0["encoder_feats"].shape == (6, 8)
    assert item0["target_tokens"].shape == (6,)
    assert torch.all(item0["target_tokens"] == 11)
    assert len(first_pipeline.calls) == 1
    cache_file = tmp_path / "track-001.chord_train.npz"
    assert cache_file.exists()

    class _ExplodingPipeline:
        def prepare(self, *args: object, **kwargs: object) -> ChordTrainingExample:
            raise AssertionError(
                "pipeline.prepare must not be called on cache hit"
            )

    second_ds = ChordTrainingDataset(
        underlying=underlying,
        pipeline=_ExplodingPipeline(),  # type: ignore[arg-type]
        cache_dir=tmp_path,
        max_duration_s=30.0,
    )
    item1 = second_ds[0]
    assert item1["encoder_feats"].shape == (6, 8)
    assert torch.allclose(item1["encoder_feats"], item0["encoder_feats"])
    assert torch.equal(item1["target_tokens"], item0["target_tokens"])
