"""Unit tests for the PR-4c melody trainer.

These tests are CPU-only, deterministic, and avoid all model downloads,
audio I/O, and large checkpoints. Each test runs in well under a second
on a laptop CPU; the whole module finishes in <30 s.

The trainer is exercised on a tiny ``MelodyDecoderConfig`` (a few
thousand parameters, not the ~50M PR-4a default) with synthetic
``encoder_feats`` / ``target_tokens`` tensors. PR-4d will swap in the
real :class:`HookTheoryDataset` + Demucs + MERT pipeline; the trainer
code is identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from ml.models.melody_decoder import MelodyDecoderConfig
from ml.models.melody_tokenizer import MelodyTokenizer
from ml.training.collate import melody_collate
from ml.training.trainer import MelodyTrainer, TrainConfig


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
_TINY_DECODER = dict(
    d_model=16,
    n_heads=2,
    n_layers=1,
    d_ff=32,
    dropout=0.0,
    max_seq_len=32,
    vocab_size=132,
    encoder_feat_dim=8,
)


class _SyntheticMelodyDataset(Dataset):
    """Random ``(encoder_feats, target_tokens)`` examples for the trainer.

    Token IDs are drawn from the real pitch range so the loss exercises
    the same vocabulary regions PR-4d will see.
    """

    def __init__(self, n_items: int = 4, t: int = 8, feat_dim: int = 8, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self._items: list[dict[str, torch.Tensor]] = []
        for _ in range(n_items):
            feats = torch.randn(t, feat_dim, generator=g)
            # Tokens drawn from [REST, pitch_tokens...] — never PAD/BOS/EOS,
            # which are reserved for the collate / loss machinery.
            tokens = torch.randint(
                MelodyTokenizer.REST,
                _TINY_DECODER["vocab_size"],
                (t,),
                generator=g,
                dtype=torch.int64,
            )
            self._items.append({"encoder_feats": feats, "target_tokens": tokens})

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._items[idx]


def _make_cfg(tmp_path: Path, **overrides) -> TrainConfig:
    """Construct a tiny trainer config under ``tmp_path``."""
    decoder_cfg = MelodyDecoderConfig(**_TINY_DECODER)
    defaults = dict(
        run_id="unit",
        runs_root=tmp_path,
        lr=5e-3,
        weight_decay=0.0,
        max_steps=2,
        warmup_steps=1,
        log_every_steps=1,
        eval_every_steps=0,
        save_every_steps=0,
        seed=1234,
        device="cpu",  # tests must not depend on CUDA / MPS availability
        decoder_cfg=decoder_cfg,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _make_loader(batch_size: int = 2, n_items: int = 4, seed: int = 0) -> DataLoader:
    dataset = _SyntheticMelodyDataset(n_items=n_items, t=8, feat_dim=8, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=melody_collate,
        shuffle=False,
    )


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #
def test_trainer_init_picks_device(tmp_path: Path) -> None:
    """Trainer autodetects a device and respects the explicit override."""
    cfg = _make_cfg(tmp_path)
    trainer = MelodyTrainer(cfg, train_loader=_make_loader())

    # With device="cpu" in cfg we expect a CPU device.
    assert trainer.device.type == "cpu"

    # The fallback path must produce one of the three supported devices
    # even when no explicit preference is provided.
    cfg_auto = _make_cfg(tmp_path, device=None, run_id="auto")
    trainer_auto = MelodyTrainer(cfg_auto, train_loader=_make_loader())
    assert trainer_auto.device.type in {"cuda", "mps", "cpu"}

    # Run directory + metrics path should exist after init.
    assert trainer.run_dir.is_dir()


def test_train_step_returns_loss(tmp_path: Path) -> None:
    """A single ``train_step`` returns the expected metric dict."""
    cfg = _make_cfg(tmp_path)
    trainer = MelodyTrainer(cfg, train_loader=_make_loader())
    batch = next(iter(_make_loader()))

    metrics = trainer.train_step(batch)

    assert set(metrics) >= {"loss", "lr", "step", "grad_norm"}
    assert isinstance(metrics["loss"], float)
    assert metrics["loss"] > 0  # CE on a fresh init is well above 0
    assert metrics["step"] == 1
    assert metrics["lr"] > 0


def test_two_steps_decrease_loss_on_overfit(tmp_path: Path) -> None:
    """Repeated steps on the same batch must reduce loss (overfit sanity)."""
    cfg = _make_cfg(tmp_path, lr=1e-2, warmup_steps=0)
    trainer = MelodyTrainer(cfg, train_loader=_make_loader())

    # Re-use the same batch — the model should be able to memorise it.
    batch = next(iter(_make_loader(batch_size=2, n_items=2)))

    first = trainer.train_step(batch)["loss"]
    for _ in range(30):
        trainer.train_step(batch)
    final = trainer.train_step(batch)["loss"]

    assert final < first, f"loss did not decrease after 30 steps: {first:.4f} -> {final:.4f}"


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    """save_checkpoint -> fresh trainer -> load_checkpoint restores state."""
    cfg = _make_cfg(tmp_path)
    trainer = MelodyTrainer(cfg, train_loader=_make_loader())

    # Mutate state with a couple of steps so save/load is meaningful.
    for batch in _make_loader():
        trainer.train_step(batch)

    ckpt_path = trainer.save_checkpoint(trainer.step)
    assert ckpt_path.exists()
    assert ckpt_path.name == f"step-{trainer.step}.pt"

    expected_step = trainer.step
    expected_state = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}

    fresh_cfg = _make_cfg(tmp_path, run_id="fresh")
    fresh = MelodyTrainer(fresh_cfg, train_loader=_make_loader())
    loaded_step = fresh.load_checkpoint(ckpt_path)

    assert loaded_step == expected_step
    assert fresh.step == expected_step

    loaded_state = fresh.model.state_dict()
    assert loaded_state.keys() == expected_state.keys()
    for k, v in expected_state.items():
        assert torch.allclose(loaded_state[k], v), f"param {k!r} did not round-trip"


def test_evaluate_returns_metrics(tmp_path: Path) -> None:
    """``evaluate`` reports the documented metric keys."""
    cfg = _make_cfg(tmp_path)
    trainer = MelodyTrainer(cfg, train_loader=_make_loader())
    val_loader = _make_loader(batch_size=2, n_items=4, seed=1)

    metrics = trainer.evaluate(val_loader)

    assert set(metrics) >= {"loss", "top1", "top1_non_rest"}
    assert metrics["loss"] > 0
    assert 0.0 <= metrics["top1"] <= 1.0
    assert 0.0 <= metrics["top1_non_rest"] <= 1.0


def test_metrics_jsonl_written(tmp_path: Path) -> None:
    """``train()`` populates ``metrics.jsonl`` with valid JSON lines."""
    cfg = _make_cfg(tmp_path, max_steps=3, log_every_steps=1)
    trainer = MelodyTrainer(cfg, train_loader=_make_loader())
    trainer.train()

    metrics_path = trainer.run_dir / "metrics.jsonl"
    assert metrics_path.exists(), "metrics.jsonl was not created by train()"

    lines = [ln for ln in metrics_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, "metrics.jsonl is empty after a successful run"

    records = [json.loads(ln) for ln in lines]
    train_records = [r for r in records if r.get("event") == "train"]
    assert train_records, "no train-event records were logged"
    for r in train_records:
        assert "step" in r
        assert "loss" in r
