"""Melody decoder training loop (PR-4c — scaffolding, no real run).

This module provides :class:`MelodyTrainer`, the framework PR-4d will
point at real HookTheory data + a GPU. PR-4c only needs:

* A working forward / backward step on synthetic tensors.
* Optimizer (AdamW) + linear-warmup scheduler wired correctly.
* Checkpointing that round-trips model / optimizer / scheduler state.
* JSONL metrics log under ``runs_root/run_id/``.
* Device autodetect (cuda → mps → cpu).

The model architecture (:class:`ml.models.melody_decoder.MelodyDecoder`)
and the tokenizer (:class:`ml.models.melody_tokenizer.MelodyTokenizer`)
are stable contracts from PR-4a and are NOT modified here.

Teacher-forcing scheme
----------------------
The data pipeline (PR-4b) produces framewise targets of length
``T_enc`` aligned 1-to-1 with the encoder features. For teacher
forcing we shift-right with a BOS prefix and drop the last position::

    encoder_feats: (B, T, D)
    target_tokens: (B, T)
    decoder_input = [BOS, target_tokens[:, 0], ..., target_tokens[:, T-2]]
    logits = model(encoder_feats, decoder_input)   # (B, T, V)
    loss   = CE(logits, target_tokens, ignore_index=PAD, smoothing=0.1)

Padding positions (PAD == 0) are ignored by both the loss and the
top-1 / non-REST top-1 metrics. A ``tgt_key_padding_mask`` derived
from ``lengths`` is passed to the decoder so cross-attention also
skips padded query positions.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ml.models.melody_decoder import MelodyDecoder, MelodyDecoderConfig
from ml.models.melody_tokenizer import MelodyTokenizer


# ---------------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Hyperparameters + IO layout for :class:`MelodyTrainer`.

    All fields are JSON-serialisable so the full config is persisted
    inside every checkpoint as ``cfg_dict``.
    """

    run_id: str = "smoke"
    runs_root: Path = Path("runs")

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.1

    # Schedule
    max_steps: int = 1000
    warmup_steps: int = 100

    # Logging / IO cadence
    log_every_steps: int = 10
    eval_every_steps: int = 200
    save_every_steps: int = 500

    # Reproducibility / device
    seed: int = 42
    device: str | None = None  # None -> autodetect

    # Model
    decoder_cfg: MelodyDecoderConfig = field(default_factory=MelodyDecoderConfig)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view of the config (Paths -> str, dataclasses -> dict)."""
        out = asdict(self)
        out["runs_root"] = str(self.runs_root)
        return out


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _pick_device(preference: str | None = None) -> torch.device:
    """Autodetect the best available device: cuda → mps → cpu.

    An explicit ``preference`` string short-circuits the search and is
    returned verbatim (callers are responsible for ensuring it's valid).
    """
    if preference is not None:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _linear_warmup_lambda(warmup_steps: int):
    """``LambdaLR`` callable: 0 → 1 over ``warmup_steps``, then constant 1."""
    if warmup_steps <= 0:
        return lambda step: 1.0
    return lambda step: min(1.0, float(step + 1) / float(warmup_steps))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _key_padding_mask(lengths: torch.Tensor, t_max: int) -> torch.Tensor:
    """Bool mask of shape ``(B, T_max)``; ``True`` where the position is padding."""
    arange = torch.arange(t_max, device=lengths.device).unsqueeze(0)
    return arange >= lengths.unsqueeze(1)


def _cycle(loader: Iterable[dict[str, torch.Tensor]]) -> Iterator[dict[str, torch.Tensor]]:
    """Iterate ``loader`` forever, restarting each epoch."""
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------- #
# Trainer
# ---------------------------------------------------------------------- #
class MelodyTrainer:
    """Drives the melody decoder through a step-based training loop.

    The trainer is intentionally framework-light (no Lightning / Accelerate)
    so the control flow is auditable and the same class can be reused in
    notebooks, the CLI in :mod:`ml.training.cli`, and the unit tests in
    ``tests/test_melody_trainer.py``.
    """

    PAD: int = MelodyTokenizer.PAD
    BOS: int = MelodyTokenizer.BOS
    REST: int = MelodyTokenizer.REST

    def __init__(
        self,
        cfg: TrainConfig,
        train_loader: DataLoader | Iterable[dict[str, torch.Tensor]],
        val_loader: DataLoader | Iterable[dict[str, torch.Tensor]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader

        _set_seed(cfg.seed)

        self.device = _pick_device(cfg.device)
        self.model = MelodyDecoder(cfg.decoder_cfg).to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=_linear_warmup_lambda(cfg.warmup_steps),
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.PAD,
            label_smoothing=cfg.label_smoothing,
        )

        self.run_dir = Path(cfg.runs_root) / cfg.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"

        self.step: int = 0

    # ------------------------------------------------------------------ #
    # Forward helpers
    # ------------------------------------------------------------------ #
    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move tensors to the trainer's device. Returns a *new* dict."""
        out: dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _decoder_input_from_targets(self, target_tokens: torch.Tensor) -> torch.Tensor:
        """Prepend BOS, drop last position. Shape preserved as ``(B, T)``."""
        b, t = target_tokens.shape
        bos_col = torch.full((b, 1), self.BOS, dtype=target_tokens.dtype, device=target_tokens.device)
        if t == 0:
            return bos_col[:, :0]
        return torch.cat([bos_col, target_tokens[:, :-1]], dim=1)

    def _forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a teacher-forced forward pass; return ``(logits, targets)``.

        Both tensors are flat in the time dimension only at the loss site;
        here they are returned as ``(B, T, V)`` and ``(B, T)`` so callers
        (loss vs. metric) can pool them differently.
        """
        encoder_feats = batch["encoder_feats"]
        target_tokens = batch["target_tokens"]
        lengths = batch.get("lengths")

        decoder_input = self._decoder_input_from_targets(target_tokens)
        if lengths is not None:
            key_padding_mask = _key_padding_mask(lengths, target_tokens.size(1))
        else:
            key_padding_mask = None

        logits = self.model(
            encoder_feats=encoder_feats,
            target_tokens=decoder_input,
            target_mask=key_padding_mask,
        )
        return logits, target_tokens

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab = logits.size(-1)
        return self.criterion(logits.reshape(-1, vocab), targets.reshape(-1))

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        """One forward + backward + optimizer step on ``batch``.

        Returns
        -------
        dict
            ``{"loss": float, "lr": float, "step": int, "grad_norm": float}``.
        """
        self.model.train()
        batch = self._prepare_batch(batch)

        self.optimizer.zero_grad(set_to_none=True)
        logits, targets = self._forward(batch)
        loss = self._compute_loss(logits, targets)
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.cfg.grad_clip_norm
            )
        )
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1

        return {
            "loss": float(loss.detach().cpu().item()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "step": int(self.step),
            "grad_norm": grad_norm,
        }

    @torch.no_grad()
    def evaluate(self, loader: Iterable[dict[str, torch.Tensor]]) -> dict[str, float]:
        """Compute cross-entropy + top-1 + non-REST top-1 over ``loader``.

        All metrics ignore padding (``PAD == 0``). The non-REST variant
        further excludes positions whose ground-truth is ``REST`` —
        important because the framewise stream is REST-heavy and a model
        that predicts REST everywhere can score deceptively well.
        """
        self.model.eval()

        total_loss_sum = 0.0
        total_loss_count = 0
        total_correct = 0
        total_count = 0
        non_rest_correct = 0
        non_rest_count = 0

        # Per-position CE for clean averaging across variable-length batches.
        ce_per_pos = nn.CrossEntropyLoss(
            ignore_index=self.PAD,
            label_smoothing=self.cfg.label_smoothing,
            reduction="sum",
        )

        for batch in loader:
            batch = self._prepare_batch(batch)
            logits, targets = self._forward(batch)
            vocab = logits.size(-1)

            flat_logits = logits.reshape(-1, vocab)
            flat_targets = targets.reshape(-1)
            valid = flat_targets != self.PAD

            loss_sum = ce_per_pos(flat_logits, flat_targets)
            n_valid = int(valid.sum().item())
            total_loss_sum += float(loss_sum.item())
            total_loss_count += n_valid

            preds = flat_logits.argmax(dim=-1)
            correct = (preds == flat_targets) & valid
            total_correct += int(correct.sum().item())
            total_count += n_valid

            non_rest = valid & (flat_targets != self.REST)
            non_rest_correct += int((correct & non_rest).sum().item())
            non_rest_count += int(non_rest.sum().item())

        def _safe_div(a: float, b: float) -> float:
            return float(a) / float(b) if b > 0 else 0.0

        return {
            "loss": _safe_div(total_loss_sum, total_loss_count),
            "top1": _safe_div(total_correct, total_count),
            "top1_non_rest": _safe_div(non_rest_correct, non_rest_count),
            "n_positions": float(total_count),
            "n_non_rest_positions": float(non_rest_count),
        }

    def train(self) -> dict[str, Any]:
        """Run the full loop until ``cfg.max_steps`` is reached.

        Returns a small summary dict (final step, last loss, last eval).
        Detailed per-step metrics are persisted to
        ``runs_root/run_id/metrics.jsonl``.
        """
        loader_iter = _cycle(self.train_loader)
        last_loss: float = float("nan")
        last_eval: dict[str, float] | None = None
        t0 = time.time()

        while self.step < self.cfg.max_steps:
            batch = next(loader_iter)
            metrics = self.train_step(batch)
            last_loss = metrics["loss"]

            if self.cfg.log_every_steps and self.step % self.cfg.log_every_steps == 0:
                self._log({**metrics, "event": "train", "elapsed_s": time.time() - t0})

            if (
                self.val_loader is not None
                and self.cfg.eval_every_steps
                and self.step % self.cfg.eval_every_steps == 0
            ):
                last_eval = self.evaluate(self.val_loader)
                self._log({**last_eval, "event": "eval", "step": self.step})

            if self.cfg.save_every_steps and self.step % self.cfg.save_every_steps == 0:
                self.save_checkpoint(self.step)

        # Always log the final state once, even if log_every_steps would
        # skip it — otherwise short runs (max_steps < log_every_steps) end
        # with an empty metrics file.
        self._log(
            {
                "event": "train",
                "step": self.step,
                "loss": last_loss,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "elapsed_s": time.time() - t0,
            }
        )

        return {
            "final_step": self.step,
            "last_loss": last_loss,
            "last_eval": last_eval,
            "run_dir": str(self.run_dir),
        }

    # ------------------------------------------------------------------ #
    # Checkpoints
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, step: int) -> Path:
        """Persist ``{step, model, optimizer, scheduler, cfg_dict}`` to disk.

        File name is ``step-<step>.pt`` under the run directory. Returns
        the resulting :class:`Path`.
        """
        path = self.run_dir / f"step-{step}.pt"
        payload = {
            "step": int(step),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "cfg_dict": self.cfg.to_dict(),
        }
        # Atomic write so a crash mid-save doesn't leave a partial file.
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def load_checkpoint(self, path: str | Path) -> int:
        """Restore model / optimizer / scheduler / step from ``path``.

        Returns the step recorded in the checkpoint.
        """
        path = Path(path)
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.scheduler.load_state_dict(payload["scheduler_state"])
        self.step = int(payload["step"])
        return self.step

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def _log(self, record: dict[str, Any]) -> None:
        """Append one JSON line to ``metrics.jsonl``."""
        record = dict(record)
        record.setdefault("step", self.step)
        # NaN / Inf are not valid JSON; coerce them so the file always parses.
        for k, v in list(record.items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                record[k] = None
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


__all__ = ["MelodyTrainer", "TrainConfig"]
