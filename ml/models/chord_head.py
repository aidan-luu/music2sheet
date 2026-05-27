"""Frame-level chord classifier head (PR-6 — model only, no training).

A compact MLP that consumes MERT encoder features and emits per-frame
logits over the 170-class :class:`ChordTokenizer` vocabulary. The
architecture is intentionally simple — the heavy lifting on chord
recognition happens in PR-7 (BACHI boundary-aware refinement on top of
these per-frame predictions).

Architecture
------------
``encoder_feats (B, T, encoder_feat_dim)`` →
    ``Linear(encoder_feat_dim, hidden_dim) → ReLU → Dropout``
        →  (``n_layers - 1``) hidden blocks of
            ``Linear(hidden_dim, hidden_dim) → ReLU → Dropout``
    → ``Linear(hidden_dim, vocab_size)``  →  ``logits (B, T, vocab_size)``

The MLP is applied **independently per frame** (no temporal mixing). This
is the standard "deep chroma" frame classifier baseline; PR-7 plugs a
boundary-aware decoder on top of the frame-logit stream.

Parameter budget
----------------
With the defaults below (``encoder_feat_dim=1024``, ``hidden_dim=512``,
``n_layers=2``, ``vocab_size=170``) the head has roughly **0.87 M**
parameters — comfortably inside the ``[0.5 M, 5 M]`` envelope the test
suite enforces.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ChordHeadConfig:
    """Hyperparameters for :class:`ChordHead`.

    ``vocab_size`` defaults to ``170`` to match :class:`ChordTokenizer`. If
    the tokenizer's vocabulary ever grows, bump this in lock-step (and
    update the test asserting the count).
    """

    encoder_feat_dim: int = 1024   # MERT-v1-330M hidden size
    hidden_dim: int = 512
    n_layers: int = 2              # number of MLP hidden layers (>=1)
    dropout: float = 0.1
    vocab_size: int = 170          # matches ChordTokenizer.vocab_size


class ChordHead(nn.Module):
    """Frame-level chord classifier over MERT features."""

    def __init__(self, cfg: ChordHeadConfig) -> None:
        super().__init__()
        if cfg.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {cfg.n_layers}")
        if cfg.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {cfg.hidden_dim}")
        if cfg.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {cfg.vocab_size}")

        self.cfg = cfg

        layers: list[nn.Module] = [
            nn.Linear(cfg.encoder_feat_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        ]
        for _ in range(cfg.n_layers - 1):
            layers.extend(
                [
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(cfg.dropout),
                ]
            )
        layers.append(nn.Linear(cfg.hidden_dim, cfg.vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, encoder_feats: torch.Tensor) -> torch.Tensor:
        """Apply the per-frame MLP and return per-frame logits.

        Parameters
        ----------
        encoder_feats:
            Shape ``(B, T, encoder_feat_dim)`` float tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T, vocab_size)`` logit tensor. No softmax is applied;
            consumers run cross-entropy / log-softmax themselves.
        """
        if encoder_feats.dim() != 3:
            raise ValueError(
                "encoder_feats must be 3-D (B, T, D); got shape "
                f"{tuple(encoder_feats.shape)}"
            )
        if encoder_feats.size(-1) != self.cfg.encoder_feat_dim:
            raise ValueError(
                "encoder_feats last dim "
                f"{encoder_feats.size(-1)} != cfg.encoder_feat_dim "
                f"{self.cfg.encoder_feat_dim}"
            )
        # nn.Linear broadcasts across the leading (B, T) dims, so no reshape
        # is needed — the MLP runs independently per frame.
        return self.net(encoder_feats)


__all__ = ["ChordHead", "ChordHeadConfig"]
