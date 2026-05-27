"""Melody decoder architecture (PR-4a — model only, no training).

A cross-attention Transformer decoder that consumes MERT-style encoder
features and emits a framewise sequence of melody pitch tokens (one
token per encoder frame, REST when no melody is active). The token
formulation matches :mod:`ml.models.melody_tokenizer` — see that module
for the special-token layout.

Architectural notes
-------------------
* Standard ``nn.TransformerDecoder`` with pre-norm (``norm_first=True``)
  for training stability, followed by an explicit final ``LayerNorm``.
* Sinusoidal positional encoding (no learned position params) registered
  as a non-persistent buffer.
* Encoder features (``encoder_feat_dim``-wide, e.g. 1024 for MERT v1-330M)
  are projected down to ``d_model`` once before cross-attention.
* The output projection is **tied** to the token embedding matrix:
  logits are computed as ``hidden @ token_embedding.weight.T``.

Deviations from the original PR-4a ticket
-----------------------------------------
The ticket states ~50M params as the target and includes a unit test
bounding the count to ``[40M, 60M]``. Two ticket defaults conflicted
with this and were adjusted:

* ``n_layers`` raised from 6 → 12. With ``d_model=512`` and
  ``d_ff=2048`` six layers gives ~21M params, well below the test
  bound. Doubling the depth yields ~51M.
* ``vocab_size`` raised from 130 → 132. The companion tokenizer reserves
  IDs 0-3 for ``PAD/BOS/EOS/REST`` and then maps the 128 MIDI pitches to
  4-131, so the embedding table must be at least 132 wide.

Both deviations are noted in the PR-4a completion summary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MelodyDecoderConfig:
    """Hyperparameters for :class:`MelodyDecoder`."""

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 2048
    vocab_size: int = 132
    encoder_feat_dim: int = 1024


def _sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Standard "Attention Is All You Need" sinusoidal table, shape (max_len, d_model)."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def _causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Additive causal mask: 0 on/under diagonal, ``-inf`` above."""
    mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    return mask.masked_fill(
        torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )


class MelodyDecoder(nn.Module):
    """Cross-attention Transformer decoder for framewise melody transcription."""

    PAD: int = 0
    BOS: int = 1
    EOS: int = 2

    def __init__(self, cfg: MelodyDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.encoder_projection = nn.Linear(cfg.encoder_feat_dim, cfg.d_model)

        self.register_buffer(
            "positional_encoding",
            _sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model),
            persistent=False,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.n_layers)
        self.final_norm = nn.LayerNorm(cfg.d_model)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _embed_targets(self, target_tokens: torch.Tensor) -> torch.Tensor:
        if target_tokens.dim() != 2:
            raise ValueError(
                f"target_tokens must be 2-D (B, T_dec); got shape {tuple(target_tokens.shape)}"
            )
        seq_len = target_tokens.size(1)
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(
                f"target_tokens length {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}"
            )
        x = self.token_embedding(target_tokens) * math.sqrt(self.cfg.d_model)
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        return x

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def forward(
        self,
        encoder_feats: torch.Tensor,
        target_tokens: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Parameters
        ----------
        encoder_feats:
            ``(B, T_enc, encoder_feat_dim)`` — MERT features. Projected to
            ``d_model`` internally before cross-attention.
        target_tokens:
            ``(B, T_dec)`` — already shifted-right (BOS-prefixed) target IDs.
        target_mask:
            Optional ``(B, T_dec)`` boolean key-padding mask. ``True``
            positions are *ignored* by attention. The causal mask is added
            internally; callers do not need to pass it.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, T_dec, vocab_size)`` over the melody vocab.
        """
        if encoder_feats.dim() != 3:
            raise ValueError(
                f"encoder_feats must be 3-D (B, T_enc, D); got shape {tuple(encoder_feats.shape)}"
            )
        if encoder_feats.size(-1) != self.cfg.encoder_feat_dim:
            raise ValueError(
                "encoder_feats last dim "
                f"{encoder_feats.size(-1)} != cfg.encoder_feat_dim {self.cfg.encoder_feat_dim}"
            )

        memory = self.encoder_projection(encoder_feats)
        tgt = self._embed_targets(target_tokens)
        causal = _causal_mask(tgt.size(1), tgt.device, tgt.dtype)

        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=target_mask,
        )
        out = self.final_norm(out)
        # Tied output projection.
        logits = out @ self.token_embedding.weight.T
        return logits

    @torch.no_grad()
    def generate(
        self,
        encoder_feats: torch.Tensor,
        max_new_tokens: int = 1024,
        beam_size: int = 1,
    ) -> torch.Tensor:
        """Autoregressive decoding from encoder features.

        Greedy decoding only (``beam_size=1``). A future PR can swap in a
        beam-search loop; the signature already accepts the parameter so
        callers don't have to migrate later.

        The BOS prefix is *not* included in the returned tensor — only the
        generated tokens are returned. Output shape is
        ``(B, n_generated)`` with ``n_generated <= max_new_tokens``.
        """
        if beam_size != 1:
            raise NotImplementedError(
                f"Only greedy decoding (beam_size=1) is implemented; got beam_size={beam_size}"
            )
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")

        was_training = self.training
        self.eval()
        try:
            batch_size = encoder_feats.size(0)
            device = encoder_feats.device
            generated = torch.full(
                (batch_size, 1),
                fill_value=self.BOS,
                dtype=torch.long,
                device=device,
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_new_tokens):
                logits = self.forward(encoder_feats, generated)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                # Sequences that already hit EOS keep emitting EOS so the batch
                # stays rectangular and we can stop early when all are done.
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, self.EOS),
                    next_token,
                )
                generated = torch.cat([generated, next_token], dim=1)
                finished = finished | (next_token.squeeze(1) == self.EOS)
                if bool(finished.all()):
                    break

            return generated[:, 1:]
        finally:
            if was_training:
                self.train()
