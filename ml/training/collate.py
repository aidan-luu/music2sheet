"""Batch collation for the melody training pipeline (PR-4b).

The training-side ``Dataset`` yields variable-length ``(encoder_feats,
target_tokens)`` pairs because input audio clips differ in length. The
collate function below pads everything in a batch to the longest sequence,
zero-filling encoder features and using :attr:`MelodyTokenizer.PAD` (==0)
for the token targets so a cross-entropy loss can mask them out with
``ignore_index=0`` later.
"""

from __future__ import annotations

import torch

from ml.models.melody_tokenizer import MelodyTokenizer


def melody_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a list of ``MelodyTrainingDataset`` items into a single batch.

    Parameters
    ----------
    batch:
        List of dicts as produced by :class:`MelodyTrainingDataset`, each
        with keys ``"encoder_feats"`` (shape ``(T_i, D)`` float) and
        ``"target_tokens"`` (shape ``(T_i,)`` long).

    Returns
    -------
    dict[str, torch.Tensor]
        ``encoder_feats`` of shape ``(B, T_max, D)``,
        ``target_tokens`` of shape ``(B, T_max)``,
        ``lengths`` of shape ``(B,)`` int64 giving each sample's true length.

    Notes
    -----
    Padding values:

    * ``encoder_feats`` is padded with ``0.0``.
    * ``target_tokens`` is padded with :attr:`MelodyTokenizer.PAD` (==0).
      This matches the convention the decoder's loss expects.
    """
    if not batch:
        raise ValueError("melody_collate received an empty batch")

    feat_dim = int(batch[0]["encoder_feats"].shape[-1])
    lengths = torch.tensor(
        [int(item["encoder_feats"].shape[0]) for item in batch], dtype=torch.int64
    )
    t_max = int(lengths.max().item())
    batch_size = len(batch)

    encoder_feats = torch.zeros(
        (batch_size, t_max, feat_dim),
        dtype=batch[0]["encoder_feats"].dtype,
    )
    target_tokens = torch.full(
        (batch_size, t_max),
        fill_value=MelodyTokenizer.PAD,
        dtype=batch[0]["target_tokens"].dtype,
    )

    for i, item in enumerate(batch):
        feats = item["encoder_feats"]
        toks = item["target_tokens"]
        t_i = int(feats.shape[0])
        if int(toks.shape[0]) != t_i:
            raise ValueError(
                f"item {i}: encoder_feats len {t_i} != target_tokens len "
                f"{int(toks.shape[0])}; pipeline alignment invariant violated"
            )
        if int(feats.shape[-1]) != feat_dim:
            raise ValueError(
                f"item {i}: encoder_feats feature dim {int(feats.shape[-1])} "
                f"!= batch[0] dim {feat_dim}"
            )
        encoder_feats[i, :t_i] = feats
        target_tokens[i, :t_i] = toks

    return {
        "encoder_feats": encoder_feats,
        "target_tokens": target_tokens,
        "lengths": lengths,
    }


__all__ = ["melody_collate"]
