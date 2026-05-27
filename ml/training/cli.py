"""Minimal CLI for the melody trainer (PR-4c).

PR-4c does not run real training — this entry point only wires the
existing PR-4a (decoder + tokenizer), PR-4b (data pipeline + dataset)
and PR-4c (:class:`MelodyTrainer`) pieces together so PR-4d can swap
real data + GPU in without touching plumbing.

Usage::

    python -m ml.training.cli --run-id smoke --manifest <path>

``--help`` is the only behaviour the PR-4c QA suite verifies. Running
the command end-to-end requires HookTheory data, Demucs + MERT model
downloads, and is intentionally out of scope here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from ml.datasets.hooktheory import HookTheoryDataset
from ml.training.collate import melody_collate
from ml.training.datasets import MelodyTrainingDataset
from ml.training.melody_pipeline import MelodyDataPipeline
from ml.training.trainer import MelodyTrainer, TrainConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.training.cli",
        description="Train the melody decoder on HookTheory data (PR-4c scaffolding).",
    )
    parser.add_argument("--run-id", required=True, help="Identifier for this run; used as the run directory name.")
    parser.add_argument("--manifest", required=True, help="Path to the HookTheory JSONL manifest.")
    parser.add_argument("--split", default="train", help="Which split inside the manifest to read.")
    parser.add_argument("--cache-dir", default=None, help="Optional directory for cached (feats, tokens) arrays.")
    parser.add_argument("--runs-root", default="runs", help="Root directory under which run artifacts are written.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--eval-every-steps", type=int, default=200)
    parser.add_argument("--save-every-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Override device autodetect (e.g. cpu, cuda, mps).")
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=30.0,
        help="Drop training entries longer than this many seconds.",
    )
    return parser


def _build_loader(args: argparse.Namespace) -> DataLoader:
    underlying = HookTheoryDataset(args.manifest, split=args.split)
    pipeline = MelodyDataPipeline()
    dataset = MelodyTrainingDataset(
        underlying=underlying,
        pipeline=pipeline,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        max_duration_s=args.max_duration_s,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=melody_collate,
        shuffle=True,
    )


def _cfg_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        run_id=args.run_id,
        runs_root=Path(args.runs_root),
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        log_every_steps=args.log_every_steps,
        eval_every_steps=args.eval_every_steps,
        save_every_steps=args.save_every_steps,
        seed=args.seed,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loader = _build_loader(args)
    cfg = _cfg_from_args(args)
    trainer = MelodyTrainer(cfg, train_loader=loader)
    summary = trainer.train()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
