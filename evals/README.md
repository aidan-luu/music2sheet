# Evaluation suite

This directory will host the offline evaluation harness for music2sheet.

## Metrics (all via `mir_eval`)

- **Melody F1** - frame-level pitch accuracy + voicing F1.
- **Chord WCSR** - weighted chord symbol recall under several vocabularies
  (`majmin`, `sevenths`, `tetrads`, `mirex`).
- **Beat F1** - beat tracking F-measure (50 ms tolerance) and downbeat F1.
- **Key accuracy** - MIREX weighted key accuracy.

## Datasets

Per-dataset runners will live alongside this README, e.g.
`run_isophonics.py`, `run_billboard.py`, `run_pop909.py`, `run_medleydb.py`.

**RWC-Pop is strictly held out** as the final blind test set. It is never
touched during training, hyperparameter selection, or development eval - it
is only invoked from the nightly job for the leaderboard.

## Leaderboard

Each completed eval run appends a row to [`LEADERBOARD.md`](LEADERBOARD.md).
Agent D owns the harness and the leaderboard; Agent C is responsible for the
model checkpoints under eval.

This README is a placeholder; the harness itself ships in PR-13.
