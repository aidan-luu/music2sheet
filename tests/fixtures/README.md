# Audio fixtures

This directory will hold a small set (3-5) of short audio clips (5-15 s each)
plus matching ground-truth annotations:

- `<clip>.wav` (or `.mp3`)
- `<clip>.mid` (reference melody / accompaniment MIDI)
- `<clip>.chords.lab` (Isophonics-style chord annotation)
- `<clip>.beats.txt` (beat + downbeat timestamps)

Audio data and annotations are procured and committed by **Agent C** as part
of PR-1 / PR-2 once licensing for each clip is confirmed. Agent D consumes
these fixtures in `tests/` and `evals/` but does not produce them.

Until then this directory is intentionally empty.
