# Sheet Sage 2

A successor to SheetSage: turn an audio recording (mp3, wav, or URL) into a
clean two-stave lead sheet with chord voicings, key, beat grid, and rendered
MusicXML / MIDI.

## Stack

- **HT-Demucs v4** for source separation (vocals / accompaniment).
- **MERT-v1-330M** as the shared acoustic feature backbone.
- **Beat Transformer** for downbeat-aware beat tracking.
- Task-specific heads on MERT for melody, chord (large-vocab + boundary
  refinement), and key, plus a learned voicing model fine-tuned on POP909.
- A quantization-to-beat-grid engine and a MusicXML / LilyPond / MIDI writer
  produce the final score.

Inference is offered as a **hosted API only** (FastAPI + Redis/RQ on Modal or
Replicate, with a thin HuggingFace Space demo UI). No local-install
distribution is supported.

## Team

This repo is built by a 4-agent team. See [`skills.md`](skills.md) for the
ownership matrix, boundaries, and handoff protocol.

## Status

Pre-alpha. PR-0 scaffolding only; no models or endpoints exist yet.

## License

MIT. See [`LICENSE`](LICENSE).
