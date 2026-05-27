"""Clone + normalise POP909.

Usage::

    python -m ml.datasets.scripts.fetch_pop909 --out ~/sheet-sage-data

POP909 is distributed exclusively as a GitHub repository (no PyPI / tarball).
This script shallow-clones ``music-x-lab/POP909-Dataset`` into
``<out>/raw/pop909/repo/`` and walks the resulting tree to enumerate the
909 song folders (named ``001`` through ``909``). Each folder contains an
aligned MIDI plus per-track ``melody.txt`` / ``chord_midi.txt`` annotation
files; the script records their on-disk paths in the manifest.

**Audio is not included** in POP909 by design (Wang et al. cite copyright as
the reason). Acquiring audio is the user's responsibility; we work from MIDI
for the voicing-model pretraining in PR-9.

Stratified split: 80/10/10 train/val/test by song id (deterministic hash).
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

from ml.datasets.scripts._common import (
    build_argparser,
    configure_logging,
    dataset_paths,
    ensure_dirs,
    git_clone,
    manifest_split_counts,
    open_manifest,
    print_summary,
    write_manifest_entry,
    write_readme,
)
from ml.datasets.scripts._constants import (
    POP909_CITATION,
    POP909_GIT_URL,
    POP909_LICENSE,
    POP909_NAME,
)

LOG = logging.getLogger("ml.datasets.scripts.fetch_pop909")


def _split_for_id(song_id: str) -> str:
    bucket = int.from_bytes(
        hashlib.sha256(song_id.encode("utf-8")).digest()[:4], "big"
    ) % 10
    if bucket < 8:
        return "train"
    if bucket < 9:
        return "val"
    return "test"


def find_song_dirs(repo_root: Path) -> list[Path]:
    """Locate the 909 numbered song directories inside the cloned repo.

    The repo layout is ``POP909-Dataset/POP909/<NNN>/`` in current upstream
    HEAD, but earlier commits had ``POP909-Dataset/<NNN>/`` directly. We
    search both layouts and return whichever has any matching folders.
    """
    candidates = [repo_root / "POP909", repo_root]
    for c in candidates:
        if not c.exists():
            continue
        dirs = sorted(p for p in c.iterdir() if p.is_dir() and p.name.isdigit())
        if dirs:
            return dirs
    return []


def parse_song_dir(song_dir: Path) -> dict[str, list]:
    """Best-effort parse of one POP909 song folder.

    POP909 ships three kinds of annotation text files:

    * ``beat_audio.txt`` / ``beat_midi.txt``: ``<time>\\t<beat_in_bar>`` pairs.
    * ``chord_audio.txt`` / ``chord_midi.txt``: ``<start>\\t<end>\\t<label>``.
    * ``melody.txt``: ``<midi>\\t<onset>\\t<offset>`` (some forks use
      ``<onset>\\t<duration>\\t<midi>``; we autodetect).

    Missing files are silently treated as empty lists. The MIDI file at
    ``<NNN>.mid`` (or ``<NNN>/<NNN>.mid``) is referenced from ``audio_path``
    for downstream MIDI-as-source training (audio_path is reused for MIDI
    here intentionally so the schema stays uniform).
    """
    beats: list[dict] = []
    notes: list[dict] = []
    chords: list[dict] = []

    beat_file = _first_existing(
        song_dir / "beat_midi.txt", song_dir / "beat_audio.txt"
    )
    if beat_file is not None:
        for line in _read_lines(beat_file):
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
                pos = int(float(parts[1]))
            except ValueError:
                continue
            beats.append({"time": t, "downbeat": pos == 1, "confidence": 1.0})

    chord_file = _first_existing(
        song_dir / "chord_midi.txt", song_dir / "chord_audio.txt"
    )
    if chord_file is not None:
        for line in _read_lines(chord_file):
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError:
                continue
            label = parts[2].strip() or "N"
            chords.append(
                {
                    "label": label,
                    "onset": start,
                    "duration": max(0.0, end - start),
                    "confidence": 1.0,
                }
            )

    melody_file = song_dir / "melody.txt"
    if melody_file.exists():
        for line in _read_lines(melody_file):
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                a, b, c = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
            # Heuristic: if the first column looks like a MIDI pitch (small
            # int between 21 and 108), treat it as <pitch onset offset>;
            # otherwise treat as <onset duration pitch>.
            if 21 <= a <= 108 and float(int(a)) == a:
                pitch = int(a)
                onset, offset = b, c
                duration = max(0.0, offset - onset)
            else:
                onset, duration = a, b
                pitch = int(c)
            notes.append(
                {"pitch": pitch, "onset": onset, "duration": duration, "velocity": 80}
            )

    return {"beats": beats, "notes": notes, "chords": chords}


def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _read_lines(path: Path) -> list[str]:
    try:
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return []


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser(
        POP909_NAME,
        description="Clone music-x-lab/POP909-Dataset and emit a normalized manifest.",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    raw_dir, manifest_path = dataset_paths(args.out, POP909_NAME)
    ensure_dirs(raw_dir, manifest_path.parent, dry_run=args.dry_run)

    repo_dir = raw_dir / "repo"
    git_clone(POP909_GIT_URL, repo_dir, force=args.force, dry_run=args.dry_run)

    write_readme(
        raw_dir,
        dataset_name=POP909_NAME,
        citation=POP909_CITATION,
        license_str=POP909_LICENSE,
        manual_steps=(
            "Audio is NOT included in POP909 (copyright). Aligned MIDI is "
            "shipped instead and is what this manifest references via "
            "``audio_path``. If your training run needs audio, you must source "
            "it independently and respect copyright."
        ),
        dry_run=args.dry_run,
    )

    entries: list[dict] = []
    if not args.dry_run:
        for song_dir in find_song_dirs(repo_dir):
            song_id = song_dir.name
            ann = parse_song_dir(song_dir)
            midi_path = _first_existing(
                song_dir / f"{song_id}.mid", song_dir / f"{song_id}.midi"
            )
            entry = {
                "id": f"pop909:{song_id}",
                "audio_path": str(midi_path) if midi_path else None,
                "beats": ann["beats"],
                "notes": ann["notes"],
                "chords": ann["chords"],
                "key": None,
                "split": _split_for_id(song_id),
            }
            entries.append(entry)

    with open_manifest(manifest_path, dry_run=args.dry_run) as fh:
        for e in entries:
            write_manifest_entry(fh, e)

    counts = manifest_split_counts(entries)
    print_summary(POP909_NAME, counts, manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
