"""Fetch + normalise RWC-Pop annotations (HELD-OUT eval set).

Usage::

    python -m ml.datasets.scripts.fetch_rwc_pop --out ~/sheet-sage-data \\
        --audio-root /path/to/rwc/audio

================================================================================
RWC-Pop is the project's held-out eval set. Every manifest entry written by
this script is tagged ``split == "test"``. Any training loop that pulls from
this manifest is a bug. The pass-criterion gate (PR-5/-6/-8) is computed
strictly on RWC-Pop and must not see leakage from training.
================================================================================

The script does two things:

1. Clones the free chord-annotation repo ``tmc323/Chord-Annotations`` (also
   carrying beat + key labels for RWC-Pop) and parses the Harte-style ``.lab``
   files into the normalised schema.

2. Optionally resolves ``audio_path`` for each track. RWC-Pop audio is
   commercial and distributed only by AIST under a signed agreement; the
   script does NOT download audio. If the user passes ``--audio-root <dir>``,
   the script looks for files matching ``RM-P<NNN>.wav`` (or ``.flac``,
   ``.mp3``) inside that directory and records absolute paths in the
   manifest. Otherwise ``audio_path`` is ``null`` and audio must be filled in
   later.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

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
    RWC_POP_AUDIO_INFO_URL,
    RWC_POP_CHORD_GIT_URL,
    RWC_POP_CITATION,
    RWC_POP_HELDOUT_BANNER,
    RWC_POP_LICENSE,
    RWC_POP_NAME,
)

LOG = logging.getLogger("ml.datasets.scripts.fetch_rwc_pop")

_RM_P_RE = re.compile(r"RM-?P0*([0-9]+)", re.IGNORECASE)
_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".aiff", ".aif", ".ogg")


def parse_lab_file(path: Path) -> list[dict[str, Any]]:
    """Parse a Harte-style ``.lab`` chord file (same format as Isophonics).

    Duplicated rather than imported from fetch_isophonics so each fetch script
    is independently runnable (no cross-script imports beyond _common).
    """
    chords: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        LOG.warning("rwc_pop: cannot read %s: %s", path, exc)
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError:
            continue
        label = " ".join(parts[2:]) or "N"
        chords.append(
            {
                "label": label,
                "onset": start,
                "duration": max(0.0, end - start),
                "confidence": 1.0,
            }
        )
    return chords


def find_audio_file(audio_root: Path, track_num: int) -> Path | None:
    """Find an audio file for ``RM-P<NNN>`` under ``audio_root``.

    AIST's distribution names files differently across CDs (``RM-P001.wav`` vs
    ``RM-P001.flac`` vs ``Track 01.wav`` in CDDA rips); we match any file in
    the tree whose name contains ``RM-P<NNN>`` or ``RM-P<padded>``.
    """
    if not audio_root.exists():
        return None
    padded = f"{track_num:03d}"
    for ext in _AUDIO_EXTS:
        for cand in audio_root.rglob(f"*RM-P{padded}*{ext}"):
            return cand
        for cand in audio_root.rglob(f"*RM-P{track_num}*{ext}"):
            return cand
    return None


def extract_track_num(filename: str) -> int | None:
    m = _RM_P_RE.search(filename)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser(
        RWC_POP_NAME,
        description=(
            "Fetch RWC-Pop chord/beat annotations (HELD-OUT eval set) and emit a "
            "normalized manifest."
        ),
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help=(
            "Optional path to the directory containing legally-acquired RWC-Pop "
            "audio. If provided, the script resolves audio_path for each track."
        ),
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    # Loud, hard-to-miss banner on every invocation.
    print(RWC_POP_HELDOUT_BANNER, file=sys.stderr, flush=True)

    raw_dir, manifest_path = dataset_paths(args.out, RWC_POP_NAME)
    ensure_dirs(raw_dir, manifest_path.parent, dry_run=args.dry_run)

    write_readme(
        raw_dir,
        dataset_name=RWC_POP_NAME,
        citation=RWC_POP_CITATION,
        license_str=RWC_POP_LICENSE,
        manual_steps=(
            "Audio acquisition (NOT automated):\n"
            f"  RWC-Pop audio is sold by AIST under a signed research-use agreement.\n"
            f"  See {RWC_POP_AUDIO_INFO_URL} for the order form.\n"
            "  After receiving the CDs, rip them to ``RM-P001.wav`` ... ``RM-P100.wav``\n"
            "  and re-run this script with --audio-root <that-dir>.\n\n"
            "Annotation acquisition (automated):\n"
            f"  Cloned from {RWC_POP_CHORD_GIT_URL}.\n"
            "  Chord + beat + key labels are free for research use; see that repo's\n"
            "  LICENSE / README for the exact terms.\n"
        ),
        extra_banner=RWC_POP_HELDOUT_BANNER,
        dry_run=args.dry_run,
    )

    chord_repo = raw_dir / "annotations" / "Chord-Annotations"
    git_clone(
        RWC_POP_CHORD_GIT_URL, chord_repo, force=args.force, dry_run=args.dry_run
    )

    entries: list[dict[str, Any]] = []
    if not args.dry_run and chord_repo.exists():
        # The Chord-Annotations repo organises files as
        #   <repo>/RWC-Pop/<filename>.lab
        # but older revisions used <repo>/RWC_Pop/, so we glob.
        for lab_path in sorted(chord_repo.rglob("*.lab")):
            # Only process files matching the RM-P pattern (the repo also
            # contains RWC-MDB-Classical and other unrelated annotations).
            track_num = extract_track_num(lab_path.name) or extract_track_num(
                str(lab_path)
            )
            if track_num is None:
                continue
            slot_id = f"RM-P{track_num:03d}"
            full_id = f"rwc_pop:{slot_id}"
            chords = parse_lab_file(lab_path)
            audio_path: str | None = None
            if args.audio_root is not None:
                found = find_audio_file(args.audio_root, track_num)
                if found is not None:
                    audio_path = str(found.resolve())
                else:
                    LOG.warning(
                        "rwc_pop: no audio file matching %s under %s",
                        slot_id,
                        args.audio_root,
                    )
            entries.append(
                {
                    "id": full_id,
                    "audio_path": audio_path,
                    "beats": [],
                    "notes": [],
                    "chords": chords,
                    "key": None,
                    "split": "test",  # invariant — see banner above.
                }
            )

    with open_manifest(manifest_path, dry_run=args.dry_run) as fh:
        for e in entries:
            write_manifest_entry(fh, e)

    counts = manifest_split_counts(entries)
    print_summary(RWC_POP_NAME, counts, manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
