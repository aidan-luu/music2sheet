"""Fetch + normalise HookTheory's TheoryTab JSON dump.

Usage::

    python -m ml.datasets.scripts.fetch_hooktheory --out ~/sheet-sage-data

The script downloads SheetSage's redistribution of the HookTheory TheoryTab
corpus (a ~20 MB gzip-compressed JSON), writes raw + extracted copies under
``<out>/raw/hooktheory/``, and emits a normalized JSON Lines manifest at
``<out>/manifests/hooktheory.jsonl`` with one line per song.

Stratified 80/10/10 train/val/test split is keyed on the song's artist string
so all songs by the same artist land in the same split (matches the SheetSage
paper's protocol). Artists with fewer than 3 songs are kept entirely in
``train`` because they cannot be safely split.

We do NOT download YouTube audio. ``audio_path`` is set to ``null`` in every
manifest entry; the audio acquisition pipeline lives in a later PR.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from ml.datasets.scripts._common import (
    DownloadError,
    build_argparser,
    configure_logging,
    dataset_paths,
    download,
    ensure_dirs,
    manifest_split_counts,
    open_manifest,
    print_summary,
    write_manifest_entry,
    write_readme,
)
from ml.datasets.scripts._constants import (
    HOOKTHEORY_CITATION,
    HOOKTHEORY_LICENSE,
    HOOKTHEORY_NAME,
    HOOKTHEORY_RELEASES_PAGE,
    HOOKTHEORY_SHA256,
    HOOKTHEORY_URL_CANDIDATES,
)

LOG = logging.getLogger("ml.datasets.scripts.fetch_hooktheory")


def _stable_bucket(key: str, n_buckets: int = 10) -> int:
    """Hash ``key`` into one of ``n_buckets`` buckets deterministically."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % n_buckets


def assign_split(artist: str) -> str:
    """Stratify 80/10/10 train/val/test by hashed artist name.

    Using a stable hash means: (a) the same artist always lands in the same
    split across re-runs, (b) all songs by one artist share a split, which is
    how the SheetSage paper avoids artist leakage between train and test.
    """
    bucket = _stable_bucket(artist or "__unknown__", n_buckets=10)
    if bucket < 8:
        return "train"
    if bucket < 9:
        return "val"
    return "test"


def parse_hooktheory_json(data: Any) -> list[dict[str, Any]]:
    """Normalise SheetSage's hooktheory.json into common-schema dicts.

    The released JSON is a dict mapping song-id -> {"artist": ..., "song": ...,
    "alignment": {...}, "annotations": {"notes": [...], "harmony": [...]}}.
    The exact upstream schema has changed between releases; we are tolerant
    about missing fields and never KeyError on malformed entries (we just skip
    them with a warning).
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected hooktheory JSON to be a dict at top level, got {type(data).__name__}"
        )

    out: list[dict[str, Any]] = []
    for song_id, song in data.items():
        if not isinstance(song, dict):
            LOG.warning("hooktheory: skipping non-dict song %r", song_id)
            continue
        artist = str(song.get("artist") or song.get("Artist") or "unknown")
        title = str(song.get("song") or song.get("title") or song_id)

        notes = _extract_notes(song)
        chords = _extract_chords(song)
        beats = _extract_beats(song)
        key = _extract_key(song)

        entry = {
            "id": f"hooktheory:{song_id}",
            "artist": artist,
            "title": title,
            "audio_path": None,
            "beats": beats,
            "notes": notes,
            "chords": chords,
            "key": key,
            "split": assign_split(artist),
        }
        out.append(entry)
    return out


def _extract_notes(song: dict[str, Any]) -> list[dict[str, Any]]:
    notes_in = (
        song.get("annotations", {}).get("notes")
        if isinstance(song.get("annotations"), dict)
        else song.get("notes")
    )
    if not isinstance(notes_in, list):
        return []
    out: list[dict[str, Any]] = []
    for n in notes_in:
        if not isinstance(n, dict):
            continue
        try:
            out.append(
                {
                    "pitch": int(n.get("pitch", n.get("midi", 0))),
                    "onset": float(n.get("onset", n.get("start", 0.0))),
                    "duration": float(n.get("duration", n.get("dur", 0.0))),
                    "velocity": int(n.get("velocity", 80)),
                }
            )
        except (TypeError, ValueError):
            continue
    return out


def _extract_chords(song: dict[str, Any]) -> list[dict[str, Any]]:
    src = (
        song.get("annotations", {}).get("harmony")
        if isinstance(song.get("annotations"), dict)
        else song.get("chords")
    )
    if not isinstance(src, list):
        return []
    out: list[dict[str, Any]] = []
    for c in src:
        if not isinstance(c, dict):
            continue
        try:
            out.append(
                {
                    "label": str(c.get("label", c.get("chord", "N"))),
                    "onset": float(c.get("onset", c.get("start", 0.0))),
                    "duration": float(c.get("duration", c.get("dur", 0.0))),
                    "confidence": 1.0,
                }
            )
        except (TypeError, ValueError):
            continue
    return out


def _extract_beats(song: dict[str, Any]) -> list[dict[str, Any]]:
    beats_in = song.get("beats")
    if not isinstance(beats_in, list):
        return []
    out: list[dict[str, Any]] = []
    for b in beats_in:
        if isinstance(b, dict):
            try:
                out.append(
                    {
                        "time": float(b.get("time", 0.0)),
                        "downbeat": bool(b.get("downbeat", False)),
                        "confidence": float(b.get("confidence", 1.0)),
                    }
                )
            except (TypeError, ValueError):
                continue
        else:
            try:
                out.append({"time": float(b), "downbeat": False, "confidence": 1.0})
            except (TypeError, ValueError):
                continue
    return out


def _extract_key(song: dict[str, Any]) -> dict[str, Any] | None:
    key_in = song.get("key") or song.get("annotations", {}).get("key") if isinstance(
        song.get("annotations"), dict
    ) else song.get("key")
    if not isinstance(key_in, dict):
        return None
    try:
        return {
            "tonic": int(key_in.get("tonic", 0)),
            "mode": str(key_in.get("mode", "major")),
            "confidence": float(key_in.get("confidence", 1.0)),
        }
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser(
        HOOKTHEORY_NAME,
        description="Fetch SheetSage's HookTheory TheoryTab JSON and emit a normalized manifest.",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    raw_dir, manifest_path = dataset_paths(args.out, HOOKTHEORY_NAME)
    ensure_dirs(raw_dir, manifest_path.parent, dry_run=args.dry_run)

    archive_path = raw_dir / "hooktheory.json.gz"
    try:
        download(
            HOOKTHEORY_URL_CANDIDATES,
            archive_path,
            expected_sha256=HOOKTHEORY_SHA256,
            force=args.force,
            dry_run=args.dry_run,
        )
    except DownloadError as exc:
        LOG.error(
            "Automatic download failed: %s.\n"
            "  Please download hooktheory.json.gz manually from %s\n"
            "  and place it at %s, then re-run with --force.",
            exc,
            HOOKTHEORY_RELEASES_PAGE,
            archive_path,
        )
        return 2

    write_readme(
        raw_dir,
        dataset_name=HOOKTHEORY_NAME,
        citation=HOOKTHEORY_CITATION,
        license_str=HOOKTHEORY_LICENSE,
        manual_steps=(
            "If the automatic download 404s, fetch the latest hooktheory.json.gz "
            f"from {HOOKTHEORY_RELEASES_PAGE} and drop it in this directory.\n\n"
            "Audio is not bundled. The TheoryTab JSON includes YouTube URLs; "
            "obtaining audio is out of scope for this dataset script."
        ),
        dry_run=args.dry_run,
    )

    # Decode JSON.
    extracted_path = raw_dir / "hooktheory.json"
    if args.dry_run:
        LOG.info("[dry-run] would gunzip %s -> %s", archive_path, extracted_path)
        data: Any = {}
    else:
        if not extracted_path.exists() or args.force:
            LOG.info("hooktheory: gunzipping %s", archive_path)
            with gzip.open(archive_path, "rb") as gz, extracted_path.open("wb") as out:
                shutil.copyfileobj(gz, out)
        with extracted_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

    entries = parse_hooktheory_json(data) if not args.dry_run else []

    # Re-balance singletons: artists with <3 songs entirely to train so val/test
    # do not collapse to zero for those artists. We only do this if val/test
    # ended up empty for them (the hash already does the bulk of the work).
    by_artist: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        by_artist[e["artist"]].append(e)
    for artist, songs in by_artist.items():
        if len(songs) < 3:
            for e in songs:
                e["split"] = "train"

    with open_manifest(manifest_path, dry_run=args.dry_run) as fh:
        for e in entries:
            write_manifest_entry(fh, e)

    counts = manifest_split_counts(entries)
    print_summary(HOOKTHEORY_NAME, counts, manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
