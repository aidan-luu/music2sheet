"""Fetch + normalise Isophonics chord annotations.

Usage::

    python -m ml.datasets.scripts.fetch_isophonics --out ~/sheet-sage-data

Isophonics releases five sub-collections of ``.lab`` chord annotations on
isophonics.net. Each sub-collection is a ``.tar.gz`` of per-song files in
Harte's chord-label format::

    <start_sec>\\t<end_sec>\\t<harte_label>

Audio is NOT distributed; ``audio_path`` is ``null`` in every manifest entry.
That's fine for chord-head training (PR-6) because we condition on MERT
features extracted elsewhere — the chord head only needs the chord-label
time-series.

Stratified split: 80/10/10 train/val/test by song, deterministic hash.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import tarfile
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
    ISOPHONICS_CITATION,
    ISOPHONICS_LICENSE,
    ISOPHONICS_NAME,
    ISOPHONICS_SUBSETS,
)

LOG = logging.getLogger("ml.datasets.scripts.fetch_isophonics")


def _split_for_id(song_id: str) -> str:
    bucket = int.from_bytes(
        hashlib.sha256(song_id.encode("utf-8")).digest()[:4], "big"
    ) % 10
    if bucket < 8:
        return "train"
    if bucket < 9:
        return "val"
    return "test"


def parse_lab_file(path: Path) -> list[dict[str, Any]]:
    """Parse one Harte ``.lab`` annotation file.

    The Harte format is whitespace-separated and always has three columns:
    ``start_sec end_sec label``. We accept tabs or spaces and ignore any
    comment lines starting with ``#``.
    """
    chords: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        LOG.warning("isophonics: cannot read %s: %s", path, exc)
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
        label = " ".join(parts[2:]).strip() or "N"
        chords.append(
            {
                "label": label,
                "onset": start,
                "duration": max(0.0, end - start),
                "confidence": 1.0,
            }
        )
    return chords


def safe_extract_tar(archive: Path, dest: Path) -> None:
    """Extract a tarball, rejecting any member that escapes ``dest``.

    Isophonics archives sometimes contain absolute paths or ``..`` segments
    from very old tooling; this guard is precautionary.
    """
    with tarfile.open(archive, "r:*") as tar:
        members = []
        for m in tar.getmembers():
            member_path = (dest / m.name).resolve()
            if not str(member_path).startswith(str(dest.resolve())):
                LOG.warning("isophonics: skipping unsafe tar member %s", m.name)
                continue
            members.append(m)
        tar.extractall(dest, members=members)


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser(
        ISOPHONICS_NAME,
        description="Fetch Isophonics .lab chord annotations and emit a normalized manifest.",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    raw_dir, manifest_path = dataset_paths(args.out, ISOPHONICS_NAME)
    ensure_dirs(raw_dir, manifest_path.parent, dry_run=args.dry_run)

    write_readme(
        raw_dir,
        dataset_name=ISOPHONICS_NAME,
        citation=ISOPHONICS_CITATION,
        license_str=ISOPHONICS_LICENSE,
        manual_steps=(
            "Audio is not included. The manifest's ``audio_path`` field is "
            "``null`` for every entry; the chord head trains directly on the "
            "Harte chord-label time-series.\n\n"
            "isophonics.net occasionally goes down. If a download fails, retry "
            "later or grab a mirror (e.g. via ``mirdata.datasets.beatles``)."
        ),
        dry_run=args.dry_run,
    )

    # Download + extract each sub-collection.
    extract_root = raw_dir / "annotations"
    if not args.dry_run:
        extract_root.mkdir(parents=True, exist_ok=True)

    for subset_name, url in ISOPHONICS_SUBSETS.items():
        archive_path = raw_dir / f"{subset_name}.tar.gz"
        try:
            download([url], archive_path, force=args.force, dry_run=args.dry_run)
        except DownloadError as exc:
            LOG.error("isophonics: failed to download %s: %s", subset_name, exc)
            LOG.error(
                "  Skipping %s — re-run with --force after fixing connectivity.",
                subset_name,
            )
            continue

        subset_dir = extract_root / subset_name
        if args.dry_run:
            LOG.info("[dry-run] would extract %s -> %s", archive_path, subset_dir)
        else:
            subset_dir.mkdir(parents=True, exist_ok=True)
            if not any(subset_dir.iterdir()) or args.force:
                LOG.info("isophonics: extracting %s", archive_path)
                safe_extract_tar(archive_path, subset_dir)

    # Walk the extracted tree and parse every .lab file.
    entries: list[dict[str, Any]] = []
    if not args.dry_run and extract_root.exists():
        for lab_path in sorted(extract_root.rglob("*.lab")):
            rel = lab_path.relative_to(extract_root)
            subset_name = rel.parts[0] if rel.parts else "unknown"
            song_id = lab_path.stem
            full_id = f"isophonics:{subset_name}:{song_id}"
            chords = parse_lab_file(lab_path)
            entries.append(
                {
                    "id": full_id,
                    "audio_path": None,
                    "beats": [],
                    "notes": [],
                    "chords": chords,
                    "key": None,
                    "split": _split_for_id(full_id),
                }
            )

    with open_manifest(manifest_path, dry_run=args.dry_run) as fh:
        for e in entries:
            write_manifest_entry(fh, e)

    counts = manifest_split_counts(entries)
    print_summary(ISOPHONICS_NAME, counts, manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
