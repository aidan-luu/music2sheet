"""Fetch + normalise the McGill Billboard chord-annotation dataset.

Usage::

    python -m ml.datasets.scripts.fetch_billboard --out ~/sheet-sage-data

The McGill Billboard Project distributes annotations for a stratified sample
of Billboard Hot 100 chart slots from 1958-1991. Annotations are in the
SALAMI-style "chord text" format, which is denser than plain Harte (it
includes section labels, time-signature events, and metric position) — we
parse out the chord segments only and normalise to Harte.

The DDMaL site requires acceptance of a research-use agreement. This script
prints the agreement URL and pauses for the user to type ``yes`` (or pass
``--agree`` to skip the interactive prompt in batch runs).

Audio is NOT distributed. ``audio_path`` is ``null`` in every manifest entry.

Stratified split: 80/10/10 train/val/test by Billboard chart slot id.
"""

from __future__ import annotations

import hashlib
import logging
import re
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
    BILLBOARD_AGREEMENT_URL,
    BILLBOARD_CHORDS_URL,
    BILLBOARD_CITATION,
    BILLBOARD_INDEX_URL,
    BILLBOARD_LICENSE,
    BILLBOARD_NAME,
)

LOG = logging.getLogger("ml.datasets.scripts.fetch_billboard")


def _split_for_id(slot_id: str) -> str:
    bucket = int.from_bytes(
        hashlib.sha256(slot_id.encode("utf-8")).digest()[:4], "big"
    ) % 10
    if bucket < 8:
        return "train"
    if bucket < 9:
        return "val"
    return "test"


# Lines we care about in salami_chords.txt look like:
#   0.000000000      silence
#   0.881632653      A, intro, | A | A | E | E |, (drum)
# We extract the leading float and any pipe-delimited chord tokens.
_TIME_LINE_RE = re.compile(r"^\s*([0-9.]+)\s+(.*)$")
_CHORD_TOKEN_RE = re.compile(r"\|([^|]+)\|")


def parse_salami_chords(path: Path) -> list[dict[str, Any]]:
    """Parse one ``salami_chords.txt`` into a list of chord segments.

    SALAMI annotations interleave time-stamped section markers with chord
    measures grouped between ``|`` pipes. We allocate each measure's duration
    evenly across its chord tokens, which is a simplification (real measures
    can have 2 chords on beats 1+3, etc.) but it is what the McGill team's
    own evaluation scripts do as a fallback.
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        LOG.warning("billboard: cannot read %s: %s", path, exc)
        return []

    # First pass: collect (time, chord_tokens) for each annotation line.
    timed: list[tuple[float, list[str]]] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _TIME_LINE_RE.match(line)
        if not m:
            continue
        try:
            t = float(m.group(1))
        except ValueError:
            continue
        rest = m.group(2)
        tokens = [tok.strip() for tok in _CHORD_TOKEN_RE.findall(rest)]
        if not tokens:
            # Words like "silence" / "end" act as zero-token markers but we
            # still want their timestamp to close the previous segment.
            tokens = [_normalise_label(rest)]
        timed.append((t, tokens))

    # Second pass: expand chord tokens between successive timestamps.
    chords: list[dict[str, Any]] = []
    for (t0, tokens), (t1, _) in zip(timed, timed[1:]):
        if not tokens:
            continue
        n = len(tokens)
        if n == 0 or t1 <= t0:
            continue
        per = (t1 - t0) / n
        for i, tok in enumerate(tokens):
            label = _normalise_label(tok)
            chords.append(
                {
                    "label": label,
                    "onset": t0 + i * per,
                    "duration": per,
                    "confidence": 1.0,
                }
            )
    return chords


def _normalise_label(token: str) -> str:
    """Map a SALAMI chord token to a Harte label.

    SALAMI tokens are already close to Harte (e.g. ``A:min``, ``G:7/B``); the
    only common variants we collapse are ``N`` / ``silence`` / ``end`` -> the
    Harte no-chord symbol ``N``.
    """
    tok = token.strip()
    if not tok:
        return "N"
    low = tok.lower()
    if low in {"n", "silence", "end", "applause", "fadeout", "fade-out"}:
        return "N"
    return tok


def _prompt_agreement(skip: bool) -> bool:
    if skip:
        LOG.info("billboard: --agree passed, skipping interactive agreement prompt.")
        return True
    print(
        "\nThe McGill Billboard chord dataset is distributed under a research-use\n"
        "agreement. Please read it at:\n\n  "
        f"{BILLBOARD_AGREEMENT_URL}\n\n"
        "Type 'yes' to confirm you have read and accept the terms: ",
        end="",
        flush=True,
    )
    try:
        resp = input().strip().lower()
    except EOFError:
        return False
    return resp == "yes"


def safe_extract_tar(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:*") as tar:
        members = []
        for m in tar.getmembers():
            member_path = (dest / m.name).resolve()
            if not str(member_path).startswith(str(dest.resolve())):
                LOG.warning("billboard: skipping unsafe tar member %s", m.name)
                continue
            members.append(m)
        tar.extractall(dest, members=members)


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser(
        BILLBOARD_NAME,
        description="Fetch McGill Billboard chord annotations and emit a normalized manifest.",
    )
    parser.add_argument(
        "--agree",
        action="store_true",
        help="Confirm the research-use agreement non-interactively.",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    raw_dir, manifest_path = dataset_paths(args.out, BILLBOARD_NAME)
    ensure_dirs(raw_dir, manifest_path.parent, dry_run=args.dry_run)

    if not args.dry_run and not _prompt_agreement(args.agree):
        LOG.error("billboard: agreement not accepted; aborting. Re-run with --agree.")
        return 3

    chords_archive = raw_dir / "mcgill-billboard-chords.tar.gz"
    index_archive = raw_dir / "mcgill-billboard-index.tar.gz"

    failed_downloads = False
    for url, target in (
        (BILLBOARD_CHORDS_URL, chords_archive),
        (BILLBOARD_INDEX_URL, index_archive),
    ):
        try:
            download([url], target, force=args.force, dry_run=args.dry_run)
        except DownloadError as exc:
            failed_downloads = True
            LOG.error(
                "billboard: download failed for %s: %s.\n"
                "  Open %s in a browser, accept the agreement, and download the "
                "archives manually into %s.",
                url,
                exc,
                BILLBOARD_AGREEMENT_URL,
                raw_dir,
            )

    write_readme(
        raw_dir,
        dataset_name=BILLBOARD_NAME,
        citation=BILLBOARD_CITATION,
        license_str=BILLBOARD_LICENSE,
        manual_steps=(
            f"Agreement: {BILLBOARD_AGREEMENT_URL}\n\n"
            "If the automated download fails (the McGill site sometimes blocks "
            "non-browser clients), open the agreement URL in a browser, accept "
            "the terms, and drop the resulting ``mcgill-billboard-chords-2.0.tar.gz`` "
            "and ``mcgill-billboard-index-2.0.tar.gz`` into this directory, then "
            "re-run the script with --force --agree.\n\n"
            "Audio is not included."
        ),
        dry_run=args.dry_run,
    )

    if failed_downloads and not args.dry_run:
        return 2

    extract_root = raw_dir / "annotations"
    if not args.dry_run:
        extract_root.mkdir(parents=True, exist_ok=True)
        for archive in (chords_archive, index_archive):
            if not archive.exists():
                continue
            LOG.info("billboard: extracting %s", archive)
            safe_extract_tar(archive, extract_root)

    entries: list[dict[str, Any]] = []
    if not args.dry_run and extract_root.exists():
        for salami_path in sorted(extract_root.rglob("salami_chords.txt")):
            slot_dir = salami_path.parent
            slot_id = slot_dir.name
            full_id = f"billboard:{slot_id}"
            chords = parse_salami_chords(salami_path)
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
    print_summary(BILLBOARD_NAME, counts, manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
