"""Shared helpers for the fetch_* scripts.

All five fetchers need the same boilerplate:

* Build a ``--out`` / ``--force`` / ``--dry-run`` argparse parser.
* Lay out the output tree (``<out>/raw/<dataset>/`` and
  ``<out>/manifests/<dataset>.jsonl``).
* Download a URL to a file with retry + SHA-256 verification, skipping the
  download when the file already exists unless ``--force`` is passed.
* Write a per-dataset README with citation + license.
* Print the final ``[dataset] N train / N val / N test ...`` summary.

We keep this module dependency-free (stdlib only) so the scripts remain
runnable on a fresh Python 3.11 install without pip installing anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ml.datasets.scripts._constants import (
    DEFAULT_OUT_DIR,
    MANIFEST_SUBDIR,
    RAW_SUBDIR,
)

LOG = logging.getLogger("ml.datasets.scripts")


# ----------------------------------------------------------------------------
# CLI scaffolding.
# ----------------------------------------------------------------------------


def build_argparser(dataset_name: str, description: str) -> argparse.ArgumentParser:
    """Construct the ``--out`` / ``--force`` / ``--dry-run`` parser.

    Each fetch script wraps this and may add dataset-specific flags (e.g.
    Billboard adds ``--agree`` to skip the interactive agreement prompt).
    """
    parser = argparse.ArgumentParser(
        prog=f"python -m ml.datasets.scripts.fetch_{dataset_name}",
        description=description,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output root directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if local files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the actions that would be taken; do not download or write files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default: INFO).",
    )
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


# ----------------------------------------------------------------------------
# Output tree.
# ----------------------------------------------------------------------------


def dataset_paths(out_root: Path, dataset_name: str) -> tuple[Path, Path]:
    """Return ``(raw_dir, manifest_path)`` for one dataset under ``out_root``."""
    raw_dir = out_root / RAW_SUBDIR / dataset_name
    manifest_path = out_root / MANIFEST_SUBDIR / f"{dataset_name}.jsonl"
    return raw_dir, manifest_path


def ensure_dirs(*dirs: Path, dry_run: bool = False) -> None:
    for d in dirs:
        if dry_run:
            LOG.info("[dry-run] mkdir -p %s", d)
        else:
            d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Downloads.
# ----------------------------------------------------------------------------


class DownloadError(RuntimeError):
    """Raised when every candidate URL for a file fails."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(
    urls: Iterable[str],
    dest: Path,
    *,
    expected_sha256: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    timeout: float = 60.0,
) -> Path:
    """Download the first reachable URL in ``urls`` to ``dest``.

    Skips the download (returns ``dest``) when the file exists and ``force`` is
    False. When ``expected_sha256`` is non-None, the function verifies the
    on-disk hash and raises :class:`DownloadError` on mismatch.
    """
    if dest.exists() and not force:
        LOG.info("download: %s already exists, skipping (use --force to redownload)", dest)
        return dest

    if dry_run:
        LOG.info("[dry-run] would download one of %s -> %s", list(urls), dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    tmp = dest.with_suffix(dest.suffix + ".part")
    for url in urls:
        LOG.info("download: GET %s", url)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp, tmp.open("wb") as out:
                shutil.copyfileobj(resp, out)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            LOG.warning("download: %s failed: %s", url, exc)
            last_err = exc
            tmp.unlink(missing_ok=True)
            continue
        tmp.replace(dest)
        break
    else:
        raise DownloadError(
            f"All download URLs failed for {dest.name}: last error: {last_err}"
        )

    if expected_sha256 is not None:
        observed = _sha256_file(dest)
        if observed.lower() != expected_sha256.lower():
            raise DownloadError(
                f"SHA-256 mismatch for {dest}: expected {expected_sha256}, got {observed}"
            )
    return dest


def git_clone(
    repo_url: str,
    dest: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> Path:
    """Shallow-clone ``repo_url`` into ``dest``.

    Skips when ``dest`` already exists unless ``force`` is True (in which case
    the existing directory is removed first). We shell out to ``git`` rather
    than fetching a tarball because POP909 and the RWC chord annotations are
    distributed only via GitHub repositories.
    """
    if dest.exists() and not force:
        LOG.info("git_clone: %s already exists, skipping (use --force to refresh)", dest)
        return dest

    if dry_run:
        LOG.info("[dry-run] git clone --depth 1 %s %s", repo_url, dest)
        return dest

    if dest.exists() and force:
        LOG.info("git_clone: removing existing %s before re-clone", dest)
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", repo_url, str(dest)]
    LOG.info("git_clone: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return dest


# ----------------------------------------------------------------------------
# Manifests.
# ----------------------------------------------------------------------------


@contextmanager
def open_manifest(
    manifest_path: Path,
    *,
    dry_run: bool = False,
) -> Iterator[Any]:
    """Open ``manifest_path`` for writing, or a no-op stub in dry-run mode."""
    if dry_run:
        LOG.info("[dry-run] would write manifest %s", manifest_path)

        class _Stub:
            def write(self, line: str) -> None:
                LOG.debug("[dry-run] manifest << %s", line.rstrip())

        yield _Stub()
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        yield fh


def write_manifest_entry(fh: Any, entry: dict[str, Any]) -> None:
    """Write one JSON line to a manifest file (or stub)."""
    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def manifest_split_counts(entries: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for e in entries:
        split = e.get("split", "train")
        counts[split] = counts.get(split, 0) + 1
    return counts


# ----------------------------------------------------------------------------
# READMEs.
# ----------------------------------------------------------------------------


def write_readme(
    raw_dir: Path,
    *,
    dataset_name: str,
    citation: str,
    license_str: str,
    manual_steps: str = "",
    extra_banner: str = "",
    dry_run: bool = False,
) -> Path:
    """Write a per-dataset README.md.

    The README is the user-facing artifact that documents the citation, the
    license terms, and any out-of-band steps (sign an agreement, source audio
    elsewhere, etc.) so the data tree is self-describing.
    """
    readme_path = raw_dir / "README.md"
    body_parts: list[str] = []
    if extra_banner:
        body_parts.append(extra_banner)
        body_parts.append("")
    body_parts.append(f"# {dataset_name}")
    body_parts.append("")
    body_parts.append(f"**License:** {license_str}")
    body_parts.append("")
    body_parts.append("## Citation")
    body_parts.append("")
    body_parts.append(citation)
    if manual_steps:
        body_parts.append("")
        body_parts.append("## Manual steps")
        body_parts.append("")
        body_parts.append(manual_steps)
    body_parts.append("")
    body = "\n".join(body_parts)

    if dry_run:
        LOG.info("[dry-run] would write README at %s (%d bytes)", readme_path, len(body))
        return readme_path

    raw_dir.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(body, encoding="utf-8")
    return readme_path


# ----------------------------------------------------------------------------
# Final summary.
# ----------------------------------------------------------------------------


def print_summary(dataset_name: str, counts: dict[str, int], manifest_path: Path) -> None:
    """Print the one-line summary line every script ends with."""
    line = (
        f"[{dataset_name}] "
        f"{counts.get('train', 0)} train / "
        f"{counts.get('val', 0)} val / "
        f"{counts.get('test', 0)} test "
        f"examples in {manifest_path}"
    )
    print(line, file=sys.stdout, flush=True)
