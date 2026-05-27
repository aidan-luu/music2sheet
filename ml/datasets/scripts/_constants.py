"""Centralised URLs, paths, and hashes for the dataset fetch scripts.

Keeping every external URL in one place makes it trivial to audit what the
scripts will touch over the network (and to rewrite to a local mirror) without
greping through five separate fetchers. Hashes are SHA-256 hex digests of the
downloaded archive; ``None`` means "no upstream hash published" — the scripts
will still record the on-disk hash in the manifest for reproducibility.
"""

from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# Default output tree.
# ----------------------------------------------------------------------------

DEFAULT_OUT_DIR = Path("~/sheet-sage-data").expanduser()

RAW_SUBDIR = "raw"
MANIFEST_SUBDIR = "manifests"


# ----------------------------------------------------------------------------
# HookTheory.
# ----------------------------------------------------------------------------

HOOKTHEORY_NAME = "hooktheory"
# SheetSage's released JSON dump. We try the v0.1.0 release tag first; if
# upstream renames the asset (it has happened — see issue tracker on the
# chrisdonahue/sheetsage repo) the script falls back to instructing the user
# to download manually.
HOOKTHEORY_URL_CANDIDATES: tuple[str, ...] = (
    "https://github.com/chrisdonahue/sheetsage/releases/download/v0.1.0/hooktheory.json.gz",
    "https://github.com/chrisdonahue/sheetsage/releases/download/v0.1.0/theorytab.json.gz",
)
HOOKTHEORY_RELEASES_PAGE = "https://github.com/chrisdonahue/sheetsage/releases"
HOOKTHEORY_SHA256: str | None = None  # upstream does not publish a hash
HOOKTHEORY_LICENSE = "CC BY-NC-SA 3.0"
HOOKTHEORY_CITATION = (
    "Donahue, C., Thickstun, J., Liang, P. \"Melody Transcription via Generative "
    "Pre-training,\" ISMIR 2022."
)


# ----------------------------------------------------------------------------
# POP909.
# ----------------------------------------------------------------------------

POP909_NAME = "pop909"
POP909_GIT_URL = "https://github.com/music-x-lab/POP909-Dataset.git"
POP909_LICENSE = "CC BY-NC 4.0"
POP909_CITATION = (
    "Wang, Z. et al. \"POP909: A Pop-song Dataset for Music Arrangement "
    "Generation,\" ISMIR 2020."
)


# ----------------------------------------------------------------------------
# Isophonics.
# ----------------------------------------------------------------------------

ISOPHONICS_NAME = "isophonics"
# Five sub-collections, each shipped as a zip of .lab files on isophonics.net.
# Note: isophonics.net is occasionally flaky; the script retries each URL.
ISOPHONICS_SUBSETS: dict[str, str] = {
    "beatles": "http://isophonics.net/files/annotations/The%20Beatles%20Annotations.tar.gz",
    "queen": "http://isophonics.net/files/annotations/Queen%20Annotations.tar.gz",
    "carole_king": "http://isophonics.net/files/annotations/CaroleKing%20Annotations.tar.gz",
    "zweieck": "http://isophonics.net/files/annotations/Zweieck%20Annotations.tar.gz",
    "robbie_williams": "http://isophonics.net/files/annotations/Robbie%20Williams%20Annotations.tar.gz",
}
ISOPHONICS_LICENSE = "CC BY-NC-SA 4.0"
ISOPHONICS_CITATION = (
    "Mauch, M. et al. \"OMRAS2 metadata project 2009,\" ISMIR 2009 (LBD)."
)


# ----------------------------------------------------------------------------
# Billboard / McGill.
# ----------------------------------------------------------------------------

BILLBOARD_NAME = "billboard"
BILLBOARD_AGREEMENT_URL = (
    "https://ddmal.music.mcgill.ca/research/"
    "The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/"
)
# Salami-format chord annotations + matched-listing index. The DDMaL site
# requires acceptance of the research-use agreement; we surface that prompt
# in the script and then download the zips by direct URL.
BILLBOARD_CHORDS_URL = (
    "https://ddmal.music.mcgill.ca/billboard/data/mcgill-billboard-chords-2.0.tar.gz"
)
BILLBOARD_INDEX_URL = (
    "https://ddmal.music.mcgill.ca/billboard/data/mcgill-billboard-index-2.0.tar.gz"
)
BILLBOARD_LICENSE = "Academic research use only (see agreement URL)"
BILLBOARD_CITATION = (
    "Burgoyne, J. A., Wild, J., Fujinaga, I. \"An Expert Ground Truth Set for "
    "Audio Chord Recognition and Music Analysis,\" ISMIR 2011."
)


# ----------------------------------------------------------------------------
# RWC-Pop. HELD-OUT eval set; do NOT use for training.
# ----------------------------------------------------------------------------

RWC_POP_NAME = "rwc_pop"
RWC_POP_AUDIO_INFO_URL = "https://staff.aist.go.jp/m.goto/RWC-MDB/"
# Free chord annotations courtesy of T. Cho / J. P. Bello (NYU MARL).
RWC_POP_CHORD_GIT_URL = "https://github.com/tmc323/Chord-Annotations.git"
RWC_POP_LICENSE = "Research-use only (audio: AIST RWC-MDB; annotations: see repo)"
RWC_POP_CITATION = (
    "Goto, M., Hashiguchi, H., Nishimura, T., Oka, R. \"RWC Music Database: "
    "Popular, Classical and Jazz Music Databases,\" ISMIR 2002."
)
RWC_POP_HELDOUT_BANNER = (
    "==========================================================================\n"
    "  RWC-Pop is the project's HELD-OUT eval set (pass-criterion gate).\n"
    "  DO NOT include any RWC-Pop example in a training-split manifest.\n"
    "  This script writes manifest entries with split == \"test\" exclusively.\n"
    "=========================================================================="
)
