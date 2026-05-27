# `ml.datasets.scripts` — dataset download + normalization

This package contains the one-shot CLI scripts that populate the local data
tree consumed by all training jobs in PR-4+.

## Output layout

Every script writes into the same root (default `~/sheet-sage-data`):

```
~/sheet-sage-data/
  raw/
    hooktheory/        # hooktheory.json.gz + hooktheory.json + README.md
    pop909/            # repo/ + README.md
    isophonics/        # annotations/<subset>/*.lab + README.md
    billboard/         # annotations/<slot>/salami_chords.txt + README.md
    rwc_pop/           # annotations/Chord-Annotations/ + README.md
  manifests/
    hooktheory.jsonl
    pop909.jsonl
    isophonics.jsonl
    billboard.jsonl
    rwc_pop.jsonl
```

Each `<dataset>.jsonl` has one JSON object per line:

```json
{
  "id": "hooktheory:abba/dancing-queen",
  "audio_path": null,
  "beats":  [{"time": 0.0, "downbeat": true, "confidence": 1.0}, ...],
  "notes":  [{"pitch": 60, "onset": 0.5, "duration": 0.25, "velocity": 80}, ...],
  "chords": [{"label": "A:maj", "onset": 0.0, "duration": 2.0, "confidence": 1.0}, ...],
  "key":    {"tonic": 9, "mode": "major", "confidence": 1.0},
  "split":  "train"
}
```

Missing modalities are empty lists (or `null` for `key` / `audio_path`) — never
`None` for a list field.

## Running

Each script is a standalone `python -m` entry point:

```bash
python -m ml.datasets.scripts.fetch_hooktheory --out ~/sheet-sage-data
python -m ml.datasets.scripts.fetch_pop909     --out ~/sheet-sage-data
python -m ml.datasets.scripts.fetch_isophonics --out ~/sheet-sage-data
python -m ml.datasets.scripts.fetch_billboard  --out ~/sheet-sage-data --agree
python -m ml.datasets.scripts.fetch_rwc_pop    --out ~/sheet-sage-data \
       --audio-root /path/to/legally/acquired/audio
```

Common flags (every script):

| Flag         | Effect                                                             |
|--------------|--------------------------------------------------------------------|
| `--out DIR`  | Output root (default `~/sheet-sage-data`).                         |
| `--force`    | Re-download / re-clone even if local files exist.                  |
| `--dry-run`  | Print actions; do not download, extract, or write the manifest.    |
| `--log-level`| `DEBUG` / `INFO` / `WARNING` / `ERROR` (default `INFO`).           |

Dataset-specific flags:

* `fetch_billboard.py`: `--agree` skips the interactive research-use prompt.
* `fetch_rwc_pop.py`:   `--audio-root PATH` resolves `audio_path` per track.

## Dataset notes (read before running)

| Dataset    | Audio shipped?         | License                                  | Notes |
|------------|------------------------|------------------------------------------|-------|
| HookTheory | No (YouTube links)     | CC BY-NC-SA 3.0                          | 80/10/10 stratified by artist. If the SheetSage release URL 404s, download `hooktheory.json.gz` manually from the SheetSage releases page and place it in `raw/hooktheory/`, then re-run with `--force`. |
| POP909     | No (aligned MIDI only) | CC BY-NC 4.0                             | `audio_path` points at the bundled MIDI; audio acquisition is the user's responsibility. |
| Isophonics | No                     | CC BY-NC-SA 4.0                          | Chord-only annotations; `audio_path` is always `null`. `isophonics.net` is occasionally down — retry later if a subset fails. |
| Billboard  | No                     | Research use only (research agreement)   | Requires accepting the agreement on the McGill DDMaL site. Pass `--agree` for batch runs. |
| RWC-Pop    | No (commercial)        | Research use (audio: AIST; chord: TMC)   | **HELD-OUT eval set.** Every manifest entry has `split == "test"`. Audio is NOT downloaded; users must legally acquire it from AIST. |

## Idempotency + reproducibility

* Each script is **idempotent**: re-running without `--force` is a no-op if
  the local files already exist.
* Splits are **deterministic**: artist/song ids are hashed (SHA-256) into
  10 buckets and assigned 0-7 → train, 8 → val, 9 → test. Re-running on the
  same upstream data produces the same manifest byte-for-byte.
* All downloads write to `<dest>.part` first and `os.replace()` on success,
  so a Ctrl-C never leaves a half-written archive looking complete.

## Boundaries

These scripts are stdlib-only (`urllib`, `gzip`, `tarfile`, `hashlib`,
`json`, `subprocess`). They DO NOT add to `pyproject.toml`. They DO NOT
load audio (that lives in `ml.audio_io`, a sibling module owned by the
same agent but unchanged by PR-3.5).
