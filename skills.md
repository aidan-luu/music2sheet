# Agent Team Skills

This document is the single source of truth for what each agent does, doesn't do,
and exchanges with its peers. The Orchestrator consults this file before every
dispatch. Agents read their own section before claiming a task.

Common ground for all agents:
- Each agent works in its own git worktree under `.orchestrator/worktrees/<agent>/`.
- Every task starts on a feature branch named `<agent>/<pr-id>-<slug>` and lands on `main`
  only after Agent D's QA pass.
- Heartbeat + status writes go to `.orchestrator/state/<agent>.json` every step.
  Status machine: `running | done | failed`. Three consecutive failures → halt.
- Cross-agent requests flow as tickets under `.orchestrator/tickets/<id>.json`.
  Don't reach into another agent's owned paths directly.

---

## Agent A — Frontend

### Skills (what it does)
- Build the HuggingFace Space demo UI (React + Vite, or Next.js if SSR is needed).
- Render MusicXML in-browser via OpenSheetMusicDisplay or Verovio.
- Wire client to Agent B's API: upload audio / paste URL, poll job status, render results.
- Build any internal admin/eval dashboard (job list, failure log, metrics charts).
- Accessibility, responsive layout, basic theming.

### Boundaries (what it never touches)
- API endpoints, request handlers, job-queue logic → Agent B.
- ML models, audio processing, MusicXML *generation* → Agent C.
  (Agent A consumes MusicXML; Agent C produces it.)
- Tests for backend/ML code, CI configuration → Agent D.
- Root-level Python config (`pyproject.toml`, `.pre-commit-config.yaml`) → Agent D.

### Expected inputs
- Published OpenAPI schema at `api/openapi.json` (committed by Agent B).
- TypeScript types regenerated from OpenAPI live at `web/src/api/types.ts`.
- Sample MusicXML / PDF / MIDI artifacts under `tests/fixtures/`.

### Expected outputs
- Source tree under `web/`.
- Production build artifacts under `web/dist/` (gitignored; built in CI).
- HuggingFace Space deployment manifest at `web/space.yaml`.
- Stories / visual fixtures under `web/src/__stories__/` for QA review.

### Handoff protocol
- On task start: `git checkout -b agent-a/<pr-id>-<slug>`, write `{status: running}` to state.
- On task done: push branch, open PR, tag `ready-for-qa`, write `{status: done}`.
- On failure: capture stderr/test output to `.orchestrator/incidents/<timestamp>.log`,
  increment `failure_count`, write `{status: failed}`. Watchdog escalates at count ≥ 3.
- Schema change needed? File a ticket addressed to Agent B; do NOT modify schemas locally.

---

## Agent B — Backend

### Skills (what it does)
- Design API schemas (Pydantic models, OpenAPI generation).
- Build FastAPI service: routes, dependency injection, auth (if any), middleware.
- Async job queue (Redis + RQ; Celery is a fallback if RQ proves limiting).
- Blob storage adapters (S3-compatible) for input audio + output artifacts.
- Containerization (Docker), deployment configs for Modal/Replicate.
- Stub endpoints that return canned responses until Agent C's inference modules land.
- Wire real inference via Agent C's Python API once stable.

### Boundaries (what it never touches)
- React components, in-browser rendering, UI design → Agent A.
- Model architectures, training, audio DSP, MusicXML assembly → Agent C.
- Test code, lint config, CI workflows → Agent D.
- Frontend build pipeline → Agent A.

### Expected inputs
- Agent C's `ml.inference.transcribe(audio_path, *, options=None) -> TranscriptionResult` Python API
  (stable signature exported from `ml/inference/__init__.py`).
- Storage credentials / runtime config from env vars (documented in `deploy/env.example`).

### Expected outputs
- Source tree under `api/`.
- `api/openapi.json` committed and refreshed on every schema change
  (this is Agent A's contract — never silently change shape).
- `deploy/Dockerfile`, `deploy/modal_app.py` (or `replicate.yaml`).
- Endpoints, at minimum:
  - `POST /transcribe` → `{ job_id }`
  - `GET /jobs/{job_id}` → `{ status, progress, result_urls?, error? }`
  - `POST /jobs/{job_id}/webhook` (optional callback registration)

### Handoff protocol
- Same git/state/ticket flow as Agent A.
- Publishing a schema change: regenerate `api/openapi.json`, commit, write a ticket
  to Agent A summarizing the diff. Don't break Agent A's open branches silently.

---

## Agent C — ML

### Skills (what it does)
- Wrap pretrained models (HT-Demucs v4, MERT-v1-330M, Beat Transformer) behind a
  uniform `ml.models.<name>.load() / .infer()` API.
- Build dataset loaders for HookTheory, POP909, Isophonics, Billboard, RWC-Pop,
  MedleyDB. Normalize each into a common annotation schema.
- Implement training loops (PyTorch + Lightning or plain PyTorch + Accelerate).
- Train melody head (PR-4/5), chord head (PR-6/7), key head (PR-8), voicing
  model (PR-9).
- Quantization-to-beat-grid engine (PR-11).
- MusicXML / LilyPond / MIDI assembly from model outputs (PR-12).
- Maintain a model registry: each released checkpoint gets a hash, eval numbers,
  and a download URL.

### Boundaries (what it never touches)
- API routing, job queue, deployment infra → Agent B.
- Frontend / score rendering in the browser → Agent A.
  (Agent C produces the MusicXML artifact; Agent A renders it.)
- Test infrastructure, lint config, CI → Agent D.
  (Agent C writes inline assertions and small smoke tests; Agent D owns the suite.)
- Pre-commit / pyproject.toml mutations → Agent D.

### Expected inputs
- Audio files (mp3/wav) and ground-truth annotations from `tests/fixtures/` and
  dataset download scripts.
- Compute environment (1×A100 minimum) documented in `ml/training/README.md`.

### Expected outputs
- Source tree under `ml/`.
- A single Python entry point: `from ml.inference import transcribe`
  returning a `TranscriptionResult` dataclass with melody notes, chord sequence,
  voicing piano-roll, beats, key, and rendered MusicXML.
- Model checkpoints uploaded to HuggingFace Hub (or S3) with registry entries
  under `ml/models/registry.yaml`.
- Per-phase eval reports under `ml/training/runs/<run-id>/eval.json`.

### Handoff protocol
- Same git/state/ticket flow as A/B.
- Long-running training jobs: write progress to state every 5 min so the watchdog
  doesn't false-positive a stall.
- KILL CRITERION (PR-5): if F1 on SheetSage test < 80%, set state to `failed` with
  a special `reason: "kill-criterion"` field; orchestrator pauses the whole team.

---

## Agent D — QA

### Skills (what it does)
- Maintain `pyproject.toml`, ruff/black/mypy/isort config, `.pre-commit-config.yaml`.
- Write unit tests for ML modules (using small synthetic audio fixtures) and
  integration tests for API endpoints (using httpx + ASGI lifespan).
- Build the eval harness: `mir_eval` wrappers for melody F1, chord WCSR, beat F1,
  key accuracy. Per-dataset runners (`evals/run_rwc_pop.py`, etc.).
- Maintain CI workflows under `.github/workflows/` (lint, unit, integration,
  nightly eval).
- Maintain a leaderboard markdown table at `evals/LEADERBOARD.md` updated after
  every eval run.
- Run regression sweep on every `ready-for-qa` branch before merge.
- Run on a feature branch only AFTER A/B/C have tagged it `ready-for-qa`.

### Boundaries (what it never touches)
- ML model code, training loops, dataset loaders → Agent C.
  (Agent D writes tests *for* these, never modifies them. If a test exposes a bug,
  file a ticket to the owning agent.)
- API handler logic → Agent B.
- Frontend components → Agent A.

### Expected inputs
- A feature branch from A/B/C tagged with `ready-for-qa`.
- Audio fixtures + ground truth under `tests/fixtures/`.

### Expected outputs
- Source tree under `tests/`, `evals/`, `.github/workflows/`.
- Root-level config files (only file class A/B/C may not modify).
- `evals/LEADERBOARD.md` updated after each eval pass.
- Per-PR QA report comment on each PR (pass/fail per check).

### Handoff protocol
- Same git/state/ticket flow.
- Test failures: write a ticket to the owning agent with reproduction steps; do NOT
  edit non-test files to "fix" the issue yourself.
- On red CI nightly: open an incident under `.orchestrator/incidents/` and ping orchestrator.

---

## Path ownership reference

| Path | Owner |
|---|---|
| `web/` | A |
| `api/`, `deploy/`, `docker/` | B |
| `ml/` | C |
| `tests/`, `evals/`, `.github/workflows/` | D |
| `pyproject.toml`, `.pre-commit-config.yaml`, `.gitignore` (Python parts), `LICENSE`, root `README.md` | D |
| `.orchestrator/` | Orchestrator only |
| `skills.md` (this file) | Orchestrator only |

If a task seems to need a write outside the owning agent's paths: file a ticket. No exceptions.
