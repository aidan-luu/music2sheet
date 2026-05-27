# music2sheet — web

Frontend for the music2sheet demo. Vite + React + TypeScript.
Renders MusicXML in-browser via [OpenSheetMusicDisplay](https://opensheetmusicdisplay.org/).

## Scripts

```bash
npm install            # install deps (node_modules/ is gitignored)
npm run dev            # dev server at http://localhost:5173
npm run build          # tsc + vite production build into dist/
npm test -- --run      # vitest one-shot
npm run lint           # eslint
```

## Routes

- `/` — upload an MP3/WAV file or paste a YouTube URL.
- `/jobs/:jobId` — poll job status every 2s; when status is `done`, render the score.

## API client

`src/api/client.ts` is a **mock** for PR-A0. `postTranscribe` returns a fake job
id; `getJobStatus` returns `pending`/`running` for the first three polls then
`done` with the bundled `src/fixtures/sample.musicxml` payload.

PR-A1 will regenerate `src/api/types.ts` from `api/openapi.json` (committed by
Agent B) and replace the mock with a real `fetch` client.

## Layout

```
web/
  src/
    api/             # client.ts (mock), types.ts (contract placeholder)
    components/      # MusicXmlViewer.tsx + tests
    pages/           # UploadPage.tsx, JobStatusPage.tsx
    fixtures/        # sample.musicxml (public-domain "Twinkle Twinkle")
    App.tsx, main.tsx, app.css
```
