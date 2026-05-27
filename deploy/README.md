# Deploy

Build and run the music2sheet API container locally:

```bash
docker build -f deploy/Dockerfile -t music2sheet-api .
docker run --rm -p 8000:8000 music2sheet-api
curl http://localhost:8000/healthz   # {"status":"ok"}
```

The image only installs the `api` extras (FastAPI + Pydantic + uvicorn +
Redis/RQ clients). The ML stack is intentionally absent from this image — the
inference workers will run in a separate image once PR-14 lands.

## Production targets

- **Modal** — `deploy/modal_app.py` (TBD, PR-14a) will wrap `api.main:app`
  with `modal.asgi_app()` and define a GPU function for inference workers.
- **Replicate** — alternative target; would consume the same image via a
  `replicate.yaml` (TBD).

For now the container is suitable for local development and HuggingFace Space
hosting (CPU-only, stubbed inference).
