FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY api ./api
RUN pip install --no-cache-dir --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels ".[api]"

FROM python:3.11-slim AS runtime
WORKDIR /app
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /wheels /wheels
COPY api ./api
COPY pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir --no-index --find-links=/wheels music2sheet[api] \
 && rm -rf /wheels
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
