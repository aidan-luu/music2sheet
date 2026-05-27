import type { JobStatus, TranscribeRequest, TranscribeResponse } from './types';

export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(status: number, message: string, body: unknown = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

function resolveBaseUrl(): string {
  const override = import.meta.env.VITE_API_BASE_URL;
  if (typeof override === 'string' && override.length > 0) {
    return override.replace(/\/$/, '');
  }
  if (import.meta.env.DEV) {
    return '/api';
  }
  return 'http://localhost:8000';
}

async function parseErrorBody(res: Response): Promise<unknown> {
  const text = await res.text().catch(() => '');
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

async function request<T>(
  path: string,
  init: RequestInit,
  signal?: AbortSignal,
): Promise<T> {
  const url = `${resolveBaseUrl()}${path}`;
  const res = await fetch(url, {
    ...init,
    signal,
    headers: {
      Accept: 'application/json',
      ...(init.body ? { 'Content-Type': 'application/json' } : {}),
      ...(init.headers ?? {}),
    },
  });
  if (!res.ok) {
    const body = await parseErrorBody(res);
    throw new ApiError(res.status, `Request failed: ${res.status} ${res.statusText}`, body);
  }
  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

export async function createJob(
  req: TranscribeRequest,
  signal?: AbortSignal,
): Promise<TranscribeResponse> {
  return request<TranscribeResponse>(
    '/transcribe',
    { method: 'POST', body: JSON.stringify(req) },
    signal,
  );
}

/**
 * Upload an audio file as multipart/form-data to `POST /transcribe/upload`.
 *
 * The browser sets the `Content-Type` header (including the multipart boundary)
 * automatically — we must NOT set it manually here.
 */
export async function createJobMultipart(
  audioFile: File,
  signal?: AbortSignal,
): Promise<TranscribeResponse> {
  const form = new FormData();
  form.append('audio_file', audioFile, audioFile.name);
  const url = `${resolveBaseUrl()}/transcribe/upload`;
  const res = await fetch(url, {
    method: 'POST',
    body: form,
    signal,
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) {
    const body = await parseErrorBody(res);
    throw new ApiError(
      res.status,
      `Request failed: ${res.status} ${res.statusText}`,
      body,
    );
  }
  return (await res.json()) as TranscribeResponse;
}

/**
 * Dispatch a transcription request to the right endpoint:
 *  - file present → multipart upload (`POST /transcribe/upload`)
 *  - url present  → JSON body (`POST /transcribe`)
 *
 * Exactly one of `file` / `url` must be provided.
 */
export async function submitTranscribe(
  input: { file?: File | null; url?: string | null },
  signal?: AbortSignal,
): Promise<TranscribeResponse> {
  const hasFile = !!input.file;
  const hasUrl = typeof input.url === 'string' && input.url.trim().length > 0;
  if (hasFile && hasUrl) {
    throw new Error('submitTranscribe: provide either file or url, not both.');
  }
  if (hasFile) {
    return createJobMultipart(input.file as File, signal);
  }
  if (hasUrl) {
    return createJob({ audio_url: (input.url as string).trim() }, signal);
  }
  throw new Error('submitTranscribe: provide a file or a url.');
}

export async function getJob(jobId: string, signal?: AbortSignal): Promise<JobStatus> {
  return request<JobStatus>(`/jobs/${encodeURIComponent(jobId)}`, { method: 'GET' }, signal);
}

export async function fetchMusicXml(url: string, signal?: AbortSignal): Promise<string> {
  const res = await fetch(url, { signal });
  if (!res.ok) {
    throw new ApiError(res.status, `Failed to fetch MusicXML: ${res.status} ${res.statusText}`);
  }
  return res.text();
}
