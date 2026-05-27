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
