import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, createJob, getJob } from './client';

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('api/client', () => {
  const fetchSpy = vi.fn<typeof fetch>();

  beforeEach(() => {
    fetchSpy.mockReset();
    vi.stubGlobal('fetch', fetchSpy);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe('createJob', () => {
    it('POSTs JSON to /transcribe and returns the typed response', async () => {
      const body = {
        job_id: '00000000-0000-0000-0000-000000000001',
        created_at: '2026-01-01T00:00:00Z',
        status: 'pending' as const,
      };
      fetchSpy.mockResolvedValueOnce(jsonResponse(201, body));

      const res = await createJob({ audio_url: 'https://example.com/clip.mp3' });

      expect(res).toEqual(body);
      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, init] = fetchSpy.mock.calls[0];
      expect(String(url)).toMatch(/\/transcribe$/);
      expect(init?.method).toBe('POST');
      expect(init?.body).toBe(JSON.stringify({ audio_url: 'https://example.com/clip.mp3' }));
      const headers = init?.headers as Record<string, string>;
      expect(headers['Content-Type']).toBe('application/json');
      expect(headers['Accept']).toBe('application/json');
    });
  });

  describe('getJob', () => {
    it('GETs /jobs/:id and returns the typed status', async () => {
      const body = {
        job_id: 'abc',
        status: 'running' as const,
        progress: 0.5,
      };
      fetchSpy.mockResolvedValueOnce(jsonResponse(200, body));

      const res = await getJob('abc');

      expect(res).toEqual(body);
      const [url, init] = fetchSpy.mock.calls[0];
      expect(String(url)).toMatch(/\/jobs\/abc$/);
      expect(init?.method).toBe('GET');
    });

    it('encodes the job id', async () => {
      fetchSpy.mockResolvedValueOnce(
        jsonResponse(200, { job_id: 'x', status: 'pending', progress: 0 }),
      );
      await getJob('a b/c');
      const [url] = fetchSpy.mock.calls[0];
      expect(String(url)).toContain('/jobs/a%20b%2Fc');
    });
  });

  describe('error handling', () => {
    it('throws ApiError with status and parsed body on 4xx', async () => {
      fetchSpy.mockResolvedValueOnce(jsonResponse(422, { detail: 'Bad input' }));
      let caught: unknown;
      try {
        await createJob({ audio_url: '' });
      } catch (err) {
        caught = err;
      }
      expect(caught).toBeInstanceOf(ApiError);
      const apiErr = caught as ApiError;
      expect(apiErr.status).toBe(422);
      expect(apiErr.body).toEqual({ detail: 'Bad input' });
    });

    it('throws ApiError with status on 5xx', async () => {
      fetchSpy.mockResolvedValueOnce(
        new Response('boom', { status: 503, statusText: 'Service Unavailable' }),
      );
      await expect(getJob('abc')).rejects.toMatchObject({
        name: 'ApiError',
        status: 503,
      });
    });

    it('propagates AbortError when the signal is aborted', async () => {
      const controller = new AbortController();
      fetchSpy.mockImplementationOnce((_url, init) => {
        return new Promise<Response>((_, reject) => {
          init?.signal?.addEventListener('abort', () => {
            reject(new DOMException('Aborted', 'AbortError'));
          });
        });
      });

      const pending = getJob('abc', controller.signal);
      controller.abort();

      await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    });
  });
});
