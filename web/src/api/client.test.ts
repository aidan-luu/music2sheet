import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  ApiError,
  createJob,
  createJobMultipart,
  getJob,
  submitTranscribe,
} from './client';

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

  describe('createJobMultipart', () => {
    it('POSTs FormData to /transcribe/upload and returns the typed response', async () => {
      const body = {
        job_id: '00000000-0000-0000-0000-000000000002',
        created_at: '2026-01-02T00:00:00Z',
        status: 'pending' as const,
      };
      fetchSpy.mockResolvedValueOnce(jsonResponse(201, body));

      const file = new File([new Uint8Array([1, 2, 3, 4])], 'song.mp3', {
        type: 'audio/mpeg',
      });
      const res = await createJobMultipart(file);

      expect(res).toEqual(body);
      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, init] = fetchSpy.mock.calls[0];
      expect(String(url)).toMatch(/\/transcribe\/upload$/);
      expect(init?.method).toBe('POST');

      // Body must be FormData carrying the file under `audio_file`.
      const sentBody = init?.body;
      expect(sentBody).toBeInstanceOf(FormData);
      const form = sentBody as FormData;
      const sentFile = form.get('audio_file');
      expect(sentFile).toBeInstanceOf(File);
      expect((sentFile as File).name).toBe('song.mp3');
      expect((sentFile as File).type).toBe('audio/mpeg');

      // Critical: we must NOT set Content-Type ourselves — the browser sets
      // it (with the multipart boundary). Header bag is allowed to carry
      // only Accept.
      const headers = init?.headers as Record<string, string> | undefined;
      if (headers) {
        expect(headers['Content-Type']).toBeUndefined();
        expect(headers['content-type']).toBeUndefined();
      }
    });

    it('throws ApiError with parsed body on 4xx', async () => {
      fetchSpy.mockResolvedValueOnce(jsonResponse(413, { detail: 'File too large' }));
      const file = new File([new Uint8Array(8)], 'big.wav', { type: 'audio/wav' });

      let caught: unknown;
      try {
        await createJobMultipart(file);
      } catch (err) {
        caught = err;
      }
      expect(caught).toBeInstanceOf(ApiError);
      const apiErr = caught as ApiError;
      expect(apiErr.status).toBe(413);
      expect(apiErr.body).toEqual({ detail: 'File too large' });
    });

    it('passes the AbortSignal through to fetch', async () => {
      const controller = new AbortController();
      fetchSpy.mockImplementationOnce((_url, init) => {
        return new Promise<Response>((_, reject) => {
          init?.signal?.addEventListener('abort', () => {
            reject(new DOMException('Aborted', 'AbortError'));
          });
        });
      });
      const file = new File([new Uint8Array(2)], 'a.mp3', { type: 'audio/mpeg' });
      const pending = createJobMultipart(file, controller.signal);
      controller.abort();
      await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    });
  });

  describe('submitTranscribe', () => {
    it('dispatches a file to the multipart endpoint', async () => {
      const body = {
        job_id: '00000000-0000-0000-0000-000000000003',
        created_at: '2026-01-03T00:00:00Z',
        status: 'pending' as const,
      };
      fetchSpy.mockResolvedValueOnce(jsonResponse(201, body));

      const file = new File([new Uint8Array(4)], 'pick.wav', { type: 'audio/wav' });
      const res = await submitTranscribe({ file });

      expect(res).toEqual(body);
      const [url, init] = fetchSpy.mock.calls[0];
      expect(String(url)).toMatch(/\/transcribe\/upload$/);
      expect(init?.body).toBeInstanceOf(FormData);
    });

    it('dispatches a URL to the JSON /transcribe endpoint', async () => {
      const body = {
        job_id: '00000000-0000-0000-0000-000000000004',
        created_at: '2026-01-04T00:00:00Z',
        status: 'pending' as const,
      };
      fetchSpy.mockResolvedValueOnce(jsonResponse(201, body));

      const res = await submitTranscribe({ url: '  https://example.com/clip.mp3  ' });

      expect(res).toEqual(body);
      const [url, init] = fetchSpy.mock.calls[0];
      expect(String(url)).toMatch(/\/transcribe$/);
      expect(String(url)).not.toMatch(/\/transcribe\/upload$/);
      expect(init?.method).toBe('POST');
      // URL is trimmed before being sent.
      expect(init?.body).toBe(
        JSON.stringify({ audio_url: 'https://example.com/clip.mp3' }),
      );
      const headers = init?.headers as Record<string, string>;
      expect(headers['Content-Type']).toBe('application/json');
    });

    it('throws when neither file nor url is provided', async () => {
      await expect(submitTranscribe({})).rejects.toThrow(/file or a url/);
      await expect(submitTranscribe({ url: '   ' })).rejects.toThrow(/file or a url/);
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it('throws when both file and url are provided', async () => {
      const file = new File([new Uint8Array(1)], 'x.mp3', { type: 'audio/mpeg' });
      await expect(
        submitTranscribe({ file, url: 'https://example.com/a.mp3' }),
      ).rejects.toThrow(/either file or url, not both/);
      expect(fetchSpy).not.toHaveBeenCalled();
    });
  });
});
