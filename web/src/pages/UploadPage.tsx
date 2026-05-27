import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { ApiError, createJob } from '../api/client';
import type { TranscribeRequest } from '../api/types';

type Mode = 'file' | 'url';

async function fileToBase64(file: File): Promise<string> {
  const buf = await file.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function formatError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.body && typeof err.body === 'object' && 'detail' in (err.body as object)) {
      const detail = (err.body as { detail: unknown }).detail;
      if (typeof detail === 'string') return detail;
      if (Array.isArray(detail) && detail.length > 0) {
        const first = detail[0] as { msg?: string };
        if (first.msg) return first.msg;
      }
    }
    return `${err.message}`;
  }
  if (err instanceof Error) return err.message;
  return 'Submission failed';
}

export function UploadPage(): JSX.Element {
  const navigate = useNavigate();
  const [mode, setMode] = useState<Mode>('file');
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      let req: TranscribeRequest;
      if (mode === 'file') {
        if (!file) throw new Error('Please choose an audio file.');
        req = { audio_file_b64: await fileToBase64(file) };
      } else {
        if (!url.trim()) throw new Error('Please enter a URL.');
        req = { audio_url: url.trim() };
      }
      const res = await createJob(req);
      navigate(`/jobs/${res.job_id}`);
    } catch (err) {
      setError(formatError(err));
    } finally {
      setSubmitting(false);
    }
  }

  const canSubmit = mode === 'file' ? !!file : url.trim().length > 0;

  return (
    <section className="upload-page">
      <h1>Transcribe a song</h1>
      <p className="upload-page__hint">Upload an audio file or paste a YouTube URL.</p>

      <div className="upload-page__tabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={mode === 'file'}
          onClick={() => setMode('file')}
        >
          Upload file
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === 'url'}
          onClick={() => setMode('url')}
        >
          YouTube URL
        </button>
      </div>

      <form onSubmit={handleSubmit}>
        {mode === 'file' ? (
          <label className="upload-page__field">
            <span>Audio file (MP3 or WAV)</span>
            <input
              type="file"
              accept="audio/mpeg,audio/wav,.mp3,.wav"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>
        ) : (
          <label className="upload-page__field">
            <span>YouTube URL</span>
            <input
              type="url"
              placeholder="https://www.youtube.com/watch?v=..."
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
          </label>
        )}

        {error && <div role="alert" className="upload-page__error">{error}</div>}

        <button type="submit" disabled={!canSubmit || submitting}>
          {submitting ? 'Submitting…' : 'Transcribe'}
        </button>
      </form>
    </section>
  );
}
