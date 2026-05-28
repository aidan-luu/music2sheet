import { useEffect, useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { ApiError, submitTranscribe } from '../api/client';
import { FileDropZone } from '../components/FileDropZone';
import { RecentJobsList } from '../components/RecentJobsList';
import {
  addRecentJob,
  clearRecentJobs,
  getRecentJobs,
  type RecentJob,
} from '../lib/recentJobs';

type Mode = 'file' | 'url';

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
  const [recent, setRecent] = useState<RecentJob[]>([]);

  useEffect(() => {
    setRecent(getRecentJobs());
  }, []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      let res;
      let inputLabel: string;
      if (mode === 'file') {
        if (!file) throw new Error('Please choose an audio file.');
        res = await submitTranscribe({ file });
        inputLabel = file.name;
      } else {
        const trimmed = url.trim();
        if (!trimmed) throw new Error('Please enter a URL.');
        res = await submitTranscribe({ url: trimmed });
        inputLabel = trimmed;
      }
      const next = addRecentJob({
        id: res.job_id,
        createdAt: res.created_at ?? new Date().toISOString(),
        inputLabel,
      });
      setRecent(next);
      navigate(`/jobs/${res.job_id}`);
    } catch (err) {
      setError(formatError(err));
    } finally {
      setSubmitting(false);
    }
  }

  function handleClearRecent(): void {
    clearRecentJobs();
    setRecent([]);
  }

  const canSubmit = mode === 'file' ? !!file : url.trim().length > 0;

  return (
    <section className="upload-page">
      <div className="upload-page__layout">
        <div className="upload-page__main">
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
              <div className="upload-page__field">
                <span className="upload-page__label">Audio file (MP3 or WAV)</span>
                <FileDropZone file={file} onFile={setFile} />
              </div>
            ) : (
              <label className="upload-page__field">
                <span className="upload-page__label">YouTube URL</span>
                <input
                  type="url"
                  placeholder="https://www.youtube.com/watch?v=..."
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                />
              </label>
            )}

            {error && (
              <div role="alert" className="upload-page__error">
                {error}
              </div>
            )}

            <button type="submit" disabled={!canSubmit || submitting}>
              {submitting ? 'Submitting…' : 'Transcribe'}
            </button>
          </form>
        </div>

        <RecentJobsList jobs={recent} onClear={handleClearRecent} />
      </div>
    </section>
  );
}
