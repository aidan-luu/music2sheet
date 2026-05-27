import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { postTranscribe } from '../api/client';

type Mode = 'file' | 'url';

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
      const req =
        mode === 'file' && file
          ? { source: { kind: 'file', file } as const }
          : { source: { kind: 'url', url } as const };
      const res = await postTranscribe(req);
      navigate(`/jobs/${res.job_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Submission failed');
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
