import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { getJobStatus } from '../api/client';
import type { JobStatus } from '../api/types';
import { MusicXmlViewer } from '../components/MusicXmlViewer';

const POLL_MS = 2000;

export function JobStatusPage(): JSX.Element {
  const { jobId } = useParams<{ jobId: string }>();
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    async function poll(): Promise<void> {
      try {
        const next = await getJobStatus(jobId!);
        if (cancelled) return;
        setStatus(next);
        if (next.status !== 'done' && next.status !== 'error') {
          timer = setTimeout(poll, POLL_MS);
        }
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Polling failed');
      }
    }
    void poll();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [jobId]);

  if (!jobId) {
    return <p>Missing job id.</p>;
  }

  return (
    <section className="job-status">
      <Link to="/">&larr; New transcription</Link>
      <h1>Job {jobId}</h1>

      {error && <div role="alert" className="job-status__error">{error}</div>}

      {!status && <p>Loading job status…</p>}

      {status && status.status !== 'done' && status.status !== 'error' && (
        <div className="job-status__progress">
          <p>Status: {status.status}</p>
          <progress value={status.progress} max={1} />
        </div>
      )}

      {status?.status === 'error' && (
        <div role="alert" className="job-status__error">
          Job failed: {status.error ?? 'unknown error'}
        </div>
      )}

      {status?.status === 'done' && status.musicxml && (
        <div className="job-status__result">
          <h2>Result</h2>
          <MusicXmlViewer musicxml={status.musicxml} />
        </div>
      )}
    </section>
  );
}
