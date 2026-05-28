import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ApiError, getJob } from '../api/client';
import { isTerminal, type JobStatus } from '../api/types';
import { MusicXmlViewer } from '../components/MusicXmlViewer';
import { DownloadButtons } from '../components/DownloadButtons';

const POLL_MS = 2000;

export function JobStatusPage(): JSX.Element {
  const { jobId } = useParams<{ jobId: string }>();
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | null = null;
    let stopped = false;

    async function poll(): Promise<void> {
      try {
        const next = await getJob(jobId!, controller.signal);
        if (stopped) return;
        setStatus(next);
        if (!isTerminal(next.status)) {
          timer = setTimeout(poll, POLL_MS);
        }
      } catch (err) {
        if (stopped || (err instanceof DOMException && err.name === 'AbortError')) return;
        if (err instanceof ApiError) {
          setError(`Status request failed (${err.status})`);
        } else if (err instanceof Error) {
          setError(err.message);
        } else {
          setError('Polling failed');
        }
      }
    }
    void poll();

    return () => {
      stopped = true;
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [jobId]);

  if (!jobId) {
    return <p>Missing job id.</p>;
  }

  const musicxmlUrl = status?.status === 'done' ? status.result_urls?.musicxml : undefined;

  return (
    <section className="job-status">
      <Link to="/">&larr; New transcription</Link>
      <h1>Job {jobId}</h1>

      {error && <div role="alert" className="job-status__error">{error}</div>}

      {!status && !error && <p>Loading job status…</p>}

      {status && !isTerminal(status.status) && (
        <div className="job-status__progress">
          <p>Status: {status.status}</p>
          <progress value={status.progress} max={1} />
        </div>
      )}

      {status?.status === 'failed' && (
        <div role="alert" className="job-status__error">
          Job failed: {status.error ?? 'unknown error'}
        </div>
      )}

      {status?.status === 'done' && status.result_urls && (
        <div className="job-status__result">
          <h2>Result</h2>
          <DownloadButtons jobId={jobId} urls={status.result_urls} />
          {musicxmlUrl ? (
            <MusicXmlViewer musicxmlUrl={musicxmlUrl} />
          ) : (
            <p>No MusicXML preview available.</p>
          )}
        </div>
      )}

      {status?.status === 'done' && !status.result_urls && (
        <p>Job finished but no result URLs were returned.</p>
      )}
    </section>
  );
}
