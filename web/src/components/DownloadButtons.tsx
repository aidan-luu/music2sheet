import { useState } from 'react';
import type { JobResultUrls } from '../api/types';

interface Props {
  jobId: string;
  urls: JobResultUrls;
}

type Kind = 'musicxml' | 'midi' | 'pdf';

const SPECS: ReadonlyArray<{ kind: Kind; label: string; ext: string }> = [
  { kind: 'musicxml', label: 'Download MusicXML', ext: 'musicxml' },
  { kind: 'midi', label: 'Download MIDI', ext: 'mid' },
  { kind: 'pdf', label: 'Download PDF', ext: 'pdf' },
];

async function downloadArtifact(
  url: string,
  filename: string,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(url, { headers: { Accept: '*/*' }, signal });
  if (!res.ok) {
    const err = new Error(`Download failed (${res.status} ${res.statusText})`);
    (err as Error & { status?: number }).status = res.status;
    throw err;
  }
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  try {
    const a = document.createElement('a');
    a.href = objectUrl;
    a.download = filename;
    a.rel = 'noopener';
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    // Defer revoke so the browser has time to start the download.
    setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
  }
}

export function DownloadButtons({ jobId, urls }: Props): JSX.Element {
  const [pending, setPending] = useState<Kind | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  async function handle(kind: Kind, ext: string): Promise<void> {
    setToast(null);
    setPending(kind);
    try {
      await downloadArtifact(urls[kind], `${jobId}.${ext}`);
    } catch (err) {
      const status = (err as { status?: number }).status;
      if (status === 404) {
        setToast(`${kind.toUpperCase()} not available yet.`);
      } else if (err instanceof Error) {
        setToast(err.message);
      } else {
        setToast('Download failed');
      }
    } finally {
      setPending(null);
    }
  }

  return (
    <div className="downloads">
      <div className="downloads__buttons">
        {SPECS.map((s) => (
          <button
            key={s.kind}
            type="button"
            className="downloads__button"
            disabled={pending !== null}
            onClick={() => void handle(s.kind, s.ext)}
            aria-busy={pending === s.kind}
          >
            {pending === s.kind ? 'Downloading…' : s.label}
          </button>
        ))}
      </div>
      {toast && (
        <div role="status" className="downloads__toast">
          {toast}
        </div>
      )}
    </div>
  );
}
