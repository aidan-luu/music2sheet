// Placeholder types matching the planned contract.
// PR-A1 will regenerate from api/openapi.json once Agent B commits it.

export type JobState = 'pending' | 'running' | 'done' | 'error';

export interface TranscribeRequest {
  /** Either an uploaded audio file (multipart) or a YouTube URL. */
  source: { kind: 'file'; file: File } | { kind: 'url'; url: string };
}

export interface TranscribeResponse {
  job_id: string;
}

export interface JobStatus {
  job_id: string;
  status: JobState;
  progress: number;
  result_urls?: {
    musicxml?: string;
    midi?: string;
    pdf?: string;
  };
  /** Inlined MusicXML payload when available (mock convenience for PR-A0). */
  musicxml?: string;
  error?: string;
}
