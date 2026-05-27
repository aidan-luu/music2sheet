// MOCK: replaced in PR-A1 with a real client generated from api/openapi.json.
// This module simulates the backend so the frontend has a working dev loop
// before Agent B's API lands.

import sampleMusicXml from '../fixtures/sample.musicxml?raw';
import type { JobStatus, TranscribeRequest, TranscribeResponse } from './types';

interface MockJob {
  id: string;
  pollsRemaining: number;
}

const jobs = new Map<string, MockJob>();

function makeId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `mock-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export async function postTranscribe(req: TranscribeRequest): Promise<TranscribeResponse> {
  if (req.source.kind === 'url' && !req.source.url.trim()) {
    throw new Error('URL is required');
  }
  if (req.source.kind === 'file' && !req.source.file) {
    throw new Error('File is required');
  }
  const id = makeId();
  jobs.set(id, { id, pollsRemaining: 3 });
  return { job_id: id };
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const job = jobs.get(jobId);
  if (!job) {
    return { job_id: jobId, status: 'error', progress: 0, error: 'Unknown job' };
  }
  if (job.pollsRemaining > 0) {
    job.pollsRemaining -= 1;
    const progress = Math.min(0.99, 0.25 * (3 - job.pollsRemaining));
    return { job_id: jobId, status: job.pollsRemaining === 2 ? 'pending' : 'running', progress };
  }
  return {
    job_id: jobId,
    status: 'done',
    progress: 1,
    musicxml: sampleMusicXml,
    result_urls: { musicxml: `mock://jobs/${jobId}/score.musicxml` },
  };
}
