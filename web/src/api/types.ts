import type { components } from './generated';

type Schemas = components['schemas'];

export type TranscribeRequest = Schemas['TranscribeRequest'];
export type TranscribeResponse = Schemas['TranscribeResponse'];
export type JobStatus = Schemas['JobStatusResponse'];
export type JobState = Schemas['JobStatus'];
export type JobResultUrls = Schemas['JobResultUrls'];
export type HTTPValidationError = Schemas['HTTPValidationError'];

export const TERMINAL_STATES: ReadonlyArray<JobState> = ['done', 'failed'];

export function isTerminal(state: JobState): boolean {
  return TERMINAL_STATES.includes(state);
}
