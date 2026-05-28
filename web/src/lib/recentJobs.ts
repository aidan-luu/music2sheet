/**
 * Persisted list of recently submitted transcription jobs.
 *
 * Stored in `localStorage` under `STORAGE_KEY`. Capped at MAX_RECENT entries
 * (newest first). Reads tolerate malformed / missing data by returning [].
 */

export const STORAGE_KEY = 'music2sheet.recentJobs';
export const MAX_RECENT = 10;

export interface RecentJob {
  /** Job id as returned by POST /transcribe(/upload). */
  id: string;
  /** ISO-8601 timestamp the job was created (client-side). */
  createdAt: string;
  /** Human-friendly label: original filename or pasted URL. */
  inputLabel: string;
}

function getStorage(): Storage | null {
  try {
    if (typeof window === 'undefined') return null;
    return window.localStorage;
  } catch {
    return null;
  }
}

function isRecentJob(value: unknown): value is RecentJob {
  if (!value || typeof value !== 'object') return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.id === 'string' &&
    typeof v.createdAt === 'string' &&
    typeof v.inputLabel === 'string'
  );
}

export function getRecentJobs(): RecentJob[] {
  const store = getStorage();
  if (!store) return [];
  const raw = store.getItem(STORAGE_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isRecentJob).slice(0, MAX_RECENT);
  } catch {
    return [];
  }
}

/**
 * Prepend `job` to the recent list, dedupe by id, and cap at MAX_RECENT.
 * Returns the new list.
 */
export function addRecentJob(job: RecentJob): RecentJob[] {
  const store = getStorage();
  const existing = getRecentJobs().filter((j) => j.id !== job.id);
  const next = [job, ...existing].slice(0, MAX_RECENT);
  if (store) {
    try {
      store.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      // quota / private mode — silently drop persistence
    }
  }
  return next;
}

export function clearRecentJobs(): void {
  const store = getStorage();
  if (!store) return;
  try {
    store.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
