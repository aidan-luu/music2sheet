import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  addRecentJob,
  clearRecentJobs,
  getRecentJobs,
  MAX_RECENT,
  STORAGE_KEY,
  type RecentJob,
} from './recentJobs';

/** Minimal in-memory localStorage shim. */
function makeStorage(): Storage {
  const map = new Map<string, string>();
  return {
    getItem: (k: string) => (map.has(k) ? (map.get(k) as string) : null),
    setItem: (k: string, v: string) => {
      map.set(k, String(v));
    },
    removeItem: (k: string) => {
      map.delete(k);
    },
    clear: () => map.clear(),
    key: (i: number) => Array.from(map.keys())[i] ?? null,
    get length() {
      return map.size;
    },
  } as Storage;
}

const sampleJob = (i: number): RecentJob => ({
  id: `job-${i}`,
  createdAt: `2026-01-${String(i).padStart(2, '0')}T00:00:00Z`,
  inputLabel: `song-${i}.mp3`,
});

describe('lib/recentJobs', () => {
  let storage: Storage;

  // Save and restore the original window.localStorage descriptor so this file
  // doesn't leak `Object.defineProperty` side effects into sibling test files.
  let originalDescriptor: PropertyDescriptor | undefined;

  beforeEach(() => {
    storage = makeStorage();
    if (typeof window !== 'undefined') {
      originalDescriptor = Object.getOwnPropertyDescriptor(window, 'localStorage');
      Object.defineProperty(window, 'localStorage', {
        value: storage,
        configurable: true,
        writable: true,
      });
    }
    vi.stubGlobal('localStorage', storage);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    if (typeof window !== 'undefined') {
      if (originalDescriptor) {
        Object.defineProperty(window, 'localStorage', originalDescriptor);
      } else {
        // jsdom default: localStorage is a non-own getter on window's prototype.
        // Removing our own property exposes the inherited one again.
        delete (window as unknown as Record<string, unknown>).localStorage;
      }
    }
  });

  it('returns [] when storage is empty', () => {
    expect(getRecentJobs()).toEqual([]);
  });

  it('returns [] on malformed JSON', () => {
    storage.setItem(STORAGE_KEY, '{not valid');
    expect(getRecentJobs()).toEqual([]);
  });

  it('returns [] when stored value is not an array', () => {
    storage.setItem(STORAGE_KEY, JSON.stringify({ foo: 'bar' }));
    expect(getRecentJobs()).toEqual([]);
  });

  it('drops malformed entries but keeps valid ones', () => {
    storage.setItem(
      STORAGE_KEY,
      JSON.stringify([
        sampleJob(1),
        { id: 7 }, // bad
        { id: 'ok', createdAt: 't', inputLabel: 'x' },
      ]),
    );
    const got = getRecentJobs();
    expect(got).toHaveLength(2);
    expect(got[0].id).toBe('job-1');
    expect(got[1].id).toBe('ok');
  });

  it('adds a job, putting it at the front', () => {
    addRecentJob(sampleJob(1));
    addRecentJob(sampleJob(2));
    const got = getRecentJobs();
    expect(got.map((j) => j.id)).toEqual(['job-2', 'job-1']);
  });

  it('dedupes by id, moving the re-added job to the front', () => {
    addRecentJob(sampleJob(1));
    addRecentJob(sampleJob(2));
    addRecentJob({ ...sampleJob(1), inputLabel: 'updated.mp3' });
    const got = getRecentJobs();
    expect(got.map((j) => j.id)).toEqual(['job-1', 'job-2']);
    expect(got[0].inputLabel).toBe('updated.mp3');
  });

  it(`caps at MAX_RECENT (${MAX_RECENT})`, () => {
    for (let i = 1; i <= MAX_RECENT + 5; i++) {
      addRecentJob(sampleJob(i));
    }
    const got = getRecentJobs();
    expect(got).toHaveLength(MAX_RECENT);
    // Newest first.
    expect(got[0].id).toBe(`job-${MAX_RECENT + 5}`);
  });

  it('persists to localStorage under STORAGE_KEY', () => {
    addRecentJob(sampleJob(42));
    const raw = storage.getItem(STORAGE_KEY);
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw as string) as RecentJob[];
    expect(parsed[0].id).toBe('job-42');
  });

  it('clearRecentJobs empties the list', () => {
    addRecentJob(sampleJob(1));
    addRecentJob(sampleJob(2));
    clearRecentJobs();
    expect(getRecentJobs()).toEqual([]);
    expect(storage.getItem(STORAGE_KEY)).toBeNull();
  });
});
