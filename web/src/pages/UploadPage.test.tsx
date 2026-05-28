import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { submitTranscribeMock } = vi.hoisted(() => ({
  submitTranscribeMock: vi.fn(),
}));

vi.mock('../api/client', async () => {
  const actual =
    await vi.importActual<typeof import('../api/client')>('../api/client');
  return {
    ...actual,
    submitTranscribe: submitTranscribeMock,
  };
});

import { UploadPage } from './UploadPage';
import { clearRecentJobs, STORAGE_KEY } from '../lib/recentJobs';

function renderPage(): void {
  render(
    <MemoryRouter>
      <UploadPage />
    </MemoryRouter>,
  );
}

function installLocalStorage(): Storage {
  const map = new Map<string, string>();
  const store: Storage = {
    getItem: (k) => (map.has(k) ? (map.get(k) as string) : null),
    setItem: (k, v) => {
      map.set(k, String(v));
    },
    removeItem: (k) => {
      map.delete(k);
    },
    clear: () => map.clear(),
    key: (i) => Array.from(map.keys())[i] ?? null,
    get length() {
      return map.size;
    },
  } as Storage;
  Object.defineProperty(window, 'localStorage', {
    value: store,
    configurable: true,
  });
  return store;
}

describe('UploadPage (PR-A3)', () => {
  beforeEach(() => {
    submitTranscribeMock.mockReset();
    installLocalStorage();
    clearRecentJobs();
    window.localStorage.removeItem(STORAGE_KEY);
  });

  afterEach(() => {
    clearRecentJobs();
  });

  it('renders the drop zone in file mode by default', () => {
    renderPage();
    const dz = screen.getByRole('button', { name: /audio file drop zone/i });
    expect(dz).toBeInTheDocument();
    expect(dz).toHaveAttribute('tabindex', '0');
    expect(
      screen.getByText(/drop an audio file here/i),
    ).toBeInTheDocument();
  });

  it('keyboard Enter on the drop zone opens the file picker', () => {
    renderPage();
    const dz = screen.getByRole('button', { name: /audio file drop zone/i });
    const input = screen.getByTestId('drop-zone-input') as HTMLInputElement;
    const clickSpy = vi.spyOn(input, 'click').mockImplementation(() => {});
    dz.focus();
    fireEvent.keyDown(dz, { key: 'Enter' });
    expect(clickSpy).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(dz, { key: ' ' });
    expect(clickSpy).toHaveBeenCalledTimes(2);
  });

  it('dropping a file selects it and Transcribe calls submitTranscribe with that file', async () => {
    submitTranscribeMock.mockResolvedValueOnce({
      job_id: '11111111-1111-1111-1111-111111111111',
      created_at: '2026-05-27T00:00:00Z',
      status: 'pending',
    });

    renderPage();
    const dz = screen.getByRole('button', { name: /audio file drop zone/i });
    const file = new File([new Uint8Array([1, 2, 3])], 'dropped.mp3', {
      type: 'audio/mpeg',
    });

    fireEvent.drop(dz, {
      dataTransfer: { files: [file] },
    });

    // Metadata renders after the drop.
    expect(await screen.findByText('dropped.mp3')).toBeInTheDocument();
    expect(screen.getAllByText(/audio\/mpeg/).length).toBeGreaterThan(0);

    const submit = screen.getByRole('button', { name: /transcribe/i });
    fireEvent.click(submit);

    await waitFor(() => {
      expect(submitTranscribeMock).toHaveBeenCalledTimes(1);
    });
    const arg = submitTranscribeMock.mock.calls[0][0] as { file?: File };
    expect(arg.file).toBeInstanceOf(File);
    expect(arg.file?.name).toBe('dropped.mp3');
  });

  it('persists the new job to the recent-jobs sidebar after submit', async () => {
    submitTranscribeMock.mockResolvedValueOnce({
      job_id: '22222222-2222-2222-2222-222222222222',
      created_at: '2026-05-27T01:00:00Z',
      status: 'pending',
    });

    renderPage();
    const dz = screen.getByRole('button', { name: /audio file drop zone/i });
    const file = new File([new Uint8Array(1)], 'cool-track.wav', { type: 'audio/wav' });
    fireEvent.drop(dz, { dataTransfer: { files: [file] } });
    fireEvent.click(screen.getByRole('button', { name: /transcribe/i }));

    await waitFor(() => expect(submitTranscribeMock).toHaveBeenCalled());

    const raw = window.localStorage.getItem(STORAGE_KEY);
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw as string) as Array<{ id: string; inputLabel: string }>;
    expect(parsed[0].id).toBe('22222222-2222-2222-2222-222222222222');
    expect(parsed[0].inputLabel).toBe('cool-track.wav');
  });

  it('YouTube URL mode dispatches to submitTranscribe with url', async () => {
    submitTranscribeMock.mockResolvedValueOnce({
      job_id: '33333333-3333-3333-3333-333333333333',
      created_at: '2026-05-27T02:00:00Z',
      status: 'pending',
    });

    renderPage();
    // Switch to URL mode.
    fireEvent.click(screen.getByRole('tab', { name: /youtube url/i }));
    const input = screen.getByPlaceholderText(/youtube\.com/i);
    fireEvent.change(input, { target: { value: 'https://www.youtube.com/watch?v=abc' } });
    fireEvent.click(screen.getByRole('button', { name: /transcribe/i }));

    await waitFor(() => expect(submitTranscribeMock).toHaveBeenCalled());
    const arg = submitTranscribeMock.mock.calls[0][0] as { url?: string };
    expect(arg.url).toBe('https://www.youtube.com/watch?v=abc');
  });

  it('renders the empty state for recent jobs', () => {
    renderPage();
    expect(screen.getByText(/no jobs yet/i)).toBeInTheDocument();
  });

  it('Clear history removes all recent entries', async () => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify([
        { id: 'a', createdAt: '2026-01-01T00:00:00Z', inputLabel: 'a.mp3' },
        { id: 'b', createdAt: '2026-01-02T00:00:00Z', inputLabel: 'b.mp3' },
      ]),
    );
    renderPage();
    expect(await screen.findByText('a.mp3')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /clear recent jobs history/i }));
    expect(screen.queryByText('a.mp3')).not.toBeInTheDocument();
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull();
  });
});
