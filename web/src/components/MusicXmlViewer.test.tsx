import { render, waitFor, cleanup } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import sampleMusicXml from '../fixtures/sample.musicxml?raw';

// Mock OSMD so the test runs fast and doesn't require a real layout engine.
// We verify the viewer wires load() + render() and produces DOM content.
vi.mock('opensheetmusicdisplay', () => {
  return {
    OpenSheetMusicDisplay: vi.fn().mockImplementation((container: HTMLElement) => {
      return {
        load: vi.fn().mockResolvedValue(undefined),
        render: vi.fn(() => {
          const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
          svg.setAttribute('data-testid', 'osmd-svg');
          container.appendChild(svg);
        }),
      };
    }),
  };
});

import { MusicXmlViewer } from './MusicXmlViewer';

describe('MusicXmlViewer', () => {
  afterEach(() => {
    cleanup();
  });

  it('renders the OSMD container and populates it after loading', async () => {
    const { getByTestId } = render(<MusicXmlViewer musicxml={sampleMusicXml} />);
    const container = getByTestId('osmd-container');
    expect(container).toBeInTheDocument();
    await waitFor(() => {
      expect(container.children.length).toBeGreaterThan(0);
    });
  });

  it('does not throw on the fixture MusicXML payload', () => {
    expect(() => render(<MusicXmlViewer musicxml={sampleMusicXml} />)).not.toThrow();
  });
});
