import { useEffect, useRef, useState } from 'react';
import { OpenSheetMusicDisplay } from 'opensheetmusicdisplay';

interface Props {
  musicxml: string;
}

export function MusicXmlViewer({ musicxml }: Props): JSX.Element {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const osmdRef = useRef<OpenSheetMusicDisplay | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let cancelled = false;
    setError(null);

    const osmd =
      osmdRef.current ??
      new OpenSheetMusicDisplay(container, {
        autoResize: true,
        drawTitle: true,
        drawSubtitle: false,
        backend: 'svg',
      });
    osmdRef.current = osmd;

    osmd
      .load(musicxml)
      .then(() => {
        if (cancelled) return;
        osmd.render();
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Failed to render score');
      });

    return () => {
      cancelled = true;
    };
  }, [musicxml]);

  return (
    <div className="musicxml-viewer">
      {error && <div role="alert" className="musicxml-viewer__error">{error}</div>}
      <div ref={containerRef} data-testid="osmd-container" />
    </div>
  );
}
