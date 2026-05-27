import { useEffect, useRef, useState } from 'react';
import { OpenSheetMusicDisplay } from 'opensheetmusicdisplay';
import { fetchMusicXml } from '../api/client';

type Props =
  | { musicxml: string; musicxmlUrl?: never }
  | { musicxmlUrl: string; musicxml?: never };

export function MusicXmlViewer(props: Props): JSX.Element {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const osmdRef = useRef<OpenSheetMusicDisplay | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [xml, setXml] = useState<string | null>(props.musicxml ?? null);

  useEffect(() => {
    if (props.musicxml !== undefined) {
      setXml(props.musicxml);
      return;
    }
    const controller = new AbortController();
    setError(null);
    setXml(null);
    fetchMusicXml(props.musicxmlUrl, controller.signal)
      .then((text) => setXml(text))
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setError(err instanceof Error ? err.message : 'Failed to fetch MusicXML');
      });
    return () => controller.abort();
  }, [props.musicxml, props.musicxmlUrl]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || xml === null) return;

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
      .load(xml)
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
  }, [xml]);

  return (
    <div className="musicxml-viewer">
      {error && <div role="alert" className="musicxml-viewer__error">{error}</div>}
      <div ref={containerRef} data-testid="osmd-container" />
    </div>
  );
}
