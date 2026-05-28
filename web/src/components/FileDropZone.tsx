import { useCallback, useRef, useState, type DragEvent, type KeyboardEvent } from 'react';

interface Props {
  file: File | null;
  onFile: (file: File | null) => void;
  accept?: string;
  /** ARIA label override for the drop zone. */
  label?: string;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/**
 * Accessible drag-and-drop file picker.
 *
 *  - Click / Enter / Space → opens the native file dialog.
 *  - Drop a file (or multiple — first is taken) → selects it.
 *  - When `file` is non-null, renders its metadata (name, size, MIME).
 *  - Focus ring via :focus-visible on the zone (see app.css).
 */
export function FileDropZone({
  file,
  onFile,
  accept = 'audio/mpeg,audio/wav,.mp3,.wav',
  label = 'Audio file drop zone',
}: Props): JSX.Element {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const openPicker = useCallback(() => {
    inputRef.current?.click();
  }, []);

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        openPicker();
      }
    },
    [openPicker],
  );

  const onDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragging(false);
      const dropped = e.dataTransfer?.files?.[0];
      if (dropped) onFile(dropped);
    },
    [onFile],
  );

  const onDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  const onDragLeave = useCallback(() => setDragging(false), []);

  return (
    <div className="drop-zone__wrap">
      <div
        role="button"
        tabIndex={0}
        aria-label={label}
        aria-describedby={file ? 'drop-zone-meta' : undefined}
        className={`drop-zone${dragging ? ' drop-zone--active' : ''}${
          file ? ' drop-zone--has-file' : ''
        }`}
        onClick={openPicker}
        onKeyDown={onKeyDown}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
      >
        {file ? (
          <div className="drop-zone__file">
            <strong className="drop-zone__file-name">{file.name}</strong>
            <span className="drop-zone__file-meta">
              {formatSize(file.size)} · {file.type || 'unknown type'}
            </span>
          </div>
        ) : (
          <div className="drop-zone__prompt">
            <strong>Drop an audio file here</strong>
            <span>or press Enter / click to browse (MP3 or WAV)</span>
          </div>
        )}
      </div>
      {file && (
        <p id="drop-zone-meta" className="drop-zone__sr-only">
          Selected file: {file.name}, {formatSize(file.size)}, {file.type || 'unknown type'}.
        </p>
      )}
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="drop-zone__input"
        data-testid="drop-zone-input"
        onChange={(e) => onFile(e.target.files?.[0] ?? null)}
      />
    </div>
  );
}
