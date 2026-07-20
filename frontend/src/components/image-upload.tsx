import { useState, useCallback, useRef, type DragEvent } from "react";

interface Props {
  /** Called when the user selects or drops an image file. */
  onDetect: (file: File) => void;
  /** Disables interaction while detection is running. */
  loading: boolean;
}

/** Drag-and-drop image upload zone with file picker fallback. */
export default function ImageUpload({ onDetect, loading }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  /** Validate and forward the selected image file. */
  const handleFile = useCallback(
    (file: File) => {
      if (file && file.type.startsWith("image/")) {
        setFileName(file.name);
        onDetect(file);
      }
    },
    [onDetect],
  );

  /** Handle a dropped file from drag-and-drop. */
  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
      onClick={() => !loading && inputRef.current?.click()}
      className="rounded-lg px-8 py-24 text-center transition-all duration-200"
      style={{
        border: `2px dashed ${dragOver ? "var(--color-accent)" : "var(--color-rule)"}`,
        cursor: loading ? "not-allowed" : "pointer",
        opacity: loading ? 0.5 : 1,
        background: dragOver ? "var(--color-accent-bg)" : "var(--color-paper)",
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />

      {loading ? (
        <div className="flex flex-col items-center justify-center gap-3 min-h-[156px]">
          <div className="w-6 h-6 border-2 border-rule border-t-accent rounded-full animate-spin" />
          <div>
            <p className="text-[0.9375rem] font-medium text-ink">{fileName ? <>Processing &ldquo;{fileName}&rdquo;&hellip;</> : "Processing image\u2026"}</p>
            <p className="text-sm text-muted mt-0.5">Running detection pipeline&hellip;</p>
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center min-h-[156px]">
          <svg width={36} height={36} fill="none" stroke="var(--color-neutral)" viewBox="0 0 24 24" className="mx-auto mb-4">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <p className="text-[0.9375rem] font-medium text-ink mb-0.5">Drop a vehicle image here</p>
          <p className="text-sm text-muted mb-4">or click to browse &middot; JPG, PNG</p>
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
            className="px-6 py-[9px] rounded-lg text-sm font-medium text-white border-0 cursor-pointer transition-all duration-150 active:scale-[0.97] bg-accent hover:bg-accent-2"
          >
            Choose File
          </button>
        </div>
      )}
    </div>
  );
}
