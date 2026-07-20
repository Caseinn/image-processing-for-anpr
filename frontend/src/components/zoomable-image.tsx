import { useState, useRef, useCallback, useEffect } from "react";

interface Props {
  src: string;
  alt: string;
  className?: string;
}

/** Shared styles for the zoom overlay control buttons. */
const btnClass =
  "w-[34px] h-[34px] flex items-center justify-center text-white border border-solid rounded-sm cursor-pointer text-base font-mono transition-colors duration-150 bg-[oklch(100%_0_0_/_0.15)] border-[oklch(100%_0_0_/_0.25)] hover:bg-[oklch(100%_0_0_/_0.25)]";

/**
 * Click-to-zoom image overlay with pan and keyboard controls.
 *
 * Features:
 * - Scroll wheel or +/- buttons to zoom (0.25×–10×)
 * - Click-and-drag to pan while zoomed
 * - 0 key to reset, Esc to close
 */
export default function ZoomableImage({ src, alt, className }: Props) {
  const [open, setOpen] = useState(false);
  const [scale, setScale] = useState(1);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef({ x: 0, y: 0 });
  const didDrag = useRef(false);
  const imgRef = useRef<HTMLImageElement>(null);

  /** Reset zoom and position to default. */
  const reset = useCallback(() => {
    setScale(1);
    setPos({ x: 0, y: 0 });
  }, []);

  const zoomIn = useCallback(() => setScale((s) => Math.min(s * 1.5, 10)), []);
  const zoomOut = useCallback(() => setScale((s) => Math.max(s / 1.5, 0.25)), []);

  /** Auto-reset on overlay open/close. */
  useEffect(() => {
    if (!open) reset();
  }, [open, reset]);

  /** Global keyboard shortcuts for the overlay. */
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
      if (e.key === "+" || e.key === "=") zoomIn();
      if (e.key === "-") zoomOut();
      if (e.key === "0") reset();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, zoomIn, zoomOut, reset]);

  return (
    <>
      <img
        ref={imgRef}
        src={src}
        alt={alt}
        className={className}
        onClick={() => setOpen(true)}
      />

      {open && (
        <div
          className="fixed inset-0 z-[9999] flex items-center justify-center select-none"
          style={{
            background: "oklch(0% 0 0 / 0.85)",
            cursor: dragging ? "grabbing" : "grab",
          }}
          onClick={(e) => { if (e.target === e.currentTarget && !didDrag.current) setOpen(false); didDrag.current = false; }}
          onWheel={(e) => {
            e.preventDefault();
            setScale((s) => {
              const next = e.deltaY < 0 ? s * 1.15 : s / 1.15;
              return Math.max(0.25, Math.min(next, 10));
            });
          }}
          onMouseDown={(e) => {
            if (e.button !== 0) return;
            didDrag.current = false;
            setDragging(true);
            dragStart.current = { x: e.clientX - pos.x, y: e.clientY - pos.y };
          }}
          onMouseMove={(e) => {
            if (!dragging) return;
            didDrag.current = true;
            setPos({ x: e.clientX - dragStart.current.x, y: e.clientY - dragStart.current.y });
          }}
          onMouseUp={() => setDragging(false)}
          onMouseLeave={() => setDragging(false)}
        >
          <div
            className="absolute top-4 right-4 flex gap-2 z-10"
            onClick={(e) => e.stopPropagation()}
          >
            <button onClick={zoomOut} className={btnClass} title="Zoom out">−</button>
            <span className="font-mono text-xs text-white self-center min-w-[36px] text-center">
              {Math.round(scale * 100)}%
            </span>
            <button onClick={zoomIn} className={btnClass} title="Zoom in">+</button>
            <button onClick={reset} className={btnClass} title="Reset">⊖</button>
            <button onClick={() => setOpen(false)} className={`${btnClass} ml-2`} title="Close">✕</button>
          </div>

          <img
            src={src}
            alt={alt}
            draggable={false}
            className="max-w-[90vw] max-h-[90vh] object-contain pointer-events-none"
            style={{
              transform: `translate(${pos.x}px, ${pos.y}px) scale(${scale})`,
              transition: dragging ? "none" : "transform 0.15s",
            }}
          />
        </div>
      )}
    </>
  );
}
