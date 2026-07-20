import { useCallback } from "react";

/** Sample test images shipped with the frontend. */
const SAMPLES = [
  { src: "/samples/test001.jpg", label: "test001" },
  { src: "/samples/test010.jpg", label: "test010" },
  { src: "/samples/test020.jpg", label: "test020" },
];

interface Props {
  /** Called when the user clicks a sample image. */
  onDetect: (file: File) => void;
  /** Disables interaction while detection is running. */
  loading: boolean;
}

/** Clickable sample image thumbnails for quick testing. */
export default function SamplePhotos({ onDetect, loading }: Props) {
  /** Fetch the sample image, wrap it as a File, and fire detection. */
  const handleClick = useCallback(
    async (src: string, name: string) => {
      if (loading) return;
      try {
        const res = await fetch(src);
        const blob = await res.blob();
        const file = new File([blob], `${name}.jpg`, { type: "image/jpeg" });
        onDetect(file);
      } catch {
        // silently fail
      }
    },
    [loading, onDetect],
  );

  return (
    <div className="flex flex-col gap-3 items-center">
      <span className="font-mono text-[0.625rem] text-neutral uppercase tracking-wider text-center">
        Samples
      </span>
      {SAMPLES.map((s) => (
        <button
          key={s.src}
          onClick={() => handleClick(s.src, s.label)}
          disabled={loading}
          className="group relative w-24 h-20 rounded-md overflow-hidden border border-rule bg-paper-2 cursor-pointer p-0 transition-all duration-150 hover:border-accent disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <img
            src={s.src}
            alt={s.label}
            className="w-full h-full object-cover transition-transform duration-150 group-hover:scale-105"
          />
          <span className="absolute bottom-0 left-0 right-0 text-[0.5625rem] font-mono text-white bg-[oklch(0%_0_0_/_0.55)] px-1 py-0.5 text-center truncate">
            {s.label}
          </span>
        </button>
      ))}
    </div>
  );
}
