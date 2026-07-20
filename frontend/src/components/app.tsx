import { useState, useCallback } from "react";
import ImageUpload from "@/components/image-upload";
import SamplePhotos from "@/components/sample-photos";
import PipelineSteps from "@/components/pipeline-steps";
import BBoxTable from "@/components/bbox-table";
import ZoomableImage from "@/components/zoomable-image";
import ScrollToTop from "@/components/scroll-to-top";

/** Response shape from the backend detection API. */
interface DetectResult {
  annotated: string;
  crops: string[];
  boxes: { id: number; x: number; y: number; w: number; h: number; aspect: number }[];
  pipeline: Record<string, string>;
}

const STAGES = [
  { num: "1.0", label: "INPUT", desc: "Select a vehicle photo to run plate detection", key: "upload" },
  { num: "2.0", label: "PROCESS", desc: "Image flows through the preprocessing pipeline", key: "pipeline" },
  { num: "3.0", label: "DETECT", desc: "Candidate plates are filtered and localised", key: "detect" },
  { num: "4.0", label: "EXTRACT", desc: "Confirmed plate regions are cropped for OCR", key: "extract" },
];

/** Root application component orchestrating the full ANPR workflow. */
export default function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DetectResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  /** Send an image file to the backend detection endpoint. */
  const handleDetect = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);

    const form = new FormData();
    form.append("image", file);

    try {
      const res = await fetch("http://localhost:8000/api/detect", {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const msg = await res.json();
        throw new Error(msg.error ?? "Detection failed");
      }
      const data: DetectResult = await res.json();
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  const stageActive = (idx: number) => {
    if (idx === 0) return true;
    if (idx === 1) return loading || result !== null;
    if (idx === 2 || idx === 3) return result !== null;
    return false;
  };

  return (
    <div className="min-h-screen flex flex-col">
      <header className="flex items-center justify-between w-full max-w-7xl mx-auto px-8 py-5">
        <a className="font-display text-lg font-semibold text-ink no-underline tracking-tight">
          ANPR
        </a>
        <a
          href="https://github.com/Caseinn/image-processing-for-anpr"
          target="_blank"
          rel="noopener noreferrer"
          className="font-mono text-xs text-muted no-underline px-3.5 py-1.5 border border-rule rounded-md transition-colors duration-150 hover:border-accent hover:text-accent"
        >
          GitHub →
        </a>
      </header>

      <main className={`max-w-7xl mx-auto px-8 pb-16 w-full ${!result && !loading ? "flex flex-col justify-center" : ""}`}>
        <StageSection num={STAGES[0].num} label={STAGES[0].label} desc={STAGES[0].desc} active={stageActive(0)}>
          <div className="flex gap-6 items-center">
            <div className="flex-1 min-w-0">
              <ImageUpload onDetect={handleDetect} loading={loading} />
            </div>
            <div className="flex-shrink-0 hidden sm:block">
              <SamplePhotos onDetect={handleDetect} loading={loading} />
            </div>
          </div>
          {error && (
            <div className="mt-4 px-4 py-3 text-sm rounded-md" style={{ background: "oklch(95% 0.02 30)", border: "1px solid oklch(85% 0.04 30)", color: "oklch(35% 0.06 30)" }}>
              {error}
            </div>
          )}
        </StageSection>

        {stageActive(1) && (
          <>
            <StageDivider />
            <StageSection num={STAGES[1].num} label={STAGES[1].label} desc={STAGES[1].desc} active={stageActive(1)}>
              {loading ? (
                <div className="flex items-center gap-3 py-8">
                  <div className="w-5 h-5 border-2 border-rule border-t-accent rounded-full animate-spin" />
                  <span className="text-muted text-sm">Running pipeline…</span>
                </div>
              ) : result ? (
                <PipelineSteps steps={result.pipeline} />
              ) : null}
            </StageSection>
          </>
        )}

        {result && (
          <>
            <StageDivider />
            <StageSection num={STAGES[2].num} label={STAGES[2].label} desc={STAGES[2].desc} active={true}>
              <div className="flex flex-col gap-6">
                <div className="border border-rule rounded-lg overflow-hidden bg-paper-2">
                  <ZoomableImage src={result.annotated} alt="Annotated detections" className="w-full block max-h-[480px] object-contain cursor-zoom-in" />
                </div>
                {result.boxes.length > 0 && (
                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <span className="text-xs font-mono text-muted uppercase tracking-wider">
                        Bounding Boxes
                      </span>
                      <span className="text-[0.6875rem] font-mono text-neutral px-[7px] py-[1px] border border-rule rounded-sm">
                        {result.boxes.length} found
                      </span>
                    </div>
                    <div className="border border-rule rounded-md overflow-hidden">
                      <BBoxTable boxes={result.boxes} />
                    </div>
                  </div>
                )}
              </div>
            </StageSection>

            <StageDivider />
            <StageSection num={STAGES[3].num} label={STAGES[3].label} desc={STAGES[3].desc} active={result.crops.length > 0}>
              {result.crops.length > 0 ? (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                  {result.crops.map((src, i) => (
                    <div key={i} className="border border-rule rounded-md overflow-hidden bg-paper">
                      <ZoomableImage src={src} alt={`Plate ${i + 1}`} className="w-full block cursor-zoom-in" />
                      <p className="text-xs font-mono text-center py-[6px] text-neutral border-t border-rule">
                        Plate {i + 1}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-muted text-sm">No plates detected in this image.</p>
              )}
            </StageSection>
          </>
        )}
      </main>

      <footer className="mt-auto border-t border-rule px-8 py-5 max-w-7xl mx-auto w-full">
        <p className="text-sm text-muted font-mono">
          ANPR Pipeline · built with OpenCV
        </p>
      </footer>

      <ScrollToTop />
    </div>
  );
}

/** A named step in the 4-stage narrative workflow. */
function StageSection({
  num,
  label,
  desc,
  active,
  children,
}: {
  num: string;
  label: string;
  desc: string;
  active: boolean;
  children: React.ReactNode;
}) {
  return (
    <section className={`transition-opacity duration-300 ${active ? "opacity-100" : "opacity-35"}`}>
      <div className="flex items-baseline gap-3 mb-2">
        <span className={`font-mono text-xl font-semibold tracking-wide transition-colors duration-300 ${active ? "text-accent" : "text-neutral"}`}>
          {num}
        </span>
        <h2 className={`font-display text-3xl font-semibold tracking-tight m-0 transition-colors duration-300 ${active ? "text-ink" : "text-neutral"}`}>
          {label}
        </h2>
      </div>
      <p className="text-base text-muted mb-6 ml-[calc(1.25rem+var(--space-3))]">
        {desc}
      </p>
      {children}
    </section>
  );
}

/** Horizontal divider used between workflow stages. */
function StageDivider() {
  return (
    <div className="my-12 flex items-center gap-3">
      <div className="flex-1 h-[2px] bg-rule rounded-[1px]" />
      <span className="text-[0.625rem] font-mono text-neutral tracking-wider">●</span>
      <div className="flex-1 h-[2px] bg-rule rounded-[1px]" />
    </div>
  );
}
