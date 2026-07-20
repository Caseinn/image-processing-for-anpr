import ZoomableImage from "@/components/zoomable-image";

interface Props {
  /** Map of pipeline step keys to base64 data URIs. */
  steps: Record<string, string>;
}

/** Ordered list of pipeline steps to display. */
const PIPELINE_ORDER = [
  { key: "grayscale", label: "Grayscale", note: "3ch → 1ch" },
  { key: "clahe", label: "CLAHE", note: "contrast adaptive" },
  { key: "blur", label: "Gaussian Blur", note: "denoise" },
  { key: "edges", label: "Canny Edges", note: "binary edge map" },
  { key: "contours", label: "All Contours", note: "findContours" },
  { key: "area_filter", label: "Area + 4 Corners", note: "quad filter" },
  { key: "aspect_filter", label: "Aspect Filtered", note: "2:1–8:1" },
];

/** Displays a responsive grid of pipeline step visualisation cards. */
export default function PipelineSteps({ steps }: Props) {
  const items = PIPELINE_ORDER.map(({ key, label, note }, idx) => {
    const dataUri = steps[key];
    if (!dataUri) return null;
    return { key, label, note, dataUri, idx: idx + 1 };
  }).filter(Boolean);

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
      {items.map((item) => (
        <div key={item!.key} className="border border-rule rounded-md overflow-hidden bg-paper flex flex-col">
          <div className="relative aspect-[16/9] overflow-hidden bg-paper-2">
            <ZoomableImage src={item!.dataUri} alt={item!.label} className="w-full h-full object-cover block cursor-zoom-in" />
          </div>
          <div className="flex items-baseline gap-2 px-[10px] py-2 border-t border-rule">
            <span className="font-mono text-[0.625rem] font-medium text-accent leading-none">
              {String(item!.idx).padStart(2, "0")}
            </span>
            <div className="min-w-0">
              <p className="text-xs font-medium text-ink m-0 leading-tight">{item!.label}</p>
              <p className="text-[0.625rem] font-mono text-muted m-0 leading-tight">{item!.note}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
