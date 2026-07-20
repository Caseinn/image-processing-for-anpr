/** A single detected bounding box. */
interface Box {
  id: number;
  x: number;
  y: number;
  w: number;
  h: number;
  aspect: number;
}

interface Props {
  boxes: Box[];
}

const COLS = ["ID", "X", "Y", "W", "H", "Aspect"];

/** Tabular display of detected bounding box coordinates and dimensions. */
export default function BBoxTable({ boxes }: Props) {
  if (boxes.length === 0) return null;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="bg-paper-2">
            {COLS.map((h) => (
              <th
                key={h}
                className="text-left px-3 py-2 font-medium text-muted border-b border-rule font-mono text-[0.6875rem] uppercase tracking-wider"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {boxes.map((b, i) => (
            <tr
              key={b.id}
              className={`transition-colors duration-100 ${i % 2 === 0 ? "bg-transparent" : "bg-paper-2"} hover:bg-accent-bg`}
            >
              {[b.id, b.x, b.y, b.w, b.h, b.aspect].map((v, ci) => (
                <td
                  key={ci}
                  className={`px-3 py-2 border-b border-rule text-ink text-sm ${ci > 0 ? "font-mono tabular-nums" : ""}`}
                >
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
