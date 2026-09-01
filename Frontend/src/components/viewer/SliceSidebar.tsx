import type { ParsedScan } from "@/interfaces";
import { cn } from "@/lib";
import DicomCanvas from "./DicomCanvas";

interface SliceSidebarProps {
  scans: ParsedScan[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

/** Thumbnail rail of every decoded slice in the uploaded series. */
export function SliceSidebar({ scans, selectedId, onSelect }: SliceSidebarProps) {
  return (
    <aside className="flex w-44 shrink-0 flex-col border-r border-border bg-card">
      <div className="px-4 pb-2 pt-4">
        <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Slices · {scans.length}
        </p>
      </div>
      <div className="flex flex-col gap-1 overflow-y-auto px-2 pb-4">
        {scans.map((scan) => {
          const selected = scan.id === selectedId;
          return (
            <button
              key={scan.id}
              type="button"
              onClick={() => onSelect(scan.id)}
              aria-current={selected}
              className={cn(
                "w-full overflow-hidden rounded-lg border text-left transition-all duration-150",
                selected
                  ? "border-accent shadow-md shadow-purple-100"
                  : "border-transparent hover:border-border",
              )}
            >
              <div className="relative bg-black" style={{ aspectRatio: `${scan.cols}/${scan.rows}` }}>
                <DicomCanvas imageData={scan.imageData} className="h-full w-full object-contain" />
                {selected && <div className="absolute inset-0 ring-2 ring-inset ring-accent" />}
              </div>
              <div className="bg-white px-2 py-1.5">
                <p className="truncate text-xs font-semibold text-foreground">
                  {scan.label || scan.modality}
                </p>
                <p className="truncate text-[10px] text-muted-foreground">{scan.slice}</p>
              </div>
            </button>
          );
        })}
      </div>
    </aside>
  );
}

export default SliceSidebar;
