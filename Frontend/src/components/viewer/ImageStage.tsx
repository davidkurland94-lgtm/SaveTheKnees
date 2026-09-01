import { useEffect } from "react";

import type { ParsedScan } from "@/interfaces";
import { Icon } from "@/components/ui";
import DicomCanvas from "./DicomCanvas";

interface ImageStageProps {
  scan: ParsedScan | undefined;
  brightness: number;
  contrast: number;
  onBrightness: (value: number) => void;
  onContrast: (value: number) => void;
  onReset: () => void;
  /** Arrow keys step through the stack. */
  onStep?: (delta: number) => void;
}

/** The dark viewport plus its window/level controls. */
export function ImageStage({
  scan,
  brightness,
  contrast,
  onBrightness,
  onContrast,
  onReset,
  onStep,
}: ImageStageProps) {
  useEffect(() => {
    if (!onStep) return;
    const handler = (event: KeyboardEvent) => {
      if (event.key === "ArrowDown" || event.key === "ArrowRight") onStep(1);
      else if (event.key === "ArrowUp" || event.key === "ArrowLeft") onStep(-1);
      else return;
      event.preventDefault();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onStep]);

  return (
    <main className="flex min-w-0 flex-1 flex-col bg-viewer">
      <div
        className="flex min-h-0 flex-1 items-start justify-center p-6"
        onWheel={(event) => onStep?.(event.deltaY > 0 ? 1 : -1)}
      >
        {scan ? (
          <div
            className="relative w-full max-w-2xl"
            style={{ aspectRatio: `${scan.cols}/${scan.rows}` }}
          >
            <DicomCanvas
              imageData={scan.imageData}
              brightness={brightness}
              contrast={contrast}
              className="h-full w-full rounded-sm"
            />
            <div className="absolute left-3 top-3 flex flex-col gap-1">
              <span className="rounded bg-black/60 px-2 py-0.5 font-mono text-xs text-accent">
                {scan.modality || "DICOM"}
              </span>
              <span className="rounded bg-black/60 px-2 py-0.5 font-mono text-xs text-white/60">
                {scan.cols} × {scan.rows}
              </span>
            </div>
            <div className="absolute right-3 top-3">
              <span className="rounded bg-black/60 px-2 py-0.5 font-mono text-xs text-white/40">
                {scan.slice}
              </span>
            </div>
            {/* Crosshair */}
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center opacity-20">
              <div className="h-px w-full bg-accent" />
            </div>
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center opacity-20">
              <div className="h-full w-px bg-accent" />
            </div>
          </div>
        ) : (
          <div className="flex h-full items-center text-sm text-white/40">
            No slices decoded for this series.
          </div>
        )}
      </div>

      <div className="flex items-center gap-8 border-t border-white/5 bg-viewer-panel px-6 py-3">
        <Slider
          icon="brightness"
          label="Brightness"
          value={brightness}
          min={50}
          max={200}
          onChange={onBrightness}
        />
        <Slider
          icon="contrast"
          label="Contrast"
          value={contrast}
          min={50}
          max={250}
          onChange={onContrast}
        />
        <button
          type="button"
          onClick={onReset}
          className="text-xs text-muted-foreground transition-colors hover:text-accent"
        >
          Reset
        </button>
      </div>
    </main>
  );
}

interface SliderProps {
  icon: "brightness" | "contrast";
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}

function Slider({ icon, label, value, min, max, onChange }: SliderProps) {
  return (
    <div className="flex flex-1 items-center gap-3">
      <Icon name={icon} size={14} className="text-accent" />
      <label className="w-20 text-xs text-white/40" htmlFor={`slider-${icon}`}>
        {label}
      </label>
      <input
        id={`slider-${icon}`}
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1 flex-1 accent-accent"
      />
      <span className="w-8 text-right font-mono text-xs text-white/30">{value}</span>
    </div>
  );
}

export default ImageStage;
