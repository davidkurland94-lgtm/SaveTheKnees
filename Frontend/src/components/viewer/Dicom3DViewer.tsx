import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent, WheelEvent } from "react";

import type { ViewerStack } from "@/interfaces";
import { cn, useControlled } from "@/lib";
import SliceCanvas, { sliceSize } from "./SliceCanvas";
import { StackTabs, ViewerButton, ViewerPlaceholder, ViewerSlider } from "./ViewerChrome";

const MIN_ZOOM = 0.3;
const MAX_ZOOM = 5;

/** How deep the stack is drawn when the page passes no `spacingMm`. */
const DEFAULT_DEPTH_RATIO = 0.7;

/** Slice textures are capped here: 40 planes at a native 512² would cost ~40 MB. */
const SLICE_TEXTURE_SIZE = 256;

const DEFAULT_VIEW = { yaw: -32, pitch: -18, zoom: 1, panX: 0, panY: 0 };

type View = typeof DEFAULT_VIEW;

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

interface Dicom3DViewerProps {
  /**
   * The same stacks `Dicom2DViewer` takes — pass both the same array (and the
   * same `stackId`) to keep the two views on one series. Nothing is fetched
   * here: the study UID goes API → page → here as pixels.
   */
  stacks: ViewerStack[];
  /** Controlled active stack; omit and the viewer owns the selection. */
  stackId?: string;
  defaultStackId?: string;
  onStackChange?: (stackId: string) => void;
  emptyLabel?: string;
  className?: string;
}

/**
 * Stacks a series back into a volume: every slice is drawn as a translucent
 * plane at its own depth, and the whole stack orbits under the pointer.
 *
 * This is a slice-stack render, not an isosurface — it needs nothing beyond the
 * pixels the 2D viewer already has, so no extra request and no 3D dependency.
 * Drag to rotate, wheel to zoom, shift-drag to pan.
 */
export function Dicom3DViewer({
  stacks,
  stackId,
  defaultStackId,
  onStackChange,
  emptyLabel = "No slices to stack.",
  className,
}: Dicom3DViewerProps) {
  const [activeId, setActiveId] = useControlled(
    stackId,
    defaultStackId ?? stacks[0]?.id ?? "",
    onStackChange,
  );
  const active = stacks.find((stack) => stack.id === activeId) ?? stacks[0];
  const slices = active?.slices ?? [];
  const count = slices.length;

  const [view, setView] = useState<View>(DEFAULT_VIEW);
  const [opacity, setOpacity] = useState(0.45);
  const [depthScale, setDepthScale] = useState(1);
  const [sliceBudget, setSliceBudget] = useState(32);
  const [brightness, setBrightness] = useState(110);
  const [contrast, setContrast] = useState(130);

  // The planes are sized off the stage, so the stack fills whatever box the
  // page gives the component.
  const stageRef = useRef<HTMLDivElement>(null);
  const [stage, setStage] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const element = stageRef.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) => {
      const box = entries[0]?.contentRect;
      if (box) setStage({ width: box.width, height: box.height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const budget = Math.min(sliceBudget, count);

  /** Evenly spaced picks through the stack, each with its −0.5…0.5 depth. */
  const planes = useMemo(() => {
    if (count === 0) return [];
    const total = Math.max(1, Math.min(budget, count));
    return Array.from({ length: total }, (_, position) => {
      const fraction = total > 1 ? position / (total - 1) : 0.5;
      const source = Math.round(fraction * (count - 1));
      return {
        slice: slices[source],
        key: slices[source].id ?? `slice-${source}`,
        depth: fraction - 0.5,
      };
    });
  }, [slices, count, budget]);

  const imageSize = slices[0] ? sliceSize(slices[0].image) : null;
  const base = Math.max(140, Math.min(stage.width, stage.height) * 0.62);
  const planeWidth = base;
  const planeHeight =
    imageSize && imageSize.width > 0 ? base * (imageSize.height / imageSize.width) : base;

  // Physical depth over physical width, so a 4 mm-slice stack of 0.3 mm pixels
  // comes out as deep as it really is rather than as a cube.
  const spacing = active?.spacingMm;
  const ratio =
    spacing && imageSize && imageSize.width > 0 && spacing[2] > 0
      ? (count * spacing[0]) / (imageSize.width * spacing[2])
      : DEFAULT_DEPTH_RATIO;
  const depth = planeWidth * clamp(ratio, 0.05, 4) * depthScale;

  const dragRef = useRef<{ x: number; y: number; pan: boolean } | null>(null);

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      x: event.clientX,
      y: event.clientY,
      pan: event.shiftKey || event.button === 1,
    };
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    dragRef.current = { ...drag, x: event.clientX, y: event.clientY };
    setView((previous) =>
      drag.pan
        ? { ...previous, panX: previous.panX + dx, panY: previous.panY + dy }
        : {
            ...previous,
            yaw: previous.yaw + dx * 0.4,
            // Screen Y points down, so dragging down tilts the top towards us.
            pitch: clamp(previous.pitch - dy * 0.4, -89, 89),
          },
    );
  };

  const onPointerUp = (event: PointerEvent<HTMLDivElement>) => {
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const onWheel = (event: WheelEvent<HTMLDivElement>) => {
    setView((previous) => ({
      ...previous,
      zoom: clamp(previous.zoom * Math.exp(-event.deltaY * 0.0015), MIN_ZOOM, MAX_ZOOM),
    }));
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const nudge = (patch: Partial<View>) => setView((previous) => ({ ...previous, ...patch }));
    if (event.key === "ArrowLeft") nudge({ yaw: view.yaw - 5 });
    else if (event.key === "ArrowRight") nudge({ yaw: view.yaw + 5 });
    else if (event.key === "ArrowUp") nudge({ pitch: clamp(view.pitch + 5, -89, 89) });
    else if (event.key === "ArrowDown") nudge({ pitch: clamp(view.pitch - 5, -89, 89) });
    else if (event.key === "+" || event.key === "=") {
      nudge({ zoom: clamp(view.zoom * 1.15, MIN_ZOOM, MAX_ZOOM) });
    } else if (event.key === "-") nudge({ zoom: clamp(view.zoom / 1.15, MIN_ZOOM, MAX_ZOOM) });
    else if (event.key === "0") setView(DEFAULT_VIEW);
    else return;
    event.preventDefault();
  };

  return (
    <section
      className={cn(
        "flex min-h-0 flex-col overflow-hidden rounded-2xl border border-border bg-viewer",
        className,
      )}
    >
      <StackTabs
        stacks={stacks}
        activeId={active?.id ?? ""}
        onSelect={setActiveId}
        detail={active?.label}
      />

      <div
        ref={stageRef}
        role="group"
        tabIndex={0}
        aria-label="3D slice stack. Drag to rotate, wheel to zoom, shift-drag to pan."
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onWheel={onWheel}
        onKeyDown={onKeyDown}
        className="relative min-h-72 flex-1 touch-none select-none overflow-hidden bg-viewer outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-accent"
        style={{ perspective: "1400px", cursor: count > 0 ? "grab" : "default" }}
      >
        {count === 0 ? (
          <div className="flex h-full items-center justify-center">
            <ViewerPlaceholder label={emptyLabel} />
          </div>
        ) : (
          <>
            <div
              className="absolute left-1/2 top-1/2"
              style={{
                transformStyle: "preserve-3d",
                transform: [
                  `translate(${view.panX}px, ${view.panY}px)`,
                  `rotateX(${view.pitch}deg)`,
                  `rotateY(${view.yaw}deg)`,
                  // scale3d, not scale: a 2D scale would stretch the planes
                  // without deepening the stack with them.
                  `scale3d(${view.zoom}, ${view.zoom}, ${view.zoom})`,
                ].join(" "),
              }}
            >
              {planes.map(({ slice, key, depth: position }) => (
                <SliceCanvas
                  key={key}
                  image={slice.image}
                  brightness={brightness}
                  contrast={contrast}
                  maxSize={SLICE_TEXTURE_SIZE}
                  className="absolute left-0 top-0"
                  style={{
                    width: `${planeWidth}px`,
                    height: `${planeHeight}px`,
                    opacity,
                    transform: `translate(-50%, -50%) translateZ(${position * depth}px)`,
                  }}
                />
              ))}
              {/* Front and back faces, so the volume keeps its edges once the
                  slices themselves are turned translucent. */}
              {[-0.5, 0.5].map((position) => (
                <div
                  key={position}
                  className="pointer-events-none absolute left-0 top-0 border border-accent/25"
                  style={{
                    width: `${planeWidth}px`,
                    height: `${planeHeight}px`,
                    transform: `translate(-50%, -50%) translateZ(${position * depth}px)`,
                  }}
                />
              ))}
            </div>

            <div className="pointer-events-none absolute left-3 top-3 flex flex-col items-start gap-1">
              {active?.plane && (
                <span className="rounded bg-black/60 px-2 py-0.5 font-mono text-[11px] text-accent">
                  {active.plane}
                </span>
              )}
              <span className="rounded bg-black/60 px-2 py-0.5 font-mono text-[11px] text-white/50">
                {planes.length} / {count} slices
              </span>
            </div>
            <div className="pointer-events-none absolute right-3 top-3">
              <span className="rounded bg-black/60 px-2 py-0.5 font-mono text-[11px] text-white/40">
                {Math.round(view.yaw)}° · {Math.round(view.pitch)}° · ×{view.zoom.toFixed(1)}
              </span>
            </div>
            <p className="pointer-events-none absolute bottom-3 left-1/2 -translate-x-1/2 text-[11px] text-white/25">
              drag to rotate · wheel to zoom · shift-drag to pan
            </p>
          </>
        )}
      </div>

      <div className="flex items-center gap-4 border-t border-white/5 bg-viewer-panel px-3 py-2">
        <ViewerSlider
          icon="layers"
          label="Slices"
          value={budget}
          min={Math.min(4, count)}
          max={Math.max(count, 4)}
          disabled={count < 5}
          onChange={setSliceBudget}
        />
        <ViewerSlider
          icon="cube"
          label="Depth"
          value={depthScale}
          min={0.25}
          max={3}
          step={0.05}
          format={(value) => `×${value.toFixed(2)}`}
          onChange={setDepthScale}
        />
        <ViewerSlider
          icon="contrast"
          label="Opacity"
          value={opacity}
          min={0.05}
          max={1}
          step={0.01}
          format={(value) => `${Math.round(value * 100)}%`}
          onChange={setOpacity}
        />
      </div>

      <div className="flex items-center gap-4 border-t border-white/5 bg-viewer-panel px-3 py-2">
        <ViewerSlider
          icon="brightness"
          label="Brightness"
          value={brightness}
          min={50}
          max={250}
          onChange={setBrightness}
        />
        <ViewerSlider
          icon="contrast"
          label="Contrast"
          value={contrast}
          min={50}
          max={300}
          onChange={setContrast}
        />
        <ViewerButton
          icon="reset"
          label="Reset view"
          onClick={() => {
            setView(DEFAULT_VIEW);
            setOpacity(0.45);
            setDepthScale(1);
            setBrightness(110);
            setContrast(130);
          }}
        />
      </div>
    </section>
  );
}

export default Dicom3DViewer;
