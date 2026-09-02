import { useCallback, useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";

import type { ViewerStack } from "@/interfaces";
import { cn, useControlled } from "@/lib";
import { Icon } from "@/components/ui";
import SliceCanvas, { sliceSize } from "./SliceCanvas";
import { StackTabs, ViewerButton, ViewerPlaceholder, ViewerSlider } from "./ViewerChrome";

/** Cine playback rate, in slices per second. */
const CINE_FPS = 12;

interface Dicom2DViewerProps {
  /**
   * One stack per axis, decoded by the page. This component never fetches: the
   * study UID goes API → page → here as pixels.
   */
  stacks: ViewerStack[];
  /** Controlled active stack; omit and the viewer owns the selection. */
  stackId?: string;
  defaultStackId?: string;
  onStackChange?: (stackId: string) => void;
  /** Controlled slice index within the active stack. */
  index?: number;
  onIndexChange?: (index: number) => void;
  emptyLabel?: string;
  className?: string;
}

/**
 * Plays a series one slice at a time: axis tabs across the top, a scrub slider
 * and a cine transport below, window/level at the bottom. The stage also takes
 * the wheel and the arrow keys.
 */
export function Dicom2DViewer({
  stacks,
  stackId,
  defaultStackId,
  onStackChange,
  index,
  onIndexChange,
  emptyLabel = "No slices to display.",
  className,
}: Dicom2DViewerProps) {
  const [activeId, setActiveId] = useControlled(
    stackId,
    defaultStackId ?? stacks[0]?.id ?? "",
    onStackChange,
  );
  // Falls back to the first stack so a late-arriving `stacks` prop, or a stale
  // selection after the page swaps studies, still renders something.
  const active = stacks.find((stack) => stack.id === activeId) ?? stacks[0];
  const slices = active?.slices ?? [];
  const count = slices.length;

  const [rawIndex, setIndex] = useControlled(index, 0, onIndexChange);
  const current = count === 0 ? 0 : Math.min(Math.max(rawIndex, 0), count - 1);
  const slice = slices[current];

  const [playing, setPlaying] = useState(false);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);

  // The interval reads the index and the setter off refs, so a page passing an
  // inline `onIndexChange` cannot restart the timer on every tick.
  const currentRef = useRef(current);
  currentRef.current = current;
  const setIndexRef = useRef(setIndex);
  setIndexRef.current = setIndex;

  useEffect(() => {
    if (!playing || count < 2) return;
    const timer = window.setInterval(() => {
      setIndexRef.current((currentRef.current + 1) % count);
    }, 1000 / CINE_FPS);
    return () => window.clearInterval(timer);
  }, [playing, count]);

  const step = useCallback(
    (delta: number) => {
      if (count === 0) return;
      setIndex(Math.min(Math.max(current + delta, 0), count - 1));
    },
    [count, current, setIndex],
  );

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "ArrowRight" || event.key === "ArrowDown") step(1);
    else if (event.key === "ArrowLeft" || event.key === "ArrowUp") step(-1);
    else if (event.key === "Home") setIndex(0);
    else if (event.key === "End") setIndex(Math.max(count - 1, 0));
    else if (event.key === " ") setPlaying((value) => !value);
    else return;
    event.preventDefault();
  };

  const size = slice ? sliceSize(slice.image) : null;

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
        detail={active?.description}
      />

      <div
        role="group"
        tabIndex={0}
        aria-label="Slice viewer. Arrow keys or the wheel step through the stack."
        onKeyDown={onKeyDown}
        onWheel={(event) => step(event.deltaY > 0 ? 1 : -1)}
        className="flex min-h-72 flex-1 items-center justify-center p-4 outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-accent"
      >
        {slice && size ? (
          // The canvas fills the stage and `object-contain` keeps the aspect,
          // so a 224² sheet tile scales up instead of sitting at native size.
          // Overlays pin to the stage corners, as viewport furniture rather
          // than image annotations.
          <div className="relative h-full w-full">
            <SliceCanvas
              image={slice.image}
              brightness={brightness}
              contrast={contrast}
              className="h-full w-full rounded-sm object-contain"
            />
            <div className="pointer-events-none absolute left-3 top-3 flex flex-col items-start gap-1">
              {active?.plane && (
                <span className="rounded bg-black/60 px-2 py-0.5 font-mono text-[11px] text-accent">
                  {active.plane}
                </span>
              )}
              <span className="rounded bg-black/60 px-2 py-0.5 font-mono text-[11px] text-white/50">
                {size.width} × {size.height}
              </span>
            </div>
            <div className="pointer-events-none absolute right-3 top-3">
              <span className="rounded bg-black/60 px-2 py-0.5 font-mono text-[11px] text-white/40">
                {slice.label ?? `Slice ${current + 1}`}
              </span>
            </div>
          </div>
        ) : (
          <ViewerPlaceholder label={emptyLabel} />
        )}
      </div>

      <div className="flex items-center gap-3 border-t border-white/5 bg-viewer-panel px-3 py-2">
        <button
          type="button"
          disabled={count < 2}
          onClick={() => setPlaying((value) => !value)}
          aria-label={playing ? "Pause" : "Play"}
          className="shrink-0 rounded-lg p-1.5 text-accent transition-colors hover:bg-white/5 disabled:pointer-events-none disabled:opacity-30"
        >
          <Icon name={playing ? "pause" : "play"} size={14} />
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(count - 1, 0)}
          value={current}
          disabled={count < 2}
          aria-label="Slice"
          onChange={(event) => setIndex(Number(event.target.value))}
          className="h-1 min-w-0 flex-1 accent-accent disabled:opacity-30"
        />
        <span className="w-16 shrink-0 text-right font-mono text-[11px] text-white/40">
          {count === 0 ? "—" : `${current + 1} / ${count}`}
        </span>
      </div>

      <div className="flex items-center gap-4 border-t border-white/5 bg-viewer-panel px-3 py-2">
        <ViewerSlider
          icon="brightness"
          label="Brightness"
          value={brightness}
          min={50}
          max={200}
          onChange={setBrightness}
        />
        <ViewerSlider
          icon="contrast"
          label="Contrast"
          value={contrast}
          min={50}
          max={250}
          onChange={setContrast}
        />
        <ViewerButton
          icon="reset"
          label="Reset"
          onClick={() => {
            setBrightness(100);
            setContrast(100);
          }}
        />
      </div>
    </section>
  );
}

export default Dicom2DViewer;
