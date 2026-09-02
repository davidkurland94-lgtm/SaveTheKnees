import { useEffect, useId, useRef, useState } from "react";

import {
  cache,
  Enums,
  RenderingEngine,
  setVolumesForViewports,
  volumeLoader,
  type Types,
} from "@cornerstonejs/core";
import {
  addTool,
  Enums as ToolEnums,
  PanTool,
  StackScrollTool,
  ToolGroupManager,
  TrackballRotateTool,
  WindowLevelTool,
  ZoomTool,
} from "@cornerstonejs/tools";

import type { ViewerStack } from "@/interfaces";
import { cn, initCornerstone, useControlled } from "@/lib";
import { StackTabs, ViewerButton, ViewerPlaceholder } from "./ViewerChrome";

/** The viewer's own dark ground, as Cornerstone wants it: 0–1 RGB. */
const BACKGROUND: Types.Point3 = [0.031, 0.031, 0.063];

/**
 * Transfer functions Cornerstone ships. Only the MR ones are offered: the CT
 * presets are built around Hounsfield units, which MR intensities are not.
 */
const PRESETS = ["MR-Default", "MR-MIP", "MR-Angio", "MR-T2-Brain"] as const;

type Layout = "volume" | "mpr";

interface ViewportSpec {
  /** Suffix; the full ID is namespaced per component instance. */
  key: string;
  type: Enums.ViewportType;
  orientation?: Enums.OrientationAxis;
  label?: string;
}

const LAYOUTS: Record<Layout, { label: string; hint: string; viewports: ViewportSpec[] }> = {
  volume: {
    label: "Volume",
    hint: "drag to rotate · wheel to zoom · middle-drag to pan",
    viewports: [{ key: "3d", type: Enums.ViewportType.VOLUME_3D }],
  },
  mpr: {
    label: "MPR",
    hint: "wheel to scroll · drag for window/level · middle-drag to pan",
    viewports: [
      {
        key: "axial",
        type: Enums.ViewportType.ORTHOGRAPHIC,
        orientation: Enums.OrientationAxis.AXIAL,
        label: "Axial",
      },
      {
        key: "sagittal",
        type: Enums.ViewportType.ORTHOGRAPHIC,
        orientation: Enums.OrientationAxis.SAGITTAL,
        label: "Sagittal",
      },
      {
        key: "coronal",
        type: Enums.ViewportType.ORTHOGRAPHIC,
        orientation: Enums.OrientationAxis.CORONAL,
        label: "Coronal",
      },
    ],
  },
};

const { MouseBindings } = ToolEnums;

/** Which tools a layout binds, and to which button. */
const BINDINGS: Record<Layout, Array<{ tool: string; binding: number }>> = {
  volume: [
    { tool: TrackballRotateTool.toolName, binding: MouseBindings.Primary },
    { tool: PanTool.toolName, binding: MouseBindings.Auxiliary },
    { tool: ZoomTool.toolName, binding: MouseBindings.Secondary },
  ],
  mpr: [
    { tool: WindowLevelTool.toolName, binding: MouseBindings.Primary },
    { tool: PanTool.toolName, binding: MouseBindings.Auxiliary },
    { tool: ZoomTool.toolName, binding: MouseBindings.Secondary },
    { tool: StackScrollTool.toolName, binding: MouseBindings.Wheel },
  ],
};

/**
 * Registering a tool class is global and one-shot — a second call for the same
 * name warns — so it happens once here rather than per tool group.
 */
let toolsRegistered = false;

function registerTools(): void {
  if (toolsRegistered) return;
  for (const tool of [
    TrackballRotateTool,
    PanTool,
    ZoomTool,
    WindowLevelTool,
    StackScrollTool,
  ]) {
    addTool(tool);
  }
  toolsRegistered = true;
}

interface Dicom3DViewerProps {
  /**
   * The same stacks `Dicom2DViewer` takes — pass both the same array (and the
   * same `stackId`) to keep the two views on one series. This viewer reads
   * `imageIds`, not `slices`.
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
 * The series as a real volume, rendered by Cornerstone3D.
 *
 * This replaced a hand-rolled stack of translucent slice planes. That version
 * had only the 24 contact-sheet tiles to work with, so it had to guess at
 * everything a volume needs and could not know any of it: which way each plane
 * faces, which way the slices run, how many millimetres they span. Handing
 * Cornerstone the DICOM files instead means the geometry comes out of the
 * headers, and with it the two things a stack of PNGs can never do — a real
 * volume rendering, and reslicing the acquisition into the other two planes.
 *
 * Nothing is fetched here: the page supplies the image IDs and Cornerstone's
 * own loader streams the pixels behind them.
 */
export function Dicom3DViewer({
  stacks,
  stackId,
  defaultStackId,
  onStackChange,
  emptyLabel = "No volume to show.",
  className,
}: Dicom3DViewerProps) {
  const [activeId, setActiveId] = useControlled(
    stackId,
    defaultStackId ?? stacks[0]?.id ?? "",
    onStackChange,
  );
  const active = stacks.find((stack) => stack.id === activeId) ?? stacks[0];
  const imageIds = active?.imageIds ?? [];

  const [layout, setLayout] = useState<Layout>("volume");
  const [preset, setPreset] = useState<(typeof PRESETS)[number]>("MR-Default");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Bumped once the engine exists, so the viewport effect below re-runs against
  // it — the engine is created asynchronously and a ref does not re-render.
  const [engineGeneration, setEngineGeneration] = useState(0);

  const specs = LAYOUTS[layout].viewports;

  // Cornerstone's engines, tool groups and viewports are keyed by string in
  // module-level registries, so two mounted viewers would collide on a constant.
  const namespace = useId().replace(/[^a-zA-Z0-9]/g, "");
  const engineId = `stk-${namespace}`;
  const viewportIds = specs.map((spec) => `${engineId}-${spec.key}`);

  const containerRef = useRef<HTMLDivElement>(null);
  const elementRefs = useRef<Array<HTMLDivElement | null>>([]);
  const engineRef = useRef<RenderingEngine | null>(null);
  const builtVolumes = useRef(new Set<string>());

  // Keyed off the resolved stack, not the raw selection: with no `stackId` and
  // no `defaultStackId` the selection stays empty and `active` is the fallback
  // to the first stack, which is the one actually on screen.
  //
  // Cached under the series UID, so switching layout reuses the volume already
  // streamed instead of pulling the series down again.
  const volumeId = active?.id ? `stkVolume:${active.id}` : "";
  // The IDs are rebuilt on every render of the page above; compare by content
  // so an equal list does not tear the viewer down and rebuild it.
  const imageIdKey = imageIds.join("|");

  // One rendering engine for the component's lifetime, and with it one WebGL
  // context. Tearing it down per layout looked tidier but was wrong: the cached
  // volume keeps its textures on the context that uploaded them, so the engine
  // built for the next layout inherited textures it did not own and drew black.
  useEffect(() => {
    let engine: RenderingEngine | null = null;
    let cancelled = false;
    const volumes = builtVolumes.current;

    void (async () => {
      await initCornerstone();
      if (cancelled) return;
      registerTools();
      engine = new RenderingEngine(engineId);
      engineRef.current = engine;
      setEngineGeneration((count) => count + 1);
    })();

    return () => {
      cancelled = true;
      engine?.destroy();
      engineRef.current = null;
      // The volumes this viewer built go with it: their textures belong to the
      // context just destroyed. The per-frame images stay cached, so a remount
      // rebuilds from memory rather than off the network.
      for (const id of volumes) cache.removeVolumeLoadObject(id);
      volumes.clear();
    };
  }, [engineId]);

  useEffect(() => {
    const engine = engineRef.current;
    const elements = specs.map((_, position) => elementRefs.current[position]);
    if (!engine || !volumeId || imageIds.length === 0) return;
    if (elements.some((element) => !element)) return;

    let cancelled = false;
    const toolGroupId = `${engineId}-${layout}`;

    setLoading(true);
    setError(null);

    const build = async () => {
      try {
        engine.setViewports(
          specs.map((spec, position) => ({
            viewportId: viewportIds[position],
            type: spec.type,
            element: elements[position]!,
            defaultOptions: { background: BACKGROUND, orientation: spec.orientation },
          })),
        );

        const toolGroup = ToolGroupManager.createToolGroup(toolGroupId);
        for (const { tool, binding } of BINDINGS[layout]) {
          toolGroup?.addTool(tool);
          toolGroup?.setToolActive(tool, { bindings: [{ mouseButton: binding }] });
        }
        for (const viewportId of viewportIds) toolGroup?.addViewport(viewportId, engineId);

        // Awaited: this reads geometry off the first, middle and last instance,
        // which is what fixes the volume's shape and orientation.
        const volume = await volumeLoader.createAndCacheVolume(volumeId, { imageIds });
        if (cancelled) return;
        builtVolumes.current.add(volumeId);

        // Not awaited: the remaining frames stream in behind this, and the
        // viewports fill in as they land. Awaiting it would hold a blank canvas
        // until the last slice of a 300-image series arrived.
        if ("load" in volume) volume.load();

        await setVolumesForViewports(engine, [{ volumeId }], viewportIds);
        if (cancelled) return;

        if (layout === "volume") {
          const viewport = engine.getViewport<Types.IVolumeViewport>(viewportIds[0]);
          viewport?.setProperties({ preset });
        }
        engine.render();
        setLoading(false);
      } catch (cause) {
        if (cancelled) return;
        setLoading(false);
        setError(cause instanceof Error ? cause.message : "Could not build the volume.");
      }
    };

    void build();

    return () => {
      cancelled = true;
      ToolGroupManager.destroyToolGroup(toolGroupId);
    };
    // `specs` and `viewportIds` are derived from `layout`, and `preset` is
    // applied by the effect below — neither should rebuild the viewports.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engineId, volumeId, imageIdKey, layout, engineGeneration]);

  // Presets are a property of the volume actor, so they change without
  // rebuilding anything. `loading` is a dependency so the choice survives the
  // rebuild a layout switch causes.
  useEffect(() => {
    if (layout !== "volume" || loading) return;
    const viewport = engineRef.current?.getViewport<Types.IVolumeViewport>(viewportIds[0]);
    if (!viewport) return;
    viewport.setProperties({ preset });
    viewport.render();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset, layout, loading]);

  // Cornerstone draws into a fixed-size canvas; without this the volume is
  // stretched by whatever the flex layout does to the panel.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => engineRef.current?.resize(true, false));
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const ready = imageIds.length > 0 && !error;

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

      <div className="flex flex-wrap items-center gap-1 border-b border-white/5 bg-viewer-panel px-3 py-2">
        {(Object.keys(LAYOUTS) as Layout[]).map((id) => (
          <button
            key={id}
            type="button"
            aria-pressed={layout === id}
            onClick={() => setLayout(id)}
            className={cn(
              "rounded-lg px-3 py-1 text-[11px] font-semibold transition-colors",
              layout === id
                ? "bg-accent/20 text-accent"
                : "text-white/40 hover:bg-white/5 hover:text-white/70",
            )}
          >
            {LAYOUTS[id].label}
          </button>
        ))}
        {layout === "volume" && (
          <select
            aria-label="Rendering preset"
            value={preset}
            onChange={(event) => setPreset(event.target.value as (typeof PRESETS)[number])}
            className="ml-auto rounded-lg bg-white/5 px-2 py-1 text-[11px] text-white/60 outline-none"
          >
            {PRESETS.map((name) => (
              <option key={name} value={name} className="bg-viewer-panel">
                {name}
              </option>
            ))}
          </select>
        )}
      </div>

      <div ref={containerRef} className="relative min-h-72 flex-1 bg-viewer">
        {ready ? (
          <>
            <div
              className={cn(
                "absolute inset-0 grid gap-px",
                layout === "mpr" ? "grid-cols-1 sm:grid-cols-3" : "grid-cols-1",
              )}
            >
              {specs.map((spec, position) => (
                <div key={spec.key} className="relative min-h-0 min-w-0">
                  {/* Cornerstone owns this element's contents; React must not
                      re-parent the canvas it puts inside, hence no children. */}
                  <div
                    ref={(element) => {
                      elementRefs.current[position] = element;
                    }}
                    onContextMenu={(event) => event.preventDefault()}
                    className="h-full w-full touch-none select-none"
                  />
                  {spec.label && (
                    <span className="pointer-events-none absolute left-2 top-2 rounded bg-black/60 px-2 py-0.5 font-mono text-[11px] text-accent">
                      {spec.label}
                    </span>
                  )}
                </div>
              ))}
            </div>

            {loading && (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-viewer/70">
                <span className="rounded bg-black/70 px-3 py-1.5 font-mono text-[11px] text-white/60">
                  Streaming {imageIds.length} slices…
                </span>
              </div>
            )}

            <div className="pointer-events-none absolute right-3 top-3 flex flex-col items-end gap-1">
              {active?.plane && (
                <span className="rounded bg-black/60 px-2 py-0.5 font-mono text-[11px] text-accent">
                  {active.plane} acquisition
                </span>
              )}
              <span className="rounded bg-black/60 px-2 py-0.5 font-mono text-[11px] text-white/40">
                {imageIds.length} slices
              </span>
            </div>
          </>
        ) : (
          <div className="flex h-full items-center justify-center">
            <ViewerPlaceholder label={error ?? emptyLabel} />
          </div>
        )}
      </div>

      <div className="flex items-center gap-4 border-t border-white/5 bg-viewer-panel px-3 py-2">
        <span className="min-w-0 flex-1 truncate text-[11px] text-white/25">
          {LAYOUTS[layout].hint}
        </span>
        <ViewerButton
          icon="reset"
          label="Reset view"
          disabled={!ready}
          onClick={() => {
            const engine = engineRef.current;
            if (!engine) return;
            for (const viewportId of viewportIds) {
              const viewport = engine.getViewport<Types.IVolumeViewport>(viewportId);
              viewport?.resetCamera();
              if (layout === "volume") viewport?.setProperties({ preset });
            }
            engine.render();
          }}
        />
      </div>
    </section>
  );
}

export default Dicom3DViewer;
