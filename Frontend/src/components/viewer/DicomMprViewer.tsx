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
  AngleTool,
  annotation,
  ArrowAnnotateTool,
  CrosshairsTool,
  EllipticalROITool,
  Enums as ToolEnums,
  LengthTool,
  PanTool,
  ProbeTool,
  StackScrollTool,
  ToolGroupManager,
  WindowLevelTool,
  ZoomTool,
} from "@cornerstonejs/tools";

import type { ViewerStack } from "@/interfaces";
import { cn, initDicomLoader, useControlled } from "@/lib";
import {
  ChromeDivider,
  SegmentedTabs,
  StackTabs,
  ViewerButton,
  ViewerPlaceholder,
  type SegmentedOption,
  type ViewSwitch,
} from "./ViewerChrome";

/** The viewer's own dark ground, as Cornerstone wants it: 0–1 RGB. */
const BACKGROUND: Types.Point3 = [0.031, 0.031, 0.063];

/**
 * The three reformats, each with the colour its reference line is drawn in
 * everywhere else. The colour is the whole point of the crosshairs: the green
 * line crossing the axial pane is the sagittal plane, so a reader can see which
 * pane they are about to scroll before they touch it.
 */
const PLANES = [
  {
    key: "axial",
    orientation: Enums.OrientationAxis.AXIAL,
    label: "Axial",
    line: "rgb(240, 100, 100)",
  },
  {
    key: "sagittal",
    orientation: Enums.OrientationAxis.SAGITTAL,
    label: "Sagittal",
    line: "rgb(110, 210, 140)",
  },
  {
    key: "coronal",
    orientation: Enums.OrientationAxis.CORONAL,
    label: "Coronal",
    line: "rgb(120, 165, 250)",
  },
] as const;

/**
 * What the left mouse button does, one at a time.
 *
 * Locate is first and is the default, because it is the answer to the question
 * this view exists for: click a point in any plane and the other two jump to
 * it. The measuring tools below it are only honest here — they read the
 * millimetres out of the DICOM headers, which is why they are not offered on
 * the model volume, whose spacing is an assumption.
 */
const TOOLS: ReadonlyArray<SegmentedOption<string>> = [
  {
    id: CrosshairsTool.toolName,
    label: "Locate",
    hint: "Drag the centre; all three planes follow",
  },
  {
    id: WindowLevelTool.toolName,
    label: "W/L",
    hint: "Drag to window and level",
  },
  {
    id: LengthTool.toolName,
    label: "Length",
    hint: "Distance between two points, in mm",
  },
  { id: AngleTool.toolName, label: "Angle", hint: "Angle between two lines" },
  {
    id: EllipticalROITool.toolName,
    label: "ROI",
    hint: "Ellipse: area and mean signal",
  },
  { id: ProbeTool.toolName, label: "Probe", hint: "Signal value at one point" },
  {
    id: ArrowAnnotateTool.toolName,
    label: "Note",
    hint: "Arrow with a typed label",
  },
];

const { MouseBindings } = ToolEnums;

/** Bound once and never changed, so the palette only has to own the left button. */
const FIXED_BINDINGS = [
  { tool: PanTool.toolName, binding: MouseBindings.Auxiliary },
  { tool: ZoomTool.toolName, binding: MouseBindings.Secondary },
  { tool: StackScrollTool.toolName, binding: MouseBindings.Wheel },
];

/**
 * Registering a tool class is global and one-shot — a second call for the same
 * name warns — so it happens once here rather than per tool group.
 */
let toolsRegistered = false;

function registerTools(): void {
  if (toolsRegistered) return;
  for (const tool of [
    CrosshairsTool,
    WindowLevelTool,
    LengthTool,
    AngleTool,
    EllipticalROITool,
    ProbeTool,
    ArrowAnnotateTool,
    PanTool,
    ZoomTool,
    StackScrollTool,
  ]) {
    addTool(tool);
  }
  toolsRegistered = true;
}

interface DicomMprViewerProps {
  /**
   * The same stacks the other two viewers take. This one reads `imageIds`, so a
   * stack without them shows `emptyLabel` — which is the normal state until the
   * page decides this view is on screen and goes to fetch them.
   */
  stacks: ViewerStack[];
  stackId?: string;
  defaultStackId?: string;
  onStackChange?: (stackId: string) => void;
  /** The page's panel switch, drawn ahead of this viewer's own controls. */
  views?: ViewSwitch;
  emptyLabel?: string;
  className?: string;
}

/**
 * The study as the scanner wrote it: three reformats of one volume, linked.
 *
 * This is the counterweight to `Dicom3DViewer`. That one renders the model's
 * input, which is the right thing to look at when the question is what the
 * scores were computed from — but the preprocessing resamples to 24 slices at
 * 224 square and drops the geometry, so a distance measured on it would be a
 * number with no unit behind it, and a point marked in one plane could not be
 * placed in another. Here the DICOM headers are intact, so both work: the
 * crosshairs tie the planes together through the frame of reference, and a
 * length is a length.
 *
 * Nothing is fetched here either. The page supplies the image IDs; Cornerstone's
 * loader streams the pixels behind them.
 */
export function DicomMprViewer({
  stacks,
  stackId,
  defaultStackId,
  onStackChange,
  views,
  emptyLabel = "No series to show.",
  className,
}: DicomMprViewerProps) {
  const [activeId, setActiveId] = useControlled(
    stackId,
    defaultStackId ?? stacks[0]?.id ?? "",
    onStackChange,
  );
  const active = stacks.find((stack) => stack.id === activeId) ?? stacks[0];
  const imageIds = active?.imageIds ?? [];

  const [tool, setTool] = useState<string>(CrosshairsTool.toolName);
  const [loading, setLoading] = useState(false);
  const [built, setBuilt] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Bumped once the engine exists, so the viewport effect below re-runs against
  // it — the engine is created asynchronously and a ref does not re-render.
  const [engineGeneration, setEngineGeneration] = useState(0);

  // Cornerstone's engines, tool groups and viewports are keyed by string in
  // module-level registries, so two mounted viewers would collide on a constant.
  const namespace = useId().replace(/[^a-zA-Z0-9]/g, "");
  const engineId = `stkref-${namespace}`;
  const toolGroupId = `${engineId}-tools`;
  const viewportIds = PLANES.map((plane) => `${engineId}-${plane.key}`);
  const lineColours = new Map(viewportIds.map((id, at) => [id, PLANES[at].line]));

  const containerRef = useRef<HTMLDivElement>(null);
  const elementRefs = useRef<Array<HTMLDivElement | null>>([]);
  const engineRef = useRef<RenderingEngine | null>(null);
  const builtVolumes = useRef(new Set<string>());

  // A scheme is wanted here, unlike on the model volume: the slices behind it
  // are real `wadouri:` image IDs, so Cornerstone's hunt for a default window
  // finds a loader that can actually satisfy it, and the series arrives
  // windowed the way its headers say rather than the way we guess.
  const volumeId = active?.id ? `stkRef:${active.id}` : "";
  // The IDs are rebuilt on every render of the page above; compare by content
  // so an equal list does not tear the viewer down and rebuild it.
  const imageIdKey = imageIds.join("|");

  // One rendering engine for the component's lifetime, and with it one WebGL
  // context — cached volumes keep their textures on the context that uploaded
  // them, so a replacement engine would inherit textures it does not own.
  useEffect(() => {
    let engine: RenderingEngine | null = null;
    let cancelled = false;
    const volumes = builtVolumes.current;

    void (async () => {
      await initDicomLoader();
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
      for (const id of volumes) {
        if (cache.getVolumeLoadObject(id)) cache.removeVolumeLoadObject(id);
      }
      volumes.clear();
    };
  }, [engineId]);

  useEffect(() => {
    const engine = engineRef.current;
    const elements = PLANES.map((_, at) => elementRefs.current[at]);
    if (!engine || !volumeId || imageIds.length === 0) return;
    if (elements.some((element) => !element)) return;

    let cancelled = false;

    setLoading(true);
    setBuilt(false);
    setError(null);

    const build = async () => {
      try {
        engine.setViewports(
          PLANES.map((plane, at) => ({
            viewportId: viewportIds[at],
            type: Enums.ViewportType.ORTHOGRAPHIC,
            element: elements[at]!,
            defaultOptions: {
              background: BACKGROUND,
              orientation: plane.orientation,
            },
          })),
        );

        const toolGroup = ToolGroupManager.createToolGroup(toolGroupId);
        if (!toolGroup) throw new Error("Could not create the tool group.");

        // The crosshairs need to know how to draw each plane's line; every
        // other tool is added with its defaults.
        toolGroup.addTool(CrosshairsTool.toolName, {
          getReferenceLineColor: (viewportId: string) =>
            lineColours.get(viewportId) ?? "rgb(200, 200, 200)",
          getReferenceLineControllable: () => true,
          getReferenceLineDraggableRotatable: () => true,
          getReferenceLineSlabThicknessControlsOn: () => false,
        });
        for (const entry of TOOLS) {
          if (entry.id !== CrosshairsTool.toolName) toolGroup.addTool(entry.id);
        }
        for (const { tool: name, binding } of FIXED_BINDINGS) {
          toolGroup.addTool(name);
          toolGroup.setToolActive(name, {
            bindings: [{ mouseButton: binding }],
          });
        }
        for (const viewportId of viewportIds) toolGroup.addViewport(viewportId, engineId);

        // Awaited: this reads geometry off the first, middle and last instance,
        // which is what fixes the volume's shape and orientation.
        const volume = await volumeLoader.createAndCacheVolume(volumeId, {
          imageIds,
        });
        if (cancelled) return;
        builtVolumes.current.add(volumeId);

        // Not awaited: the remaining frames stream in behind this and the panes
        // fill as they land. Awaiting it would hold three blank canvases until
        // the last slice of a 300-image series arrived.
        if ("load" in volume) volume.load();

        await setVolumesForViewports(engine, [{ volumeId }], viewportIds);
        if (cancelled) return;

        engine.render();
        setLoading(false);
        setBuilt(true);
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
    // `viewportIds` and `lineColours` are derived from `engineId`, and the tool
    // palette is applied by the effect below — none should rebuild the panes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engineId, volumeId, imageIdKey, engineGeneration]);

  // Only the left button changes hands. The others stay where the build put
  // them, and every tool not currently driving it is left passive rather than
  // disabled, so annotations already on screen stay selectable and the
  // crosshairs keep drawing their reference lines while a measurement is taken.
  useEffect(() => {
    const toolGroup = ToolGroupManager.getToolGroup(toolGroupId);
    if (!toolGroup || !built) return;
    for (const entry of TOOLS) {
      if (entry.id === tool) {
        toolGroup.setToolActive(entry.id, {
          bindings: [{ mouseButton: MouseBindings.Primary }],
        });
      } else {
        toolGroup.setToolPassive(entry.id);
      }
    }
  }, [tool, built, toolGroupId]);

  // Cornerstone draws into a fixed-size canvas; without this the panes are
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
        {views && (
          <>
            <SegmentedTabs {...views} />
            <ChromeDivider />
          </>
        )}
        <SegmentedTabs options={TOOLS} active={tool} onSelect={setTool} />
      </div>

      <div ref={containerRef} className="relative min-h-72 flex-1 bg-viewer">
        {ready ? (
          <>
            <div className="absolute inset-0 grid grid-cols-1 gap-px sm:grid-cols-3">
              {PLANES.map((plane, at) => (
                <div key={plane.key} className="relative min-h-0 min-w-0">
                  {/* Cornerstone owns this element's contents; React must not
                      re-parent the canvas it puts inside, hence no children. */}
                  <div
                    ref={(element) => {
                      elementRefs.current[at] = element;
                    }}
                    onContextMenu={(event) => event.preventDefault()}
                    className="h-full w-full touch-none select-none"
                  />
                  <span
                    className="pointer-events-none absolute left-2 top-2 rounded bg-black/60 px-2 py-0.5 font-mono text-[11px]"
                    style={{ color: plane.line }}
                  >
                    {plane.label}
                  </span>
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

            <div className="pointer-events-none absolute right-3 top-3">
              <span className="rounded bg-black/60 px-2 py-0.5 font-mono text-[11px] text-white/40">
                {imageIds.length} slices · full resolution
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
          study geometry · wheel to scroll · middle-drag to pan · right-drag to zoom
        </span>
        <ViewerButton
          label="Clear marks"
          disabled={!built}
          onClick={() => {
            annotation.state.removeAllAnnotations();
            engineRef.current?.render();
          }}
        />
        <ViewerButton
          icon="reset"
          label="Reset view"
          disabled={!built}
          onClick={() => {
            const engine = engineRef.current;
            if (!engine) return;
            for (const viewportId of viewportIds) {
              engine.getViewport<Types.IVolumeViewport>(viewportId)?.resetCamera();
            }
            const crosshairs = ToolGroupManager.getToolGroup(toolGroupId)?.getToolInstance(
              CrosshairsTool.toolName,
            ) as CrosshairsTool | undefined;
            crosshairs?.resetCrosshairs();
            engine.render();
          }}
        />
      </div>
    </section>
  );
}

export default DicomMprViewer;
