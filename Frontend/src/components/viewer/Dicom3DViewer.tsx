import { useEffect, useId, useRef, useState } from "react";

import {
  CONSTANTS,
  Enums,
  RenderingEngine,
  setVolumesForViewports,
  type Types,
} from "@cornerstonejs/core";
import {
  addTool,
  Enums as ToolEnums,
  PanTool,
  ToolGroupManager,
  TrackballRotateTool,
  ZoomTool,
} from "@cornerstonejs/tools";

import type { ViewerStack } from "@/interfaces";
import { buildModelVolume, cn, initCornerstone, releaseVolume, useControlled } from "@/lib";
import {
  ChromeDivider,
  SegmentedTabs,
  StackTabs,
  ViewerButton,
  ViewerPlaceholder,
  type ViewSwitch,
} from "./ViewerChrome";

/** The viewer's own dark ground, as Cornerstone wants it: 0–1 RGB. */
const BACKGROUND: Types.Point3 = [0.031, 0.031, 0.063];

/**
 * Why none of Cornerstone's own presets are used.
 *
 * They are written against raw MR intensity and their ramps are steep and
 * early: MR-Default reaches its last opacity stop at 220 of 1024, MR-MIP goes
 * fully opaque at 417. The tensor is percentile-windowed to [0, 1], and on a
 * knee series about 36% of it is air below 0.05 while 57% is tissue spread
 * between 0.20 and 0.70. Put one against the other and every tissue voxel lands
 * in the flat top of the ramp at the same opacity, with the air showing through
 * as haze — which is why the four presets that used to be on offer all looked
 * alike, and all looked bad.
 *
 * So the ramps below are cut to that histogram instead. `applyPreset` maps a
 * preset's stops straight onto the scalar values, so these read in tensor
 * units: 0 is air and 1 is the 99.5th percentile of the series.
 */

/**
 * Anatomy. Air is held fully transparent past the background spike, and opacity
 * then climbs the whole way across the tissue band rather than saturating at
 * the bottom of it, so density differences read as depth. Gradient opacity
 * fades homogeneous interiors and keeps boundaries, which is what turns a blob
 * into a surface.
 */
const SURFACE: Types.ViewportPreset = {
  name: "stk-surface",
  scalarOpacity: "16 0 0 0.07 0 0.2 0.03 0.33 0.1 0.47 0.25 0.62 0.55 0.8 0.85 1 1",
  colorTransfer:
    "24 0 0 0 0 0.2 0.25 0.12 0.08 0.37 0.62 0.36 0.24 0.53 0.85 0.64 0.47 0.7 0.95 0.86 0.75 1 1 1 1",
  gradientOpacity: "4 0 0.35 0.2 1",
  shade: "1",
  ambient: "0.3",
  diffuse: "0.8",
  specular: "0.15",
  specularPower: "10",
  interpolation: "1",
};

/**
 * The bright quarter of the volume, and nothing else: everything below the 75th
 * percentile is dropped outright, so what is left on a fluid-sensitive sequence
 * is fluid, marrow oedema and vessel. Unshaded and cool-tinted so it cannot be
 * confused with the anatomy view at a glance.
 *
 * Named for what it shows rather than "MIP", which it is not — a true
 * maximum-intensity blend is unreachable here: `VolumeViewport3D.setBlendMode`
 * returns null in @cornerstonejs/core 5.8, as does `setSlabThickness`.
 */
const FLUID: Types.ViewportPreset = {
  name: "stk-fluid",
  scalarOpacity: "12 0 0 0.47 0 0.64 0.12 0.78 0.5 0.9 0.9 1 1",
  colorTransfer: "20 0 0 0 0 0.47 0.15 0.25 0.4 0.68 0.45 0.7 0.95 0.86 0.85 0.95 1 1 1 1 1",
  gradientOpacity: "4 0 1 1 1",
  shade: "0",
  ambient: "1",
  diffuse: "0",
  specular: "0",
  specularPower: "1",
  interpolation: "1",
};

const MODES = [
  { id: SURFACE.name, label: "Surface", hint: "Shaded anatomy" },
  { id: FLUID.name, label: "Fluid", hint: "The brightest quarter only" },
] as const;

type Mode = (typeof MODES)[number]["id"];

/**
 * Presets are addressed by name, and the name is looked up in the list
 * Cornerstone ships, so ours have to join it. The alternative is passing the
 * preset object, which `setPreset` accepts at runtime but `setProperties` does
 * not admit in its types.
 */
function registerPresets(): void {
  for (const preset of [SURFACE, FLUID]) {
    if (!CONSTANTS.VIEWPORT_PRESETS.some((entry) => entry.name === preset.name)) {
      CONSTANTS.VIEWPORT_PRESETS.push(preset);
    }
  }
}

const { KeyboardBindings, MouseBindings } = ToolEnums;

/**
 * How the volume is flown around.
 *
 * Every one of the three has a route that needs nothing but a left button and a
 * two-finger scroll, because that is all a trackpad has. The middle- and
 * right-button bindings are kept beside them for anyone on a real mouse, and
 * cost nothing: Cornerstone matches a binding on button *and* modifier, so
 * plain-drag and shift-drag resolve to different tools without ambiguity.
 */
const BINDINGS = [
  { tool: TrackballRotateTool.toolName, bindings: [{ mouseButton: MouseBindings.Primary }] },
  {
    tool: PanTool.toolName,
    bindings: [
      { mouseButton: MouseBindings.Primary, modifierKey: KeyboardBindings.Shift },
      { mouseButton: MouseBindings.Auxiliary },
    ],
  },
  {
    tool: ZoomTool.toolName,
    bindings: [
      // The one that was missing. The footer had promised wheel-zoom since this
      // viewer was written, but nothing was ever bound to `Wheel`, so scrolling
      // over the volume did nothing at all.
      { mouseButton: MouseBindings.Wheel },
      // Alt rather than Ctrl: macOS delivers ctrl+left-click as a *secondary*
      // click carrying ctrlKey, so a Primary+Ctrl binding would never match
      // there — and neither would the plain Secondary one below it, whose
      // modifier is undefined. Alt is remapped by no platform.
      { mouseButton: MouseBindings.Primary, modifierKey: KeyboardBindings.Alt },
      { mouseButton: MouseBindings.Secondary },
    ],
  },
];

/**
 * Registering a tool class is global and one-shot — a second call for the same
 * name warns — so it happens once here rather than per tool group.
 */
let toolsRegistered = false;

function registerTools(): void {
  if (toolsRegistered) return;
  for (const tool of [TrackballRotateTool, PanTool, ZoomTool]) addTool(tool);
  registerPresets();
  toolsRegistered = true;
}

interface Dicom3DViewerProps {
  /**
   * The same stacks `DicomMprViewer` takes — pass both the same array (and the
   * same `stackId`) to keep the two views on one series. This viewer reads
   * `volume`, the reference view reads `imageIds`.
   */
  stacks: ViewerStack[];
  /** Controlled active stack; omit and the viewer owns the selection. */
  stackId?: string;
  defaultStackId?: string;
  onStackChange?: (stackId: string) => void;
  /** The page's panel switch, drawn ahead of this viewer's own controls. */
  views?: ViewSwitch;
  emptyLabel?: string;
  className?: string;
}

/**
 * The model's input as a volume, rendered by Cornerstone3D.
 *
 * It used to build from the raw DICOM files instead, streamed slice by slice
 * out of `/studies/{uid}/series/{uid}/instances`. That gave a geometrically
 * honest volume of a series the model never sees: full resolution where the
 * network gets 224², every slice where the network gets 24 of them, and each
 * series windowed on its own header rather than on the one percentile window
 * the preprocessing applies. Two viewers side by side disagreed about what the
 * study contained, and the 3D one was the one disagreeing with the scores in
 * the rail. It now renders the tensor, so what is on screen is what was
 * classified.
 *
 * What that costs is the geometry: the tensor carries no millimetres, so the
 * volume is drawn to an assumed aspect (see `buildModelVolume`) and there is no
 * reslicing on offer — a reformat cut across 24 sampled slices would be a worse
 * answer than the acquisition tabs above already give.
 *
 * Nothing is fetched here: the page supplies the tensor.
 */
export function Dicom3DViewer({
  stacks,
  stackId,
  defaultStackId,
  onStackChange,
  views,
  emptyLabel = "No volume to show.",
  className,
}: Dicom3DViewerProps) {
  const [activeId, setActiveId] = useControlled(
    stackId,
    defaultStackId ?? stacks[0]?.id ?? "",
    onStackChange,
  );
  const active = stacks.find((stack) => stack.id === activeId) ?? stacks[0];
  const volume = active?.volume;

  const [mode, setMode] = useState<Mode>(MODES[0].id);
  const [built, setBuilt] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Bumped once the engine exists, so the viewport effect below re-runs against
  // it — the engine is created asynchronously and a ref does not re-render.
  const [engineGeneration, setEngineGeneration] = useState(0);

  // Cornerstone's engines, tool groups and viewports are keyed by string in
  // module-level registries, so two mounted viewers would collide on a constant.
  const namespace = useId().replace(/[^a-zA-Z0-9]/g, "");
  const engineId = `stk-${namespace}`;
  const viewportId = `${engineId}-3d`;

  const containerRef = useRef<HTMLDivElement>(null);
  const elementRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<RenderingEngine | null>(null);
  const builtVolumes = useRef(new Set<string>());

  // Keyed off the resolved stack, not the raw selection: with no `stackId` and
  // no `defaultStackId` the selection stays empty and `active` is the fallback
  // to the first stack, which is the one actually on screen.
  //
  // Cached under the series UID, so returning to a series reuses the volume
  // already uploaded to the GPU instead of rebuilding it.
  //
  // Deliberately colon-free. Cornerstone reads a scheme out of everything before
  // the first colon and, when it decides a volume's slices look like real image
  // IDs, hunts for a loader that can re-read the middle one to pick a default
  // window — a load this volume's slices could never satisfy, since they were
  // put straight into the cache and no loader owns them. A prefix with no colon
  // is what makes it skip that, which is right anyway: the preset writes the
  // transfer function, so there is no window to guess at.
  const volumeId = active?.id ? `stkVolume-${active.id}` : "";

  // One rendering engine for the component's lifetime, and with it one WebGL
  // context. Tearing it down per selection looked tidier but was wrong: the
  // cached volume keeps its textures on the context that uploaded them, so a
  // later engine inherited textures it did not own and drew black.
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
      // context just destroyed.
      for (const id of volumes) releaseVolume(id);
      volumes.clear();
    };
  }, [engineId]);

  useEffect(() => {
    const engine = engineRef.current;
    const element = elementRef.current;
    if (!engine || !element || !volumeId || !volume) return;

    let cancelled = false;
    const toolGroupId = `${engineId}-tools`;

    setBuilt(false);
    setError(null);

    const build = async () => {
      try {
        engine.setViewports([
          {
            viewportId,
            type: Enums.ViewportType.VOLUME_3D,
            element,
            defaultOptions: { background: BACKGROUND },
          },
        ]);

        const toolGroup = ToolGroupManager.createToolGroup(toolGroupId);
        for (const { tool, bindings } of BINDINGS) {
          toolGroup?.addTool(tool);
          toolGroup?.setToolActive(tool, { bindings });
        }
        toolGroup?.addViewport(viewportId, engineId);

        // Synchronous: the page already holds every voxel, so there is nothing
        // to stream and no partial state to paint around.
        buildModelVolume(volumeId, volume, active?.spacingMm);
        builtVolumes.current.add(volumeId);

        await setVolumesForViewports(engine, [{ volumeId }], [viewportId]);
        if (cancelled) return;

        engine.getViewport<Types.IVolumeViewport>(viewportId)?.setProperties({ preset: mode });
        engine.render();
        setBuilt(true);
      } catch (cause) {
        if (cancelled) return;
        setError(cause instanceof Error ? cause.message : "Could not build the volume.");
      }
    };

    void build();

    return () => {
      cancelled = true;
      ToolGroupManager.destroyToolGroup(toolGroupId);
    };
    // `viewportId` is derived from `engineId`, and `mode` is applied by the
    // effect below — neither should rebuild the viewport.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engineId, volumeId, volume, active?.spacingMm, engineGeneration]);

  // The preset is a property of the volume actor, so switching mode changes
  // nothing else. Gated on `built` because there is no actor to set it on until
  // the effect above has finished.
  useEffect(() => {
    if (!built) return;
    const viewport = engineRef.current?.getViewport<Types.IVolumeViewport>(viewportId);
    if (!viewport) return;
    viewport.setProperties({ preset: mode });
    viewport.render();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, built]);

  // Cornerstone draws into a fixed-size canvas; without this the volume is
  // stretched by whatever the flex layout does to the panel.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => engineRef.current?.resize(true, false));
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const ready = Boolean(volume) && !error;

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
        <SegmentedTabs options={MODES} active={mode} onSelect={setMode} />
      </div>

      <div ref={containerRef} className="relative min-h-72 flex-1 bg-viewer">
        {ready ? (
          <>
            {/* Cornerstone owns this element's contents; React must not
                re-parent the canvas it puts inside, hence no children. */}
            <div
              ref={elementRef}
              onContextMenu={(event) => event.preventDefault()}
              className="absolute inset-0 touch-none select-none"
            />

            {!built && (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-viewer/70">
                <span className="rounded bg-black/70 px-3 py-1.5 font-mono text-[11px] text-white/60">
                  Building the volume…
                </span>
              </div>
            )}

            <div className="pointer-events-none absolute right-3 top-3 flex flex-col items-end gap-1">
              {active?.plane && (
                <span className="rounded bg-black/60 px-2 py-0.5 font-mono text-[11px] text-accent">
                  {active.plane} acquisition
                </span>
              )}
              {/* The tensor's own shape, next to the series' real slice count in
                  the tabs above: together they say what the preprocessing did. */}
              <span className="rounded bg-black/60 px-2 py-0.5 font-mono text-[11px] text-white/40">
                {volume?.shape.join(" × ")}
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
          model input · drag to rotate · scroll to zoom · shift-drag to pan
        </span>
        <ViewerButton
          icon="reset"
          label="Reset view"
          disabled={!built}
          onClick={() => {
            const engine = engineRef.current;
            const viewport = engine?.getViewport<Types.IVolumeViewport>(viewportId);
            if (!engine || !viewport) return;
            viewport.resetCamera();
            viewport.setProperties({ preset: mode });
            engine.render();
          }}
        />
      </div>
    </section>
  );
}

export default Dicom3DViewer;
