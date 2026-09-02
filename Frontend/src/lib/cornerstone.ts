import { cache, volumeLoader, init as initCore, type Types } from "@cornerstonejs/core";
import { init as initTools } from "@cornerstonejs/tools";

import type { ModelVolume } from "@/interfaces";

let started: Promise<void> | null = null;

/**
 * Initialises Cornerstone once per page; safe to call from every viewer.
 *
 * No image loader here. Most of what the app draws never touches a DICOM file —
 * `Dicom3DViewer` builds its volume out of the model's tensor — so the loader
 * and its workers are left to `initDicomLoader` below.
 */
export function initCornerstone(): Promise<void> {
  started ??= (async () => {
    initCore();
    initTools();
  })();
  return started;
}

let dicomStarted: Promise<void> | null = null;

/**
 * Decoding is the expensive part of a full-resolution series, and it happens
 * off the main thread. Capped well under `hardwareConcurrency` because the
 * viewer shares the machine with whatever else the page is doing.
 */
const MAX_WORKERS = 4;

/**
 * The same, plus the DICOM image loader — for `DicomMprViewer`, the only thing
 * that reads the study's actual files.
 *
 * Imported dynamically, and that is the point rather than a detail: the loader
 * drags in four WASM codecs and a worker script, about 3 MB, and a reader who
 * never opens the reference view should never pay for them. A static import
 * would put all of it in the entry chunk.
 */
export function initDicomLoader(): Promise<void> {
  dicomStarted ??= (async () => {
    await initCornerstone();
    const { init } = await import("@cornerstonejs/dicom-image-loader");
    init({
      maxWebWorkers: Math.max(1, Math.min(navigator.hardwareConcurrency ?? 2, MAX_WORKERS)),
    });
  })();
  return dicomStarted;
}

/** The image ID for one DICOM file the app serves over HTTP. */
export function wadouriImageId(url: string): string {
  return `wadouri:${url}`;
}

/**
 * Voxel spacing for a tensor whose geometry has been preprocessed away.
 *
 * `sequence_to_tensor` resizes every slice to 224² and samples the series down
 * to 24 slices, whatever the field of view and the slice gap actually were, so
 * there are no millimetres left to read. What survives is a ratio: on a knee
 * the 24 slices span roughly the same physical extent as the in-plane field of
 * view, which puts the slice axis at `columns / slices` times the in-plane
 * spacing. Without it the volume renders as a flat wafer.
 *
 * Only a shape, never a size — a page holding the real spacing passes it.
 */
function assumedSpacing(shape: ModelVolume["shape"]): Types.Point3 {
  const [slices, , columns] = shape;
  return [1, 1, columns / slices];
}

/**
 * The model's input tensor as a Cornerstone volume, ready for a 3D viewport.
 *
 * Synchronous and idempotent: `createLocalVolume` hands back the cached volume
 * if this ID has been built before, so re-selecting a series costs nothing.
 *
 * @param spacingMm real `(slice, row, column)` spacing, when the caller knows it.
 */
export function buildModelVolume(
  volumeId: string,
  volume: ModelVolume,
  spacingMm?: [number, number, number],
): void {
  if (cache.getVolume(volumeId)) return;

  const [slices, rows, columns] = volume.shape;
  // The tensor is C-ordered `(slice, row, column)`, so columns vary fastest —
  // which is exactly the `(i, j, k)` a Cornerstone volume expects.
  const dimensions: Types.Point3 = [columns, rows, slices];
  const spacing: Types.Point3 = spacingMm
    ? [spacingMm[2], spacingMm[1], spacingMm[0]]
    : assumedSpacing(volume.shape);
  const direction: Types.Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const origin: Types.Point3 = [0, 0, 0];

  // The tensor goes in as it arrives, on its own [0, 1] scale, and the presets
  // in `Dicom3DViewer` are written against that scale rather than the raw MR
  // intensity Cornerstone's stock ones assume. Handed over rather than copied:
  // Cornerstone slices it into per-image views and only ever reads them, and a
  // copy would double what a multi-series study holds to no purpose.
  volumeLoader.createLocalVolume(volumeId, {
    scalarData: volume.data,
    dimensions,
    spacing,
    origin,
    direction,
    metadata: {
      BitsAllocated: 32,
      BitsStored: 32,
      HighBit: 31,
      SamplesPerPixel: 1,
      PixelRepresentation: 0,
      PhotometricInterpretation: "MONOCHROME2",
      Modality: "MR",
      ImageOrientationPatient: [1, 0, 0, 0, 1, 0],
      PixelSpacing: [spacing[1], spacing[0]],
      // The volume stands alone in its own space, so it is its own frame of
      // reference; Cornerstone only uses this to refuse to mix two volumes.
      FrameOfReferenceUID: volumeId,
      Rows: rows,
      Columns: columns,
      voiLut: [],
      VOILUTFunction: "LINEAR",
    },
  });
}

/** Drops a volume built here, if it is still cached. */
export function releaseVolume(volumeId: string): void {
  if (cache.getVolumeLoadObject(volumeId)) cache.removeVolumeLoadObject(volumeId);
}
